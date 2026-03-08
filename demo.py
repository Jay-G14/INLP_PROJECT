"""
demo.py — Terminal Toucher's LLM
A Textual TUI demo for the Knowledge Unlearning via SAE project.

Features:
  - Loads GPT-2 Medium via TransformerLens
  - Loads pre-trained SAE from sae_layer_12.pt
  - Auto-discovers HP-specific features (fast mini diff-means) and caches them
  - Toggle SAE feature ablation on/off to see knowledge unlearning in action
  - White & yellow theme, scrollable chat interface
"""

import os
import sys
import asyncio
import threading
from pathlib import Path
from typing import List, Optional

# ── path setup so src.* imports work ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Input,
    Button,
    Static,
    LoadingIndicator,
    Label,
    RichLog,
)
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual import work, on
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.console import Group

# ── project imports ────────────────────────────────────────────────────────────
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook

# ── constants ──────────────────────────────────────────────────────────────────
SAE_PATH        = ROOT / "sae_layer_12.pt"
FEATURES_CACHE  = ROOT / "results" / "layer_12_features.pt"
HP_CORPUS_TXT   = ROOT / "Harry_Potter_all_books_preprocessed.txt"
MODEL_NAME      = "gpt2-medium"
LAYER           = 12
K               = 32
EXPANSION       = 16
NUM_FEATURES    = 100
ABLATION_SCALE  = -3.0
MAX_NEW_TOKENS  = 120
TEMPERATURE     = 0.7
TOP_P           = 0.9
FREQ_PENALTY    = 1.0

# A handful of Harry Potter anchor texts used for fast feature discovery
HP_ANCHORS = [
    "Harry Potter and the Philosopher's Stone. Dumbledore, Hermione, Ron Weasley at Hogwarts.",
    "Lord Voldemort raised his wand and cast Avada Kedavra. The Death Eaters surrounded Hogwarts.",
    "Quidditch match on broomsticks. Gryffindor versus Slytherin. Harry caught the Golden Snitch.",
    "Professor Snape looked at Harry coldly. Potions class in the Hogwarts dungeons.",
    "The Sorting Hat placed Hermione into Gryffindor. Dumbledore smiled from the high table.",
    "Hagrid knocked on the door of number four Privet Drive and handed Harry his letter.",
]
NEUTRAL_ANCHORS = [
    "The capital of France is Paris, located on the Seine river in northern Europe.",
    "Photosynthesis is the process by which plants convert sunlight to chemical energy.",
    "The American Civil War began in 1861 and ended in 1865 with Union victory.",
    "Machine learning algorithms learn patterns from large datasets without explicit programming.",
    "The mitochondria are organelles that produce ATP energy inside eukaryotic cells.",
    "Shakespeare wrote Hamlet, King Lear, and Macbeth in the early seventeenth century.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Model manager (runs entirely on a background thread)
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Holds GPT-2 Medium + SAE, handles feature discovery and generation."""

    def __init__(self):
        self.model = None
        self.sae: Optional[TopKSAE] = None
        self.feature_indices: Optional[torch.Tensor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False
        self.status_cb = None          # callable(str) for status updates
        self.ablation_enabled = False

    # ── loading ────────────────────────────────────────────────────────────────
    def load(self, status_cb):
        self.status_cb = status_cb
        try:
            self._load_model()
            self._load_sae()
            self._load_or_discover_features()
            self.ready = True
            self.status_cb("✅  Model ready")
        except Exception as e:
            self.status_cb(f"❌  Load error: {e}")

    def _load_model(self):
        self.status_cb(f"⏳  Loading {MODEL_NAME} via TransformerLens…")
        from transformer_lens import HookedTransformer
        self.model = HookedTransformer.from_pretrained(MODEL_NAME, device=self.device)
        self.model.eval()
        self.status_cb(f"✅  {MODEL_NAME} loaded on {self.device}")

    def _load_sae(self):
        self.status_cb("⏳  Loading SAE from sae_layer_12.pt…")
        d_model = self.model.cfg.d_model          # 1024 for gpt2-medium
        d_sae   = d_model * EXPANSION              # 16384
        sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=K)
        state = torch.load(str(SAE_PATH), map_location=self.device, weights_only=True)
        sae.load_state_dict(state)
        sae = sae.float()   # SAE is fp16 (set by model.py import); cast to fp32 to match TransformerLens activations
        sae.to(self.device)
        sae.eval()
        self.sae = sae
        self.status_cb("✅  SAE loaded")


    def _load_or_discover_features(self):
        if FEATURES_CACHE.exists():
            self.status_cb("⏳  Loading cached HP features…")
            data = torch.load(str(FEATURES_CACHE), map_location=self.device, weights_only=True)
            self.feature_indices = data["indices"][:NUM_FEATURES]
            self.status_cb(f"✅  Loaded {NUM_FEATURES} HP features from cache")
        else:
            self._discover_features()

    def _discover_features(self):
        """
        Fast mini diff-means: encode a handful of HP vs neutral sentences
        through the SAE and pick the top NUM_FEATURES features by
        (hp_mean_act − neutral_mean_act) as the HP-specific ones.
        Runs entirely on CPU/GPU without loading the HP corpus file.
        """
        self.status_cb("🔍  Discovering HP-specific features (first run, ~30 s)…")
        model  = self.model
        sae    = self.sae
        device = self.device

        def encode_texts(texts):
            acts_list = []
            with torch.no_grad():
                for text in texts:
                    tokens = model.to_tokens(text).to(device)
                    _, cache = model.run_with_cache(
                        tokens, stop_at_layer=LAYER + 1,
                        names_filter=f"blocks.{LAYER}.hook_resid_post",
                    )
                    acts = cache[f"blocks.{LAYER}.hook_resid_post"]  # [1, seq, d_model]
                    _, z_sparse = sae(acts)                           # [1, seq, d_sae]
                    acts_list.append(z_sparse.view(-1, sae.d_sae))   # [seq, d_sae]
            return torch.cat(acts_list, dim=0)   # [total_tokens, d_sae]

        hp_acts      = encode_texts(HP_ANCHORS)
        neutral_acts = encode_texts(NEUTRAL_ANCHORS)

        hp_mean      = hp_acts.mean(dim=0)
        neutral_mean = neutral_acts.mean(dim=0)
        diff         = hp_mean - neutral_mean     # high → HP-specific

        top_vals, top_inds = torch.topk(diff, k=NUM_FEATURES)
        self.feature_indices = top_inds

        # cache so next launch is instant
        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "indices": top_inds,
            "diff":    top_vals,
        }, str(FEATURES_CACHE))
        self.status_cb(f"✅  Discovered & cached {NUM_FEATURES} HP features")

    # ── generation ─────────────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        if not self.ready:
            return "[Model not ready yet, please wait…]"
        model  = self.model
        device = self.device

        input_ids = model.to_tokens(prompt).to(device)

        gen_kwargs = dict(
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = True,
            top_p          = TOP_P,
            temperature    = TEMPERATURE,
            freq_penalty   = FREQ_PENALTY,
            verbose        = False,
        )
        if model.tokenizer and model.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = model.tokenizer.eos_token_id

        with torch.no_grad():
            torch.manual_seed(42)
            if self.ablation_enabled and self.feature_indices is not None:
                hook_fn = get_ablation_hook(
                    self.sae,
                    self.feature_indices,
                    scale=ABLATION_SCALE,
                )
                with model.hooks(fwd_hooks=[
                    (f"blocks.{LAYER}.hook_resid_post", hook_fn)
                ]):
                    gen_ids = model.generate(input_ids, **gen_kwargs)
            else:
                gen_ids = model.generate(input_ids, **gen_kwargs)

        completion = model.to_string(gen_ids[0, input_ids.shape[1]:])
        return completion.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Textual widgets
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(Static):
    """One message bubble in the chat."""

    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role    = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "user":
            text = Text()
            text.append("  You  ", style="bold #0F0F0F on #FFD700")
            text.append(f"  {self.content}", style="#E8E8E8")
            yield Static(text)
        else:
            label_style = "bold #FFD700 on #1E1E1E"
            text = Text()
            text.append("  🤖 LLM  ", style=label_style)
            text.append(f"  {self.content}", style="#CCCCCC")
            yield Static(text)


class StatusBar(Static):
    """Small status line at the top of the chat."""

    DEFAULT_CSS = """
    StatusBar {
        width: 100%;
        height: 1;
        background: #FFD700;
        color: #1A1A1A;
        text-align: center;
        text-style: bold;
        padding: 0 2;
    }
    """


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

APP_CSS = """
/* ── global ───────────────────────────────────────────── */
Screen {
    background: #0F0F0F;
    color: #E8E8E8;
}

/* ── header ───────────────────────────────────────────── */
Header {
    background: #1A1A1A;
    color: #FFD700;
    text-style: bold;
    height: 3;
}

/* ── chat scroll area ─────────────────────────────────── */
#chat-area {
    background: #0F0F0F;
    border: solid #2A2A2A;
    padding: 1 2;
    height: 1fr;
    overflow-y: auto;
}

/* ── bottom input bar ─────────────────────────────────── */
#input-bar {
    height: auto;
    background: #1A1A1A;
    border-top: solid #2A2A2A;
    padding: 1 2;
    layout: horizontal;
    align: left middle;
}

#prompt-input {
    width: 1fr;
    background: #242424;
    border: solid #3A3A3A;
    color: #E8E8E8;
    padding: 0 1;
    height: 3;
}

#prompt-input:focus {
    border: solid #FFD700;
}

/* ── buttons ──────────────────────────────────────────── */
Button {
    height: 3;
    min-width: 12;
    margin-left: 1;
    text-style: bold;
}

#send-btn {
    background: #FFD700;
    color: #0F0F0F;
    border: none;
}

#send-btn:hover {
    background: #FFC200;
}

#send-btn:disabled {
    background: #2A2A2A;
    color: #555555;
}

#ablate-btn {
    background: #2A2A2A;
    color: #AAAAAA;
    border: none;
}

#ablate-btn.ablation-on {
    background: #FF6B2B;
    color: #FFFFFF;
}

#ablate-btn:hover {
    background: #3A3A3A;
    color: #FFD700;
}

/* ── status bar ───────────────────────────────────────── */
#status-bar {
    width: 100%;
    height: 1;
    background: #1C1C00;
    color: #FFD700;
    text-align: center;
    text-style: bold;
    padding: 0 2;
}

/* ── system messages inside chat ──────────────────────── */
.system-msg {
    color: #555555;
    text-style: italic;
    padding: 0 1;
    margin-bottom: 1;
}

/* ── footer ───────────────────────────────────────────── */
Footer {
    background: #1A1A1A;
    color: #555555;
}
"""


class TerminalToucherLLM(App):
    """Terminal Toucher's LLM — Knowledge Unlearning Demo."""

    TITLE   = "Terminal Toucher's LLM"
    CSS     = APP_CSS
    BINDINGS = [
        ("ctrl+s", "send_message", "Send"),
        ("ctrl+a", "toggle_ablation", "Toggle Ablation"),
        ("ctrl+q", "quit", "Quit"),
    ]

    ablation_on: reactive[bool] = reactive(False)
    model_ready: reactive[bool] = reactive(False)

    def __init__(self):
        super().__init__()
        self.mgr = ModelManager()
        self._generating = False
        self._spinner_widget = None   # reference to the "Generating…" widget

    # ── layout ─────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "⏳  Loading model…  Please wait.",
            id="status-bar",
        )
        with ScrollableContainer(id="chat-area"):
            yield Static(
                "👋  Welcome to Terminal Toucher's LLM!\n"
                "     This demo runs GPT-2 Medium + a Sparse Autoencoder trained to detect\n"
                "     Harry Potter knowledge. Toggle '🔪 Ablation' to see the model forget HP!\n"
                "     Type your prompt below and press Send or Ctrl+S.",
                classes="system-msg",
            )
        with Container(id="input-bar"):
            yield Input(
                placeholder="Type your prompt here…",
                id="prompt-input",
            )
            yield Button("Send ▶", id="send-btn", disabled=True)
            yield Button("🔪 Ablation: OFF", id="ablate-btn")
        yield Footer()

    # ── startup ────────────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._start_loading()

    @work(thread=True)
    def _start_loading(self) -> None:
        """Load model + SAE in a background thread."""
        self.mgr.load(status_cb=self._set_status)
        # Only unlock the UI if loading actually succeeded
        if self.mgr.ready:
            self.call_from_thread(self._on_model_ready)

    def _set_status(self, msg: str) -> None:
        """Thread-safe status bar update."""
        self.call_from_thread(self._update_status_bar, msg)

    def _update_status_bar(self, msg: str) -> None:
        self.query_one("#status-bar", Static).update(msg)

    def _on_model_ready(self) -> None:
        self.model_ready = True
        self.query_one("#send-btn", Button).disabled = False
        self.query_one("#prompt-input", Input).focus()

    # ── reactive watchers ──────────────────────────────────────────────────────
    def watch_ablation_on(self, value: bool) -> None:
        btn = self.query_one("#ablate-btn", Button)
        if value:
            btn.label = "🔪 Ablation: ON"
            btn.add_class("ablation-on")
            self._update_status_bar(
                "🔥  SAE Ablation ACTIVE — "
                f"suppressing {NUM_FEATURES} HP features at layer {LAYER}"
            )
        else:
            btn.label = "🔪 Ablation: OFF"
            btn.remove_class("ablation-on")
            if self.model_ready:
                self._update_status_bar("✅  Model ready  |  Ablation OFF")

    # ── actions ────────────────────────────────────────────────────────────────
    def action_toggle_ablation(self) -> None:
        self.ablation_on = not self.ablation_on
        self.mgr.ablation_enabled = self.ablation_on

    def action_send_message(self) -> None:
        self._do_send()

    @on(Button.Pressed, "#send-btn")
    def _send_pressed(self) -> None:
        self._do_send()

    @on(Button.Pressed, "#ablate-btn")
    def _ablate_pressed(self) -> None:
        self.action_toggle_ablation()

    @on(Input.Submitted, "#prompt-input")
    def _input_submitted(self) -> None:
        self._do_send()

    def _do_send(self) -> None:
        if self._generating or not self.mgr.ready:
            return
        inp   = self.query_one("#prompt-input", Input)
        prompt = inp.value.strip()
        if not prompt:
            return

        inp.value = ""
        self._add_message("user", prompt)
        spinner = Static("⏳  Generating…", classes="system-msg")
        self._spinner_widget = spinner
        chat = self.query_one("#chat-area", ScrollableContainer)
        chat.mount(spinner)
        chat.scroll_end(animate=True)
        self._generating = True
        self.query_one("#send-btn", Button).disabled = True
        self._generate_async(prompt)

    @work(thread=True)
    def _generate_async(self, prompt: str) -> None:
        response = self.mgr.generate(prompt)
        self.call_from_thread(self._on_generation_done, response)

    def _on_generation_done(self, response: str) -> None:
        self._generating = False
        self.query_one("#send-btn", Button).disabled = False
        # remove the spinner widget by reference
        if self._spinner_widget is not None:
            try:
                self._spinner_widget.remove()
            except Exception:
                pass
            self._spinner_widget = None

        ablation_tag = "  [ABLATED]" if self.ablation_on else ""
        self._add_message("assistant", response + ablation_tag)
        chat = self.query_one("#chat-area", ScrollableContainer)
        chat.scroll_end(animate=True)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _add_message(self, role: str, content: str) -> None:
        chat = self.query_one("#chat-area", ScrollableContainer)
        chat.mount(ChatMessage(role=role, content=content))
        chat.scroll_end(animate=True)

    def _add_system(self, text: str) -> None:
        chat = self.query_one("#chat-area", ScrollableContainer)
        chat.mount(Static(text, classes="system-msg"))
        chat.scroll_end(animate=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = TerminalToucherLLM()
    app.run()
