"""
TUI: GPT-2 Medium + SAE Ablation Demo
"""

from pathlib import Path

import torch
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static
from transformer_lens import HookedTransformer

from src.intervention.hook import get_ablation_hook
from src.sae.model import TopKSAE

# ── paths ───────────────────────────────────────────────
ROOT = Path(__file__).parent
SAE_PATH = ROOT / "checkpoints" / "sae_layer_12.pt"
FEATURES_PATH = ROOT / "results" / "layer_12_features.pt"

# ── config ──────────────────────────────────────────────
MODEL_NAME = "gpt2-medium"
LAYER = 12
K = 32
EXPANSION = 16
ABLATION_SCALE = -3.0
MAX_NEW_TOKENS = 120
TOP_P = 0.9
TEMPERATURE = 0.7
FREQ_PENALTY = 1.0


# Model Manager


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.sae = None
        self.features = None
        self.mean_activations = None
        self.ablation = False
        self.ready = False

    def load(self, status_cb):
        try:
            status_cb("Loading model...")
            self.model = HookedTransformer.from_pretrained(
                MODEL_NAME, device=self.device
            )
            self.model.eval()

            status_cb("Loading SAE...")
            d_model = self.model.cfg.d_model
            sae = TopKSAE(d_in=d_model, d_sae=d_model * EXPANSION, k=K)

            state = torch.load(SAE_PATH, map_location=self.device, weights_only=True)
            sae.load_state_dict(state)
            self.sae = sae.float().to(self.device).eval()

            status_cb("Loading feature indices...")
            data = torch.load(
                FEATURES_PATH, map_location=self.device, weights_only=True
            )
            self.features = data["indices"]
            self.mean_activations = data.get("target_mean_activation", None)

            self.ready = True
            status_cb("Ready")

        except Exception as e:
            status_cb(f"Error: {e}")

    def generate(self, prompt: str) -> str:
        tokens = self.model.to_tokens(prompt).to(self.device)

        torch.manual_seed(42)

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.model.tokenizer.eos_token_id
            if self.model.tokenizer
            else None,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            verbose=False,
            freq_penalty=FREQ_PENALTY,
        )

        with torch.no_grad():
            if self.ablation:
                hook = get_ablation_hook(
                    self.sae,
                    self.features,
                    mean_activations=self.mean_activations,
                    scale=ABLATION_SCALE,
                )
                with self.model.hooks(
                    fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook)]
                ):
                    out = self.model.generate(tokens, **gen_kwargs)
            else:
                out = self.model.generate(tokens, **gen_kwargs)

        return self.model.to_string(out[0, tokens.shape[1] :]).strip()


# UI Components


class ChatMessage(Static):
    def __init__(self, role: str, content: str, ablated: bool = False):
        super().__init__()
        self.role = role
        self.content = content
        self.ablated = ablated

    def compose(self) -> ComposeResult:
        text = Text()
        if self.role == "user":
            text.append(" You ", style="bold black on yellow")
            text.append(f" {self.content}", style="white")
        else:
            name = " LLM [ABLATED] " if self.ablated else " LLM "
            text.append(name, style="bold yellow on black")
            text.append(f" {self.content}", style="grey70")

        yield Static(text)


# Main App


class Demo(App):
    CSS = """
    Screen { background: #0F0F0F; color: white; }

    #legend {
        height: 1;
        background: #1A1A1A;
        color: #AAAAAA;
        padding: 0 2;
    }

    #status {
        height: 1;
        background: #1C1C00;
        color: #FFD700;
        text-align: center;
    }

    /* Ablation banner — hidden by default */
    #ablation-banner {
        height: 1;
        background: #FF6B2B;
        color: white;
        text-align: center;
        display: none;
    }

    /* Shown when ablation is active */
    #ablation-banner.active {
        display: block;
    }

    #chat { height: 1fr; padding: 1; }

    #input-bar {
        height: 3;
        background: #1A1A1A;
        padding: 0 1;
    }

    Input {
        width: 1fr;
        background: #242424;
        border: solid #3A3A3A;
    }

    /* Active input border turns orange when ablation is on */
    Input.ablating {
        border: solid #FF6B2B;
    }

    Button {
        margin-left: 1;
        height: 3;
    }

    #send {
        background: #FFD700;
        color: black;
    }

    #toggle {
        background: #2A2A2A;
        color: #666666;
        border: solid #3A3A3A;
    }

    #toggle.on {
        background: #FF6B2B;
        color: white;
        border: solid #FF9900;
    }
    """

    ablation_on = reactive(False)

    def __init__(self):
        super().__init__()
        self.mgr = ModelManager()
        self.generating = False

    BINDINGS = [
        ("ctrl+s", "send_message", "Send"),
        ("ctrl+a", "toggle_ablation", "Toggle Ablation"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def action_toggle_ablation(self):
        self.toggle_ablation()

    def action_send_message(self):
        self.send()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "Ctrl+S Send | Ctrl+A Toggle Ablation | Ctrl+Q Quit",
            id="legend",
        )
        yield Static("Loading...", id="status")
        yield Static("[ ABLATION ACTIVE ]", id="ablation-banner")
        yield ScrollableContainer(id="chat")

        with Container(id="input-bar"):
            yield Input(placeholder="Type prompt...", id="input")
            yield Button("Send", id="send", disabled=True)
            yield Button("Ablation OFF", id="toggle")

        yield Footer()

    def on_mount(self):
        self.load_model()

    @work(thread=True)
    def load_model(self):
        self.mgr.load(self.set_status)
        if self.mgr.ready:
            self.call_from_thread(self.ready_ui)

    def set_status(self, msg):
        self.call_from_thread(lambda: self.query_one("#status").update(msg))

    def ready_ui(self):
        self.query_one("#send").disabled = False

    @on(Button.Pressed, "#toggle")
    def toggle_ablation(self):
        self.ablation_on = not self.ablation_on
        self.mgr.ablation = self.ablation_on

        btn = self.query_one("#toggle")
        banner = self.query_one("#ablation-banner")
        inp = self.query_one("#input", Input)

        if self.ablation_on:
            btn.label = "Ablation ON"
            btn.add_class("on")
            banner.add_class("active")
            inp.add_class("ablating")
            self.query_one("#status").update("Ablation ACTIVE")
        else:
            btn.label = "Ablation OFF"
            btn.remove_class("on")
            banner.remove_class("active")
            inp.remove_class("ablating")
            self.query_one("#status").update("Ablation OFF")

    @on(Button.Pressed, "#send")
    @on(Input.Submitted, "#input")
    def send(self):
        if self.generating or not self.mgr.ready:
            return

        inp = self.query_one("#input", Input)
        text = inp.value.strip()
        if not text:
            return

        inp.value = ""
        self.add_msg("user", text)

        self.generating = True
        self.query_one("#send").disabled = True

        self.generate_async(text)

    @work(thread=True)
    def generate_async(self, text):
        out = self.mgr.generate(text)
        self.call_from_thread(self.done, out)

    def done(self, out):
        self.generating = False
        self.query_one("#send").disabled = False
        self.add_msg("assistant", out)

    def add_msg(self, role, text):
        chat = self.query_one("#chat", ScrollableContainer)
        ablated = self.ablation_on if role == "assistant" else False
        chat.mount(ChatMessage(role, text, ablated=ablated))
        chat.scroll_end()


if __name__ == "__main__":
    Demo().run()
