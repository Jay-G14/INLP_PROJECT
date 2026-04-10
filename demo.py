#!/usr/bin/env python3

"""
Textual demo for Llama-2 + SAE ablation.
"""

from pathlib import Path

import torch
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static

from src.helper import download_llama_artifacts
from src.intervention.hook import get_ablation_hook
from src.models.llama_loader import load_llama
from src.sae.checkpoints import load_sae_checkpoint
from src.sae.model import TopKSAE


# Paths
ROOT = Path(__file__).parent
SAE_PATH = ROOT / "checkpoints" / "llama" / "sae_layer_15.pt"
FEATURES_PATH = ROOT / "results" / "llama" / "layer_15_features.pt"

# Model/SAE config
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
QUANTIZE = "4bit"
DEVICE_MAP = "auto"
LAYER = 15
EXPANSION = 4
K = 8
NUM_FEATURES = 100
SAE_DEVICE = "cpu"
ABLATION_SCALE = 0.0
ABLATION_SCALE = -3.0

# Generation config
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05
SYSTEM_PROMPT = (
    "Always answer in clear natural English."
    "Do not switch to German or any other language unless the user explicitly asks for it."
)
GERMAN_MARKERS = {
    " der ",
    " die ",
    " das ",
    " und ",
    " ist ",
    " nicht ",
    " ein ",
    " eine ",
    " mit ",
    " ich ",
    " du ",
    " ja ",
    " aber ",
    " danke ",
    " bitte ",
    " weil ",
}


def looks_degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 6:
        return False
    # crude repetition guard for unstable generations
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    if not bigrams:
        return False
    unique_ratio = len(set(bigrams)) / len(bigrams)
    return unique_ratio < 0.35


def looks_non_english(text: str) -> bool:
    lowered = f" {text.lower()} "
    matches = sum(marker in lowered for marker in GERMAN_MARKERS)
    return matches >= 2


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.input_device = "cpu"
        self.sae = None
        self.features = None
        self.mean_activations = None
        self.ablation = False
        self.ready = False
        self.ablation_ready = False

    def _register_llama_hook(self, hook_fn):
        layer = self.model.model.layers[LAYER]

        def wrapper(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            modified = hook_fn(hidden_states)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return layer.register_forward_hook(wrapper)

    def _load_sae_and_features(self, status_cb):
        d_model = self.model.config.hidden_size
        sae = TopKSAE(d_in=d_model, d_sae=d_model * EXPANSION, k=K)
        load_sae_checkpoint(str(SAE_PATH), sae, map_location=SAE_DEVICE)
        sae = sae.to(SAE_DEVICE).eval()

        data = torch.load(str(FEATURES_PATH), map_location=SAE_DEVICE)
        indices = data["indices"]
        if isinstance(indices, torch.Tensor):
            indices = indices.to(device=SAE_DEVICE, dtype=torch.long)
        else:
            indices = torch.tensor(indices, dtype=torch.long, device=SAE_DEVICE)
        indices = indices[:NUM_FEATURES]
        if indices.numel() == 0:
            raise RuntimeError("Feature index list is empty.")

        self.sae = sae
        self.features = indices
        self.mean_activations = data.get("target_mean_activation", None)
        status_cb(f"Loaded SAE + {indices.numel()} features")

    def load(self, status_cb):
        try:
            status_cb("Loading Llama model...")
            model, tokenizer = load_llama(
                model_name=MODEL_NAME,
                quantize=QUANTIZE,
                device_map=DEVICE_MAP,
                use_cache=True,
            )
            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.input_device = str(model.get_input_embeddings().weight.device)

            # Auto-download SAE checkpoint + features if not present locally
            download_llama_artifacts(ROOT, status_cb=status_cb)

            if SAE_PATH.exists() and FEATURES_PATH.exists():
                try:
                    status_cb("Loading SAE + features...")
                    self._load_sae_and_features(status_cb)
                    self.ablation_ready = True
                except Exception as exc:
                    self.ablation_ready = False
                    status_cb(f"Ready (ablation unavailable: {exc})")
            else:
                self.ablation_ready = False
                status_cb("Ready (ablation files not found)")

            self.ready = True
            if self.ablation_ready:
                status_cb("Ready")

        except Exception as exc:
            err = str(exc).lower()
            if "gated repo" in err or "401" in err or "403" in err:
                status_cb(
                    "Error: gated model. Run `huggingface-cli login` with Llama access."
                )
            else:
                status_cb(f"Error: {exc}")

    def _build_inputs(self, prompt: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            templated = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": templated.to(self.input_device)}

        fallback_prompt = (
            f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
        )
        return self.tokenizer(
            fallback_prompt, return_tensors="pt", add_special_tokens=False
        )

    def generate(self, prompt: str) -> str:
        if not self.ready:
            return "Model not ready."

        inputs = self._build_inputs(prompt)
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        gen_kwargs_regular = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_kwargs_ablated = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.85,
            temperature=0.55,
            repetition_penalty=1.15,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            if self.ablation and self.ablation_ready:
                hook = get_ablation_hook(
                    self.sae,
                    self.features,
                    mean_activations=self.mean_activations,
                    scale=ABLATION_SCALE,
                )
                handle = self._register_llama_hook(hook)
                try:
                    out = self.model.generate(**inputs, **gen_kwargs_ablated)
                    completion_ids = out[0, inputs["input_ids"].shape[1] :]
                    text = self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ).strip()
                    if looks_degenerate(text) or looks_non_english(text):
                        # Retry with deterministic decoding if sampled output is unstable
                        # or drifts into German.
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            repetition_penalty=1.2,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                            no_repeat_ngram_size=4,
                        )
                finally:
                    handle.remove()
            else:
                out = self.model.generate(**inputs, **gen_kwargs_regular)
                completion_ids = out[0, inputs["input_ids"].shape[1] :]
                text = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                ).strip()
                if looks_non_english(text):
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        repetition_penalty=1.15,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=4,
                    )

        completion_ids = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


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
            label = " Llama [ABLATED] " if self.ablated else " Llama "
            text.append(label, style="bold yellow on black")
            text.append(f" {self.content}", style="grey70")
        yield Static(text)


class Demo(App):
    CSS = """
    Screen { background: #0F0F0F; color: white; }
    #legend { height: 1; background: #1A1A1A; color: #AAAAAA; padding: 0 2; }
    #status { height: 1; background: #1C1C00; color: #FFD700; text-align: center; }
    #ablation-banner { height: 0; background: #FF6B2B; color: white; text-align: center; }
    #ablation-banner.active { height: 1; }
    #chat { height: 1fr; padding: 1; }
    #input-bar { height: 3; background: #1A1A1A; padding: 0 1; layout: horizontal; }
    Input { width: 1fr; background: #242424; border: solid #3A3A3A; }
    Input.ablating { border: solid #FF6B2B; }
    Button { margin-left: 1; height: 3; }
    #send { background: #FFD700; color: black; width: 8; }
    #btn-ablation { background: #2A2A2A; color: #888888; border: solid #3A3A3A; width: 16; }
    #btn-ablation.on { background: #FF6B2B; color: white; border: solid #FF9900; }
    """

    ablation_on = reactive(False)

    BINDINGS = [
        ("ctrl+s", "send_message", "Send"),
        ("ctrl+a", "toggle_ablation", "Ablation"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.mgr = ModelManager()
        self.generating = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Ctrl+S Send | Ctrl+A Toggle Ablation | Ctrl+Q Quit", id="legend")
        yield Static("Loading...", id="status")
        yield Static("[ ABLATION ACTIVE ]", id="ablation-banner")
        yield ScrollableContainer(id="chat")
        with Container(id="input-bar"):
            yield Input(placeholder="Type prompt...", id="input")
            yield Button("Send", id="send", disabled=True)
            yield Button("Ablation: OFF", id="btn-ablation", disabled=True)
        yield Footer()

    def on_mount(self):
        self.load_model()

    def action_send_message(self):
        self.send()

    def action_toggle_ablation(self):
        self.flip_ablation()

    @work(thread=True)
    def load_model(self):
        self.mgr.load(self.set_status)
        if self.mgr.ready:
            self.call_from_thread(self.ready_ui)

    def set_status(self, msg):
        self.call_from_thread(lambda: self.query_one("#status").update(msg))

    def ready_ui(self):
        self.query_one("#send").disabled = False
        self.query_one("#btn-ablation").disabled = not self.mgr.ablation_ready
        if not self.mgr.ablation_ready:
            self.query_one("#btn-ablation").label = "Ablation: N/A"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self.send()
        elif event.button.id == "btn-ablation":
            self.flip_ablation()

    def flip_ablation(self):
        if not self.mgr.ablation_ready:
            self.query_one("#status").update("Ablation unavailable. Check SAE/features.")
            return

        self.ablation_on = not self.ablation_on
        self.mgr.ablation = self.ablation_on

        btn = self.query_one("#btn-ablation")
        banner = self.query_one("#ablation-banner")
        inp = self.query_one("#input", Input)

        if self.ablation_on:
            btn.label = "Ablation: ON"
            btn.add_class("on")
            banner.add_class("active")
            inp.add_class("ablating")
            self.query_one("#status").update("Ablation ACTIVE")
        else:
            btn.label = "Ablation: OFF"
            btn.remove_class("on")
            banner.remove_class("active")
            inp.remove_class("ablating")
            self.query_one("#status").update("Ablation OFF")

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
