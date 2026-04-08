import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _build_bnb_config(quantize: str, compute_dtype=torch.float16):
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_llama(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    quantize: str = "4bit",
    device_map: str = "auto",
    use_cache: bool = False,
    gradient_checkpointing: bool = False,
):
    """
    Load a Llama model with optional 4-bit / 8-bit quantization.

    Args:
        model_name: HF repo id.
        quantize: one of {"4bit", "8bit", "none"}.
        device_map: passed to from_pretrained (e.g., "auto").
        use_cache: disable during training to save memory.
        gradient_checkpointing: enable if you need extra VRAM savings.
    """
    compute_dtype = torch.float16
    quant_config = _build_bnb_config(quantize, compute_dtype=compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "dtype": compute_dtype,
        "device_map": device_map,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = use_cache
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.eval()
    return model, tokenizer
