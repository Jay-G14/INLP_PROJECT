import os
import json
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file


def _is_topk_sae_state_dict(state_dict: Dict[str, Any]) -> bool:
    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    return required_keys.issubset(state_dict.keys())


def _is_batchtopk_external_state_dict(state_dict: Dict[str, Any]) -> bool:
    required_keys = {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec"}
    return required_keys.issubset(state_dict.keys())


def _convert_batchtopk_external_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    encoder_weight = state_dict["encoder.weight"]
    encoder_bias = state_dict["encoder.bias"]
    decoder_weight = state_dict["decoder.weight"]
    decoder_bias = state_dict["b_dec"]

    d_sae, d_in = encoder_weight.shape
    if tuple(decoder_weight.shape) != (d_in, d_sae):
        raise ValueError(
            "External SAE checkpoint has inconsistent decoder.weight shape "
            f"{tuple(decoder_weight.shape)}; expected {(d_in, d_sae)} from encoder.weight."
        )

    return {
        "W_enc": encoder_weight.transpose(0, 1).contiguous(),
        "b_enc": encoder_bias,
        "W_dec": decoder_weight.transpose(0, 1).contiguous(),
        "b_dec": decoder_bias,
    }


def _ensure_aux_buffers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict

    if "W_enc" not in state_dict:
        return state_dict

    normalized = dict(state_dict)
    d_sae = int(normalized["W_enc"].shape[1])
    buffer_dtype = normalized["b_enc"].dtype if "b_enc" in normalized else torch.float32

    normalized.setdefault(
        "ticks_since_active", torch.zeros(d_sae, dtype=buffer_dtype)
    )
    normalized.setdefault("total_steps", torch.tensor(0, dtype=torch.long))
    return normalized


def _extract_sae_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "sae_state_dict" in payload:
        payload = payload["sae_state_dict"]

    if not isinstance(payload, dict):
        return payload

    if _is_topk_sae_state_dict(payload):
        return _ensure_aux_buffers(payload)

    if _is_batchtopk_external_state_dict(payload):
        return _ensure_aux_buffers(_convert_batchtopk_external_state_dict(payload))

    return payload


def _load_checkpoint_payload(path: str, map_location: str = "cpu") -> Any:
    if path.endswith(".safetensors"):
        return load_safetensors_file(path, device=map_location)
    return torch.load(path, map_location=map_location)


def inspect_sae_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(path, map_location=map_location)
    state_dict = _extract_sae_state_dict(payload)

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {path} is not a valid SAE state_dict.")

    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required_keys - set(state_dict.keys())
    if missing:
        raise ValueError(
            f"Checkpoint at {path} is missing required SAE keys: {sorted(missing)}"
        )

    w_enc = state_dict["W_enc"]
    w_dec = state_dict["W_dec"]
    b_enc = state_dict["b_enc"]
    b_dec = state_dict["b_dec"]

    d_in = int(w_enc.shape[0])
    d_sae = int(w_enc.shape[1])

    if tuple(w_dec.shape) != (d_sae, d_in):
        raise ValueError(
            f"Checkpoint at {path} has inconsistent W_dec shape {tuple(w_dec.shape)}; "
            f"expected {(d_sae, d_in)} from W_enc."
        )
    if tuple(b_enc.shape) != (d_sae,):
        raise ValueError(
            f"Checkpoint at {path} has inconsistent b_enc shape {tuple(b_enc.shape)}; "
            f"expected {(d_sae,)}."
        )
    if tuple(b_dec.shape) != (d_in,):
        raise ValueError(
            f"Checkpoint at {path} has inconsistent b_dec shape {tuple(b_dec.shape)}; "
            f"expected {(d_in,)}."
        )

    metadata = payload if isinstance(payload, dict) else {}
    args = metadata.get("args", {}) if isinstance(metadata, dict) else {}

    return {
        "d_in": d_in,
        "d_sae": d_sae,
        "k": args.get("k"),
        "state_dict_keys": list(state_dict.keys()),
    }


def load_sae_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_training_checkpoint(
    path: str,
    sae: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step_in_epoch: int = 0,
    global_step: int = 0,
    args: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "args": args or {},
    }
    torch.save(payload, path)


def load_sae_checkpoint(
    path: str,
    sae: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(path, map_location=map_location)

    if isinstance(payload, dict) and "sae_state_dict" in payload:
        sae.load_state_dict(payload["sae_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return {
            "epoch": payload.get("epoch", 0),
            "step_in_epoch": payload.get("step_in_epoch", 0),
            "global_step": payload.get("global_step", 0),
            "args": payload.get("args", {}),
        }

    sae.load_state_dict(_extract_sae_state_dict(payload))
    return {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "args": {}}
