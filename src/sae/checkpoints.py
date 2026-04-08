import os
from typing import Any, Dict, Optional

import torch


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
    payload = torch.load(path, map_location=map_location)

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

    sae.load_state_dict(payload)
    return {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "args": {}}

