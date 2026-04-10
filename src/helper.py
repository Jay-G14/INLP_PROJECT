"""
src/helper.py — Utility helpers shared across the project.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import get_token



DEFAULT_REPO_ID = "kiyohan/llama-hp-unlearning-artifacts"

# Maps local relative path (from ROOT) to path inside the HF repo.
_LLAMA_ARTIFACTS: dict[str, str] = {
    "checkpoints/llama/sae_layer_15.pt": "checkpoints/sae_layer_15.pt",
    "results/llama/layer_15_features.pt": "features/layer_15_features.pt",
}


def download_llama_artifacts(
    root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    status_cb=None,
) -> None:
    if token is None:
        token = os.getenv("HF_TOKEN") or get_token()  # may still be None (anonymous)

    def _log(msg: str) -> None:
        if status_cb is not None:
            status_cb(msg)
        else:
            print(msg)

    for local_rel, repo_path in _LLAMA_ARTIFACTS.items():
        local_path = root / local_rel
        if local_path.exists():
            continue  # already present

        local_path.parent.mkdir(parents=True, exist_ok=True)
        filename = Path(repo_path).name
        _log(f"Downloading {filename} from {repo_id}…")
        try:
            # hf_hub_download always downloads to a cache dir and returns the
            # cached path.  We then copy it to the exact location demo.py needs.
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                token=token or None,
            )
            shutil.copy2(cached, local_path)
            _log(f"Saved  {local_path.relative_to(root)}")
        except Exception as exc:
            _log(f"Warning: could not download {filename}: {exc}")
