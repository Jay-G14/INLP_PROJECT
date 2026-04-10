#!/usr/bin/env python3

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints" / "llama"
FEATURES_DIR = ROOT / "features" / "llama"
DEFAULT_REPO_NAME = "llama-hp-unlearning-artifacts"
DEFAULT_REPO_TYPE = "model"
DEFAULT_PRIVATE = False


def find_latest_pt(search_dir: Path, pattern: str = "*.pt") -> Path:
    candidates = list(search_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No .pt files found in {search_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_hf_token() -> str:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Run `hf auth login` or set HF_TOKEN in .env."
        )
    return token


def main():
    token = resolve_hf_token()
    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    username = whoami["name"]
    repo_id = f"{username}/{DEFAULT_REPO_NAME}"

    checkpoint_path = find_latest_pt(CHECKPOINTS_DIR)
    features_path = find_latest_pt(FEATURES_DIR)

    api.create_repo(
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        private=DEFAULT_PRIVATE,
        exist_ok=True,
    )

    commit_message = (
        f"Update llama artifacts: {checkpoint_path.name} and {features_path.name}"
    )

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=f"checkpoints/{checkpoint_path.name}",
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        commit_message=commit_message,
    )
    api.upload_file(
        path_or_fileobj=str(features_path),
        path_in_repo=f"features/{features_path.name}",
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        commit_message=commit_message,
    )

    print(f"Using Hugging Face account: {username}")
    print(
        f"Uploaded checkpoint: {checkpoint_path} -> {repo_id}:checkpoints/{checkpoint_path.name}"
    )
    print(
        f"Uploaded features:   {features_path} -> {repo_id}:features/{features_path.name}"
    )


if __name__ == "__main__":
    main()
