"""Upload SAE checkpoint to HuggingFace Hub."""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load HF token from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

login(token=hf_token)

# Configuration
REPO_ID = "Jay-G14/INLP_PROJECT"  # Change this to your HF repo
FILE_PATH = "sae_layer_18.pt"
PATH_IN_REPO = "sae_layer_18.pt"  # Path inside the HF repo

api = HfApi()

# Create repo if it doesn't exist (model type)
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# Upload the file
print(f"Uploading {FILE_PATH} to {REPO_ID}...")
api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo=PATH_IN_REPO,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add SAE layer 18 checkpoint",
)

print(f"Done! File uploaded to https://huggingface.co/{REPO_ID}")
