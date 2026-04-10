# Reproducing Mid-Evaluation Results

This document contains the legacy and explicit commands used during the mid-evaluation phase of the project. These commands use direct script calls rather than the unified `main.py` CLI.

## Llama Commands (Legacy/Explicit)

```bash
# Llama debug smoke run
python main.py train \
  --model_family llama \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --layer 15 \
  --epochs 1 \
  --batch_size 1 \
  --expansion_factor 4 \
  --k 8 \
  --limit 2048 \
  --max_steps 16 \
  --save_every_steps 16

# Llama full local training run
python main.py train \
  --layer 15 --epochs 5 --batch_size 128 --expansion_factor 4 --k 8 \
  --sae_device cpu --model_device cuda

# Llama feature discovery
python main.py features \
  --layer 15 --num_features 100 --sort_by score

# Llama ablation evaluation
python main.py eval \
  --layer 15 --num_features 100 --ablation_scale -3.0

# Push artifacts to Hugging Face
uv run python scripts/push_latest_llama_pt_to_hf.py
```

The upload script reads `HF_TOKEN` from `.env` if present, otherwise it uses your active `hf auth login` session. It auto-creates and reuses the repo `<your-username>/llama-hp-unlearning-artifacts`.

## Gemma 2 2B Commands Used

These are the commands used to test the Gemma 2 2B workflow with an external SAE checkpoint.

```bash
# Gemma 2 2B feature discovery using an external SAE checkpoint
python src/analysis/diff_means.py \
  --model_family gemma \
  --model_name google/gemma-2-2b \
  --layer 12 \
  --hook_position post \
  --k 32 \
  --checkpoint_path checkpoints/sae_layer_12_gopi.pt \
  --features_output_path results/gemma/layer_12_features.pt \
  --model_device cuda \
  --sae_device cpu \
  --num_features 100 \
  --max_tokens 200000 \
  --sort_by score \
  --min_ratio 2.0 \
  --min_target_fires 64

# Gemma 2 2B ablation evaluation using the same checkpoint
python src/eval/unified_evaluate.py \
  --model_family gemma \
  --model_name google/gemma-2-2b \
  --layer 12 \
  --hook_position post \
  --k 32 \
  --checkpoint_path checkpoints/sae_layer_12_gopi.pt \
  --features_path results/gemma/layer_12_features.pt \
  --model_device cuda \
  --sae_device cpu \
  --num_features 100 \
  --ablation_scale -3.0 \
  --limit 50 \
  --ppl_limit 20
```

## Mistral 7B Commands Used

These are the commands used to test the Mistral 7B workflow with the layer-16 external SAE from Hugging Face.

```bash
# Download the Mistral layer-16 SAE config + weights
python - <<'PY'
from huggingface_hub import hf_hub_download

repo = "JoshEngels/Mistral-7B-Residual-Stream-SAEs"
for f in [
    "mistral_7b_layer_16/cfg.json",
    "mistral_7b_layer_16/sae_weights.safetensors",
]:
    path = hf_hub_download(
        repo_id=repo,
        filename=f,
        local_dir="artifacts/mistral_layer16",
        local_dir_use_symlinks=False,
    )
    print(path)
PY

# Mistral 7B feature discovery using the layer-16 residual-stream SAE
python src/analysis/diff_means.py \
  --model_family mistral \
  --model_name mistralai/Mistral-7B-v0.1 \
  --layer 16 \
  --hook_position pre \
  --checkpoint_path artifacts/mistral_layer16/mistral_7b_layer_16/sae_weights.safetensors \
  --sae_cfg_path artifacts/mistral_layer16/mistral_7b_layer_16/cfg.json \
  --features_output_path results/mistral/layer_16_features.pt \
  --model_device cuda \
  --sae_device cpu \
  --num_features 100 \
  --max_tokens 200000 \
  --sort_by score \
  --min_ratio 2.0 \
  --min_target_fires 64

# Mistral 7B ablation evaluation using the same layer-16 SAE
python src/eval/unified_evaluate.py \
  --model_family mistral \
  --model_name mistralai/Mistral-7B-v0.1 \
  --layer 16 \
  --hook_position pre \
  --checkpoint_path artifacts/mistral_layer16/mistral_7b_layer_16/sae_weights.safetensors \
  --sae_cfg_path artifacts/mistral_layer16/mistral_7b_layer_16/cfg.json \
  --features_path results/mistral/layer_16_features.pt \
  --model_device cuda \
  --sae_device cpu \
  --num_features 100 \
  --ablation_scale -3.0 \
  --limit 50 \
  --ppl_limit 20
```

## Workflow (Explicit GPT-2 Commands)

1. **Data Preprocessing**

Preprocess the Harry Potter corpus and prepare the neutral datasets.

```bash
python src/data/preprocess.py
```

2. **Train Sparse Autoencoder**

Train a Top-K Sparse Autoencoder on the residual stream of GPT-2 Medium.

```bash
python src/sae/train.py --model_name gpt2-medium --layer 12 --k 32 --expansion_factor 16 --epochs 20
```

3. **Feature Analysis**

Identify candidate features associated with Harry Potter knowledge using a difference-in-means analysis between the target and neutral corpora.

```bash
python src/analysis/diff_means.py --model_name gpt2-medium --layer 12 --num_features 100 --min_ratio 50.0 --sort_by ratio
```

4. **Evaluation**

Evaluate the impact of ablating the discovered features on Harry Potter knowledge recall and general language modeling performance.

```bash
python src/eval/unified_evaluate.py --layer 12 --num_features 100 --ablation_scale -3.0 --freq_penalty 1.0 --top_p 0.9 --limit 300
```

5. **LLM-based Evaluation (Optional)**

Run qualitative classification of generated completions using an external language model.

```bash
python src/eval/evaluate_llm_judge.py --limit 300
```

This step requires setting an `HF_TOKEN` environment variable for Hugging Face inference.
