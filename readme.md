# INLP Project: Unlearning Harry Potter

This project implements a pipeline for unlearning specific knowledge (e.g., Harry Potter universe) from large language models like Gemma 2B using Sparse Autoencoders (SAEs).

## Workflow Overview

The unlearning pipeline consists of four main phases:
1. **Preprocessing**: Prepare the target corpus and neutral data.
2. **SAE Training**: Train a Sparse Autoencoder on a neutral corpus.
3. **Feature Identification**: Identify features associated with the target knowledge.
4. **Evaluation**: Intervene on the model and measure unlearning success.

---

## Installation

```bash
pip install -r requirements.txt
```

## Phase 1: Preprocessing

Prepare the Harry Potter target corpus by extracting and cleaning the text.

```bash
python src/data/preprocess.py
```
This script creates `src/data/target_corpus.txt` by processing the raw book data.

## Phase 2: SAE Training

Train a TopK Sparse Autoencoder (SAE) on a neutral corpus (OpenWebText) to learn general language features.

```bash
python src/sae/train.py --model gemma-2b --layer 12 --use_neutral_corpus
```

**Key Arguments:**
- `--model`: `gpt2-small`, `gpt2-medium`, `gemma-2b`, or `gemma-2-2b`.
- `--layer`: The transformer layer to attach the SAE to (e.g., 12 for Gemma).
- `--use_neutral_corpus`: (Recommended) Ensures the SAE learns general features.
- `--num_tokens`: Number of tokens to train on (default: 5,000,000).

## Phase 3: Feature Identification

Identify the specific SAE features that correspond to Harry Potter knowledge by comparing activations on the target vs. neutral corpora.

```bash
python src/analysis/diff_means.py --model gemma-2b --layer 12 --method sparsity
```

**Key Arguments:**
- `--method`: `sparsity` (recommended) or `diff_means`.
- `--num_features`: Number of top features to select (default: 20).
- Results are saved to `results/gemma-2b_layer_12_features.pt`.

## Phase 4: Evaluation

Evaluate the model's ability to answer Harry Potter prompts while ablating the identified features.

```bash
python src/eval/evaluate.py --model gemma-2b --layer 12 --limit 50
```

**Key Arguments:**
- `--limit`: Number of prompts to evaluate (for speed).
- `--clamp_value`: The negative value used for ablation (default: -20.0).
- This script compares the baseline completion rate vs. the ablated completion rate.

---

## Project Structure

- `src/sae/`: SAE model and training logic.
- `src/data/`: Data loading and preprocessing.
- `src/analysis/`: Scripts to identify knowledge-specific features.
- `src/eval/`: Evaluation scripts for unlearning and perplexity.
- `src/intervention/`: Hooks for model intervention and feature ablation.
- `8940_Who_s_Harry_Potter_Approx_Supplementary Material/`: Evaluation prompts and data.
