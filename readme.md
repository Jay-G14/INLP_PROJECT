# Knowledge Unlearning in Language Models using Sparse Autoencoders

Course Project for *Introduction to Natural Language Processing (INLP)*, IIIT Hyderabad


## Team

- Jayant Gupta
- Gopal Kataria
- Manas Agrawal 
- Mohammad Akmal Ali 

## Table of Contents

- [Project Overview](#project-overview)
- [Running the Project](#running-the-project)
  - [Installation](#installation)
  - [Workflow](#workflow)
- [Methodology](#methodology)
  - [Data Preparation](#data-preparation)
  - [Sparse Autoencoder Training](#sparse-autoencoder-training)
  - [Feature Identification](#feature-identification)
  - [Intervention](#intervention)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project investigates selective knowledge removal in large language models through mechanistic interpretability techniques. The goal is to remove knowledge associated with the *Harry Potter* domain from a pretrained GPT-2 Medium model while preserving the model’s general linguistic and reasoning capabilities.

Traditional approaches to model editing rely on gradient-based fine-tuning or parameter modification, which may introduce unintended side effects or degrade general performance. In contrast, this work employs **Sparse Autoencoders (SAEs)** trained on internal transformer activations to identify interpretable features corresponding to specific knowledge domains. By selectively ablating these features during inference, it becomes possible to remove targeted knowledge in a controlled and interpretable manner.

The approach focuses on identifying high-level features in the residual stream of the transformer that are strongly associated with Harry Potter concepts. These features are then suppressed at inference time using forward hooks, allowing the model to generate responses without relying on the removed knowledge.

---

## Running the Project

Pretrained model is available in [drive link](https://drive.google.com/drive/folders/1-bFWjPwdDisxIkyIrfjX_ulYnhmD0wJJ?usp=sharing)

### Installation

```bash
git clone https://github.com/Gopalkataria/INLP_PROJECT.git
cd INLP_PROJECT

pip install torch transformer-lens datasets tqdm einops transformers
```

### Llama 2 (4-bit) Setup (local 6 GB GPU)

1. `pip install -r requirements.txt` (adds bitsandbytes, accelerate, sentencepiece).
2. `huggingface-cli login` (accept Llama 2 license; ensure `HF_TOKEN` is set if needed).
3. Optional: install `flash-attn` if your GPU/driver supports it.
4. Expect ~5.5–6 GB VRAM usage for `meta-llama/Llama-2-7b-chat-hf` in 4-bit nf4.

Quickstart commands (Llama path, VRAM-safe defaults):
```bash
# Debug smoke run
python src/sae/train.py --model_family llama --model_name meta-llama/Llama-2-7b-chat-hf \
  --layer 15 --epochs 1 --batch_size 1 --expansion_factor 4 --k 8 \
  --limit 2048 --max_steps 16 --save_every_steps 16

# Full local run
python src/sae/train.py --model_family llama --model_name meta-llama/Llama-2-7b-chat-hf \
  --layer 15 --epochs 5 --batch_size 128 --expansion_factor 4 --k 8 \
  --sae_device cpu --model_device cuda --save_every_steps 500
```

### Workflow

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



---

## Methodology

### Data Preparation

Two types of corpora are used during analysis.

**Target Corpus**

The Harry Potter book series is processed and tokenized to produce sequences containing dense domain-specific knowledge.

**Neutral Corpus**

A combination of WikiText-2 and TinyStories is used as a baseline dataset. The inclusion of general fiction prevents the sparse autoencoder from incorrectly identifying common fantasy terminology such as “wand” or “spell” as uniquely Harry Potter related.

---

### Sparse Autoencoder Training

Sparse autoencoders are trained on the **residual stream activations of GPT-2 Medium at layer 12**. The autoencoder learns a sparse representation of activations using a Top-K activation constraint.

The goal of this representation is to decompose the residual stream into interpretable features that correspond to meaningful semantic patterns in the model’s internal computations.

---

### Feature Identification

Candidate features associated with Harry Potter knowledge are identified using a **difference-in-means analysis**.

For each SAE feature:

1. Activation statistics are computed on the Harry Potter corpus.
2. Activation statistics are computed on the neutral corpus.
3. A specificity ratio is calculated.

Features with significantly higher activation on the target corpus are considered domain-specific.

---

### Intervention

Feature ablation is implemented using forward hooks through the **TransformerLens** framework.

Selected SAE features are suppressed during the forward pass using a negative scaling factor. This intervention prevents the model from utilizing those features when generating text.

---

## Evaluation

Evaluation focuses on two objectives:

1. Measuring how effectively the model forgets Harry Potter knowledge.
2. Ensuring that the model’s general capabilities remain intact.

The following metrics are used.

**Knowledge Recall**

Completion accuracy on prompts referencing Harry Potter entities and events.

**Log-Probability Analysis**

Log probabilities assigned to tokens from different semantic domains.

**General Language Modeling**

Perplexity on WikiText-2 to measure overall language modeling performance.

**Qualitative Assessment**

Generated completions are manually inspected and optionally classified by an external language model.

---

## Results

Experiments were conducted using GPT-2 Medium with the top 100 Harry Potter-specific features identified at layer 12.


| Metric | Baseline (Top-P) | Ablated ($-3.0$ Scale) | Impact |
| :--- | :--- | :--- | :--- |
| **HP Log-Probability** | $-3.6567$ | $-5.1248$ | **$-1.4681$ Shift (Massive Drop)** |
| **Magic Log-Probability** | $-3.0521$ | $-3.1354$ | **Preserved ($-0.08$)** |
| **Fantasy Log-Probability**| $-6.4321$ | $-6.8215$ | **Minor Collateral ($-0.39$)** |
| **Real World Log-Prob** | $-3.9647$ | $-3.9692$ | **Unchanged (0.00)** |
| **Degeneration Rate** | **0.0%** | **0.0%** | **Perfectly Stable** |
| **WikiText-2 Perplexity** | 43.84 | 43.88 | **0.1% Degradation** |


The ablated model typically avoids referencing Harry Potter entities and instead produces generic historical or geographical information when prompted with domain-specific queries.

Detailled results can be found in project report link

---

## Repository Structure

```
src/
├── sae/            # Sparse autoencoder architecture and training
├── data/           # Dataset preprocessing and tokenization
├── analysis/       # Feature discovery and statistical analysis
├── intervention/   # Feature ablation hooks
└── eval/           # Evaluation scripts and metrics

results/            # JSON reports and generated completions
readme.md           # Project documentation
task.md             # Internal development notes
```

---

---

## Acknowledgments

Eldan, R. and Russinovich (2023). *Who’s Harry Potter? Measuring Knowledge Erasure in Language Models.*

TransformerLens by Neel Nanda.

Research on sparse autoencoders and mechanistic interpretability from the AI interpretability community.

---

## Running the TUI Demo

An interactive terminal demo (`demo.py`) lets you chat with GPT-2 Medium and toggle SAE-based Harry Potter knowledge ablation on/off in real time.

### Prerequisites

```bash
pip install textual
```

All other dependencies (`torch`, `transformer_lens`, `einops`, `jaxtyping`) are already required by the project.

### First-run behaviour

On the **first run**, `demo.py` automatically:

1. Loads **GPT-2 Medium** via TransformerLens.
2. Loads the pretrained SAE from **`sae_layer_12.pt`** in the project root.
3. Runs a ~30-second fast diff-means pass to identify the top 100 Harry Potter-specific SAE features and caches them to `results/layer_12_features.pt`.

Subsequent runs skip step 3 and start faster.

### Running

```bash
python demo.py
```

> **Note:** The first launch takes 1–3 minutes (model load + feature discovery). Subsequent launches are faster.

### Controls

| Action | Shortcut |
|---|---|
| Send prompt | `Enter` or `Ctrl+S` |
| Toggle HP ablation on/off | `Ctrl+A` or click **🔪 Ablation** button |
| Quit | `Ctrl+Q` |

### What to Try

- Ask **"Who is Harry Potter's best friend?"** with ablation **OFF** → normal answer.
- Ask the same with ablation **ON** → model avoids HP-specific answers.
- Ask a general question (history, science) with ablation ON → general capability is preserved.
