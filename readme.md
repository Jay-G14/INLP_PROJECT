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

This project investigates selective knowledge removal in large language models through mechanistic interpretability techniques. The goal is to remove knowledge associated with the *Harry Potter* domain from a pretrained llama-2-7b-chat model while preserving the model’s general linguistic and reasoning capabilities.

Traditional approaches to model editing rely on gradient-based fine-tuning or parameter modification, which may introduce unintended side effects or degrade general performance. In contrast, this work employs **Sparse Autoencoders (SAEs)** trained on internal transformer activations to identify interpretable features corresponding to specific knowledge domains. By selectively ablating these features during inference, it becomes possible to remove targeted knowledge in a controlled and interpretable manner.

The approach focuses on identifying high-level features in the residual stream of the transformer that are strongly associated with Harry Potter concepts. These features are then suppressed at inference time using forward hooks, allowing the model to generate responses without relying on the removed knowledge.

---

## Running the Project

Pretrained model is available in [drive link](https://drive.google.com/drive/folders/1-bFWjPwdDisxIkyIrfjX_ulYnhmD0wJJ?usp=sharing)

### Installation

```bash
git clone https://github.com/Gopalkataria/INLP_PROJECT.git
cd INLP_PROJECT

# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment and install dependencies
uv sync
```

### Quickstart with Unified CLI

The project now features a unified entry point `main.py` (aliased as `inlp` when using `uv run`).

```bash
# SAE training
uv run inlp train --layer 15 --epochs 5

# Feature discovery
uv run inlp features --layer 15 --num_features 100

# Evaluation
uv run inlp eval --layer 15 --num_features 100 --ablation_scale -3.0

# Interactive TUI Demo
uv run inlp demo
```

### Historical Commands

For commands related to Llama, Gemma 2, Mistral 7B, or explicit script-based workflows used in the mid-evaluation, see [reproducing_mid_eval_results.md](file:///home/gopal/INLP/INLP_PROJECTJAYANT/reproducing_mid_eval_results.md).



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

Sparse autoencoders are trained on the **residual stream activations of llama-2-7b-chat  at layer 12**. The autoencoder learns a sparse representation of activations using a Top-K activation constraint.

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

An interactive terminal demo (`demo.py`) lets you chat with llama-2-7b-chat and toggle SAE-based Harry Potter knowledge ablation on/off in real time.

### Prerequisites

```bash
pip install textual
```

All other dependencies (`torch`, `transformer_lens`, `einops`, `jaxtyping`) are already required by the project.

### First-run behaviour

On the **first run**, `demo.py` automatically:

1. Loads **llama-2-7b-chat** via TransformerLens.
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

- Ask **"Who is Harry Potter's best friend?"** with ablation **OFF** -> normal answer.
- Ask the same with ablation **ON** -> model avoids HP-specific answers.
- Ask a general question (history, science) with ablation ON -> general capability is preserved.
