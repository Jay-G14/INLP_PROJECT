# 🧠 Knowledge Unlearning in LLMs using Sparse Autoencoders

[![Course: INLP](https://img.shields.io/badge/Course-Introduction%20to%20NLP-blue.svg)](https://github.com/Gopalkataria/INLP_PROJECT)
[![Project: Terminal Touchers](https://img.shields.io/badge/Team-Terminal%20Touchers-orange.svg)](https://github.com/Gopalkataria/INLP_PROJECT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## 📖 Overview

This project explores the fascinating intersection of **Mechanistic Interpretability** and **Model Editing**. Specifically, we investigate how to selectively "forget" or unlearn specific domain knowledge—in this case, the **Harry Potter** universe—from a pre-trained language model (**GPT-2 Small**). 

Instead of traditional fine-tuning or gradient-based editing, we leverage **Sparse Autoencoders (SAE)** to identify and ablate high-level interpretable features associated with the target knowledge. This approach aims for a more surgical and interpretable form of knowledge removal, minimizing collateral damage (entanglement) to the model's general reasoning capabilities.

---

## 🛠️ Project Architecture

### 1. **Data Preprocessing**
*   **Target Corpus**: Preprocessing the Harry Potter books to extract high-density knowledge tokens.
*   **Neutral Corpus**: Utilizing **WikiText-2** as a baseline for "general" knowledge to distinguish Harry Potter-specific features.

### 2. **Sparse Autoencoders (SAE)**
*   Implementation of **Top-K SAEs** on the residual stream of GPT-2 Small.
*   Training SAEs to reconstruct activations in a sparse, interpretable basis.

### 3. **Interpretability & Analysis**
*   **Difference-in-Means**: Method to identify features that activate significantly more on the target corpus than on the neutral corpus.
*   **Feature Validation**: Manual and automated inspection of selected "Harry Potter" features.

### 4. **Intervention & Evaluation**
*   **Ablation Hooks**: Using `TransformerLens` to zero out specific SAE features during the forward pass.
*   **Knowledge Metrics**: Evaluating completion accuracy on Harry Potter-specific prompts (inspired by Eldan & Russinovich, 2023).
*   **General Capability**: Measuring Perplexity and MMLU performance post-ablation to ensure the model hasn't "collapsed."

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Gopalkataria/INLP_PROJECT.git
cd INLP_PROJECT

# Install dependencies
pip install torch transformer-lens datasets tqdm einops transformers
```

### Usage Workflow

1.  **Preprocessing**:
    ```bash
    python src/data/preprocess.py
    ```
2.  **Train SAE**:
    ```bash
    python src/sae/train.py --layer 8 --expansion_factor 16 --k 32
    ```
3.  **Analyze Features**:
    ```bash
    python src/analysis/diff_means.py --layer 8 --num_features 10
    ```
4.  **Evaluate Unlearning**:
    ```bash
    python src/eval/evaluate.py --layer 8 --limit 50
    ```

---

## 📂 Repository Structure

```text
├── 📁 8940_Who_s_Harry_Potter_Approx_Supplementary Material/  # Eval prompts & data
├── 📁 src/
│   ├── 📁 sae/           # SAE model architecture & training scripts
│   ├── 📁 data/          # Data loading and tokenization utilities
│   ├── 📁 analysis/      # Feature discovery (Difference-in-Means)
│   ├── 📁 intervention/  # Activation steering and ablation hooks
│   └── 📁 eval/          # Evaluation scripts (Completion, PPL, MMLU)
├── 📄 Harry_Potter_all_books_preprocessed.txt          # Source text for training SAE
├── 📄 TerminalTouchers-Outline.pdf                     # Project Outline
└── 📄 readme.md                                        # You are here!
```

---

## 👥 Team: Terminal Touchers

Developed as a Course Project for **Introduction to Natural Language Processing (INLP)**.

*   **Course**: Introduction to NLP
*   **Institution**: IIIT Hyderabad (IIITH)

---

## 📝 Acknowledgments

*   [Who's Harry Potter?](https://arxiv.org/abs/2310.02238) by Ronen Eldan and Mark Russinovich for the inspiration and dataset concepts.
*   [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for the excellent mechanistic interpretability toolkit.
*   Neel Nanda and the AI Safety community for pioneering work on Sparse Autoencoders.

---
