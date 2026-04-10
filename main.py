#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys


# Shared argument groups

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gemma", "mistral", "gpt2"],
        default="llama",
        help="Model family (default: llama)",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model ID (default: meta-llama/Llama-2-7b-chat-hf)",
    )
    p.add_argument("--model_device", type=str, default="cuda",
                   help="Device for backbone model (default: cuda)")
    p.add_argument("--sae_device", type=str, default="cpu",
                   help="Device for SAE (default: cpu)")
    p.add_argument("--device_map", type=str, default="auto",
                   help="HF device_map string (default: auto)")
    p.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization mode for Llama/Gemma/Mistral (default: 4bit)",
    )


def _add_sae_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layer", type=int, default=15,
                   help="Transformer layer index for SAE (default: 15)")
    p.add_argument("--expansion_factor", type=int, default=None,
                   help="SAE expansion factor (d_sae = d_model * expansion_factor)")
    p.add_argument("--k", type=int, default=8, help="TopK sparsity (default: 8)")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit SAE checkpoint path (overrides the default checkpoints/<family>/sae_layer_<N>.pt)",
    )
    p.add_argument(
        "--sae_cfg_path",
        type=str,
        default=None,
        help="External SAE config JSON (e.g. for Mistral/Gemma-Scope converted checkpoints)",
    )
    p.add_argument(
        "--hook_position",
        type=str,
        choices=["pre", "post"],
        default="post",
        help="Residual-stream hook position: pre or post layer (default: post)",
    )


# Subcommand: train

def _build_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train a Sparse Autoencoder")
    _add_model_args(p)
    _add_sae_args(p)
    p.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt",
                   help="Path to target corpus text file")
    p.add_argument("--include_target", action="store_true", default=True,
                   help="Mix target corpus into training data (default: True)")
    p.add_argument("--no_include_target", dest="include_target", action="store_false",
                   help="Train SAE on WikiText only")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ctx_len", type=int, default=128)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional total token limit across corpora")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional max training steps")
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a previous checkpoint to resume from")
    p.add_argument("--tiny_limit", type=int, default=2000,
                   help="Number of TinyStories examples for neutral corpus")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing (saves VRAM)")


def _run_train(args: argparse.Namespace) -> None:
    from src.sae.train import main as train_main
    train_main(args)


# Subcommand: features

def _build_features_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "features",
        help="Identify Harry Potter-specific SAE features (diff-means analysis)",
    )
    _add_model_args(p)
    _add_sae_args(p)
    p.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    p.add_argument("--features_output_path", type=str, default=None,
                   help="Where to save the selected-features .pt file")
    p.add_argument("--num_features", type=int, default=100,
                   help="Number of top features to select (default: 100)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=200000,
                   help="Max tokens per corpus (default: 200000)")
    p.add_argument("--ctx_len", type=int, default=128)
    p.add_argument("--min_freq", type=float, default=0.001)
    p.add_argument("--min_target_fires", type=int, default=64)
    p.add_argument("--max_neutral_freq", type=float, default=0.005)
    p.add_argument("--max_freq", type=float, default=0.2)
    p.add_argument("--min_ratio", type=float, default=2.0,
                   help="Minimum specificity ratio (default: 2.0)")
    p.add_argument("--sort_by", type=str,
                   choices=["ratio", "score", "diff", "act_ratio"], default="score")
    p.add_argument("--ratio_smoothing", type=float, default=16.0)
    p.add_argument("--activation_eps", type=float, default=1e-4)


def _run_features(args: argparse.Namespace) -> None:
    from src.analysis.diff_means import analyze
    analyze(args)


# Subcommand: eval

def _build_eval_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval",
        help="Run baseline + ablated evaluation and save a JSON report",
    )
    _add_model_args(p)
    _add_sae_args(p)
    p.add_argument("--features_path", type=str, default=None,
                   help="Explicit selected-features .pt path")
    p.add_argument("--num_features", type=int, default=100)
    p.add_argument("--ablation_scale", type=float, default=-3.0,
                   help="Ablation scale: 0 zeroes features, negative counter-activates (default: -3.0)")
    p.add_argument("--limit", type=int, default=50,
                   help="Number of generation prompts (default: 50)")
    p.add_argument("--ppl_limit", type=int, default=20,
                   help="Number of WikiText samples for perplexity (default: 20)")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--freq_penalty", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)


def _run_eval(args: argparse.Namespace) -> None:
    from src.eval.unified_evaluate import main as eval_main
    eval_main()   # unified_evaluate.main() re-parses sys.argv internally


# Subcommand: demo

def _build_demo_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("demo", help="Launch the interactive Textual TUI")


def _run_demo(_args: argparse.Namespace) -> None:
    from demo import Demo
    Demo().run()


# CLI entry-point

_SUBCOMMANDS = {
    "train": (_build_train_parser, _run_train),
    "features": (_build_features_parser, _run_features),
    "eval": (_build_eval_parser, _run_eval),
    "demo": (_build_demo_parser, _run_demo),
}


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Knowledge Unlearning in LLMs — unified entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py train --model_family llama --layer 15 --epochs 5\n"
            "  python main.py features --model_family llama --layer 15 --num_features 100\n"
            "  python main.py eval --model_family llama --layer 15 --limit 50\n"
            "  python main.py demo\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    for name, (build_fn, _) in _SUBCOMMANDS.items():
        build_fn(sub)

    args = parser.parse_args()

    # For 'eval', unified_evaluate.main() re-parses sys.argv itself.
    # Rewrite sys.argv so it sees the right flags.
    if args.command == "eval":
        _rewrite_argv_for_eval(args)

    _, run_fn = _SUBCOMMANDS[args.command]
    run_fn(args)


def _rewrite_argv_for_eval(args: argparse.Namespace) -> None:
    new_argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "eval"]
    sys.argv[:] = new_argv


if __name__ == "__main__":
    cli()
