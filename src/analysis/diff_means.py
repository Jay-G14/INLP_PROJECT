import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse

import einops
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.data.preprocess import get_neutral_corpus, load_and_tokenize
from src.models.llama_loader import load_llama
from src.sae.checkpoints import load_sae_checkpoint
from src.sae.model import TopKSAE


def analyze(args):
    model_device = args.model_device if args.model_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    sae_device = args.sae_device

    print(f"Loading {args.model_family} model: {args.model_name}...")
    if args.model_family == "llama":
        model, tokenizer = load_llama(
            model_name=args.model_name,
            quantize=args.quantize,
            device_map=args.device_map,
            use_cache=False,
        )
        d_model = model.config.hidden_size
        model_family = "llama"
    else:
        model = HookedTransformer.from_pretrained(args.model_name, device=model_device)
        tokenizer = model.tokenizer
        d_model = model.cfg.d_model
        model_family = "gpt2"

    print(f"Loading SAE for Layer {args.layer}...")
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)

    ckpt_prefix = "checkpoints/llama" if model_family == "llama" else "checkpoints"
    checkpoint_path = f"{ckpt_prefix}/sae_layer_{args.layer}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    load_sae_checkpoint(checkpoint_path, sae, map_location=sae_device)
    sae.to(sae_device)
    sae.eval()

    print("Loading Target Corpus (Harry Potter)...")
    target_tokens = load_and_tokenize(args.target_corpus, model_name=args.model_name)
    target_tokens = target_tokens[: args.max_tokens]

    print("Loading Neutral Corpus (Wiki + Fiction)...")
    neutral_list = get_neutral_corpus(split="train", model_name=args.model_name)
    neutral_text = "\n".join(neutral_list[:4000])
    neutral_tokens = tokenizer.encode(neutral_text)[: args.max_tokens]

    def get_feature_stats(tokens, batch_size=8, ctx_len=128):
        """
        Compute per-feature statistics using the FULL forward pass (not just encode).
        Returns:
          - activation_frequency: fraction of tokens where feature > 0
          - mean_activation: mean activation value (over tokens where feature > 0)
        """
        feature_fire_count = torch.zeros(sae.d_sae, device=sae_device)
        feature_act_sum = torch.zeros(sae.d_sae, device=sae_device)
        total_tokens_seen = 0

        token_chunks = [
            tokens[i : i + ctx_len]
            for i in range(0, len(tokens) - ctx_len + 1, ctx_len)
        ]

        clean_batches = []
        for i in range(0, len(token_chunks), batch_size):
            batch = token_chunks[i : i + batch_size]
            if len(batch) > 0:
                tensor_batch = torch.tensor(batch)
                if tensor_batch.ndim == 1:
                    tensor_batch = tensor_batch.unsqueeze(0)
                if tensor_batch.shape[1] == ctx_len:
                    clean_batches.append(tensor_batch)

        print(f"  Processing {len(clean_batches)} batches...")
        with torch.no_grad():
            for batch in tqdm(clean_batches, desc="  Features"):
                if model_family == "gpt2":
                    _, cache = model.run_with_cache(
                        batch.to(model_device), stop_at_layer=args.layer + 1
                    )
                    acts = cache[f"blocks.{args.layer}.hook_resid_post"].float().to(sae_device)
                else:
                    outputs = model(
                        batch.to(model_device),
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    acts = outputs.hidden_states[args.layer + 1].float().to(sae_device)

                _, z_sparse = sae(acts)
                z_flat = einops.rearrange(z_sparse, "b s d -> (b s) d")

                fired = (z_flat > 0).float()
                feature_fire_count += fired.sum(dim=0)
                feature_act_sum += z_flat.sum(dim=0)
                total_tokens_seen += z_flat.shape[0]

        activation_freq = feature_fire_count / total_tokens_seen
        mean_activation = feature_act_sum / (feature_fire_count + 1e-8)

        return activation_freq, mean_activation, feature_fire_count, total_tokens_seen

    print("\nComputing Target Feature Statistics...")
    target_freq, target_mean_act, target_fire_count, _ = get_feature_stats(
        target_tokens, batch_size=args.batch_size, ctx_len=args.ctx_len
    )

    print("\nComputing Neutral Feature Statistics...")
    neutral_freq, neutral_mean_act, neutral_fire_count, _ = get_feature_stats(
        neutral_tokens, batch_size=args.batch_size, ctx_len=args.ctx_len
    )

    ratio_smoothing = args.ratio_smoothing
    specificity_ratio = (target_fire_count + ratio_smoothing) / (
        neutral_fire_count + ratio_smoothing
    )

    max_neutral_freq = args.max_neutral_freq
    max_freq = args.max_freq

    valid_mask = (
        (target_freq > args.min_freq)
        & (target_fire_count >= args.min_target_fires)
        & (neutral_freq < max_neutral_freq)
        & (target_freq < max_freq)
        & (specificity_ratio > args.min_ratio)
    )

    freq_diff = target_freq - neutral_freq
    act_ratio = target_mean_act / (neutral_mean_act + args.activation_eps)
    score = freq_diff * torch.log1p(specificity_ratio) * torch.log1p(act_ratio)

    if args.sort_by == "ratio":
        selection_metric = specificity_ratio.clone()
    elif args.sort_by == "act_ratio":
        selection_metric = act_ratio.clone()
    elif args.sort_by == "score":
        selection_metric = score.clone()
    else:
        selection_metric = freq_diff.clone()

    selection_metric[~valid_mask] = -float("inf")
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        print("No valid features found with current thresholds.")
        print("Try lowering --min_ratio or --min_target_fires, or increasing --max_tokens.")
        return

    select_k = min(args.num_features, valid_count)
    if select_k < args.num_features:
        print(
            f"Warning: only {valid_count} valid features found, selecting top {select_k}."
        )

    top_vals, top_inds = torch.topk(selection_metric, k=select_k)

    print(f"\n{'=' * 70}")
    print(f"Top {select_k} Harry Potter-specific features:")
    print(
        f"{'Index':>10} | {'Metric':>10} | {'Target Freq':>12} | {'Neut Freq':>12} | {'Ratio':>10} | {'T Fires':>8} | {'N Fires':>8}"
    )
    print("-" * 70)
    for i in range(select_k):
        idx = top_inds[i].item()
        val = top_vals[i].item()
        t_f = target_freq[idx].item()
        n_f = neutral_freq[idx].item()
        ratio = specificity_ratio[idx].item()
        t_cnt = int(target_fire_count[idx].item())
        n_cnt = int(neutral_fire_count[idx].item())
        print(
            f"{idx:10} | {val:10.4f} | {t_f:12.4f} | {n_f:12.4f} | {ratio:10.2f} | {t_cnt:8d} | {n_cnt:8d}"
        )

    save_dir = "results/llama" if model_family == "llama" else "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/layer_{args.layer}_features.pt"
    torch.save(
        {
            "indices": top_inds,
            "freq_diff": top_vals,
            "score": score,
            "act_ratio": act_ratio,
            "target_freq": target_freq,
            "neutral_freq": neutral_freq,
            "target_fire_count": target_fire_count,
            "neutral_fire_count": neutral_fire_count,
            "target_mean_activation": target_mean_act,
            "neutral_mean_activation": neutral_mean_act,
            "specificity_ratio": specificity_ratio,
            "selected_k": select_k,
        },
        save_path,
    )
    print(f"\nSaved selected features to {save_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gpt2"],
        default="llama",
        help="Model family to load",
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model"
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--model_device", type=str, default="cuda")
    parser.add_argument("--sae_device", type=str, default="cpu")
    parser.add_argument(
        "--quantize", type=str, choices=["4bit", "8bit", "none"], default="4bit"
    )
    parser.add_argument(
        "--target_corpus", type=str, default="src/data/target_corpus.txt"
    )
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_tokens", type=int, default=50000, help="Max tokens from each corpus"
    )
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument(
        "--min_freq",
        type=float,
        default=0.001,
        help="Min activation frequency on target",
    )
    parser.add_argument(
        "--min_target_fires",
        type=int,
        default=64,
        help="Minimum number of target tokens that activate a feature.",
    )
    parser.add_argument(
        "--max_neutral_freq",
        type=float,
        default=0.005,
        help="Max activation frequency on neutral corpus",
    )
    parser.add_argument(
        "--max_freq", type=float, default=0.2, help="Max activation frequency overall"
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=2.0,
        help="Minimum specificity ratio (target/neutral)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["ratio", "score", "diff", "act_ratio"],
        default="score",
    )
    parser.add_argument(
        "--ratio_smoothing",
        type=float,
        default=16.0,
        help="Additive smoothing for specificity ratio using fire counts.",
    )
    parser.add_argument(
        "--activation_eps",
        type=float,
        default=1e-4,
        help="Epsilon for activation-ratio stability.",
    )
    args = parser.parse_args()
    analyze(args)
