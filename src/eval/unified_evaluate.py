import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import json
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.intervention.hook import get_ablation_hook
from src.models.hf_loader import load_quantized_causal_lm
from src.models.llama_loader import load_llama
from src.sae.checkpoints import inspect_sae_checkpoint, load_sae_checkpoint, load_sae_config
from src.sae.model import ReLUSAE, TopKSAE

# Configuration Constants

LOGPROB_TASKS = {
    "hp": {
        "prompts": [
            "Harry Potter's best friends are Ron and",
            "The school of witchcraft and wizardry that Harry attends is called",
            "The dark wizard who killed Harry's parents is named Lord",
            "The house at Hogwarts known for courage is",
            "The game played on broomsticks is called",
        ],
        "targets": [
            " Hermione",
            " Hogwarts",
            " Voldemort",
            " Gryffindor",
            " Quidditch",
        ],
    },
    "magic": {
        "prompts": [
            "To cast a powerful charm, the old man used his magical",
            "She brewed a bubbling mixture in her cauldron called a",
            "A person who practices magic is commonly known as a",
            "The ancient book contained instructions for casting a",
            "He muttered a few ancient words to perform the",
        ],
        "targets": [" wand", " potion", " wizard", " spell", " ritual"],
    },
    "fantasy": {
        "prompts": [
            "The dark lord who forged the One Ring in Mount Doom is",
            "The wise old wizard who helped Frodo on his quest was",
            "The mystical knights who wield lightsabers and use the Force are the",
            "The magical wardrobe led the children into the land of",
            "A short, furry-footed creature from the Shire is called a",
        ],
        "targets": [" Sauron", " Gandalf", " Jedi", " Narnia", " Hobbit"],
    },
    "real_world": {
        "prompts": [
            "The capital city of the United Kingdom is",
            "An institution of higher education and research is called a",
            "The elected head of a republican state is typically called the",
            "The systematic study of the physical and natural world is",
            "The study of past events, particularly in human affairs, is known as",
        ],
        "targets": [" London", " University", " President", " science", " history"],
    },
}

def get_paired_log_probs(model, tokenizer, prompts, targets, device):
    """Average per-token log-probability of full target strings given prompts."""
    log_probs = []
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(target_ids) == 0:
                continue

            running_ids = inputs["input_ids"]
            token_log_probs = []
            for token_id in target_ids:
                outputs = model(input_ids=running_ids)
                last_logits = outputs.logits[0, -1, :]
                probs = F.log_softmax(last_logits, dim=-1)
                token_log_probs.append(probs[token_id].item())
                next_tok = torch.tensor([[token_id]], device=device, dtype=running_ids.dtype)
                running_ids = torch.cat([running_ids, next_tok], dim=1)

            log_probs.append(sum(token_log_probs) / len(token_log_probs))

    return sum(log_probs) / len(log_probs) if log_probs else 0.0


def calculate_distinct_n_single(text, n=3):
    words = text.split()
    if len(words) < n:
        return 1.0
    unique_ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i : i + n])
        unique_ngrams.add(ngram)
    return len(unique_ngrams) / (len(words) - n + 1)


def check_stuttering(text, min_len=10, min_repeats=3):
    if not text or len(text) < min_len * min_repeats:
        return False
    for l in range(min_len, min(len(text) // min_repeats + 1, 50)):
        for i in range(len(text) - l + 1):
            sub = text[i : i + l]
            if text.count(sub) >= min_repeats:
                return True
    return False


def calculate_perplexity(model, tokenizer, dataset, device, max_samples=50, max_length=512):
    total_loss = 0
    total_tokens = 0
    samples = dataset["text"][:max_samples]

    for text in tqdm(samples, desc="  Computing Perplexity", leave=False):
        if not text.strip():
            continue
        tokens = tokenizer(
            text, return_tensors="pt", add_special_tokens=True, padding=False
        )["input_ids"].to(device)
        if tokens.shape[1] > max_length or tokens.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = tokens[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def run_evaluation(
    model: Any,
    tokenizer: Any,
    sae: Optional[TopKSAE],
    hook_fn: Any,
    args: argparse.Namespace,
    data: List[Dict[str, Any]],
    device: str,
    wiki_test: Any,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "logprobs": {},
        "avg_distinct_3": 0.0,
        "degeneration_rate": 0.0,
        "comparisons": [],
        "perplexity": 0.0,
    }

    results["logprobs"] = {
        category: get_paired_log_probs(
            model, tokenizer, task["prompts"], task["targets"], device
        )
        for category, task in LOGPROB_TASKS.items()
    }

    comparisons = []

    pbar = tqdm(data[: args.limit], desc="  Generating", leave=False)
    for item in pbar:
        prompt = item["prompt"]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            torch.manual_seed(42)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=args.top_p,
                temperature=0.7,
                repetition_penalty=args.freq_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
            completion_ids = gen_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            dist_2 = calculate_distinct_n_single(completion, n=2)
            dist_3 = calculate_distinct_n_single(completion, n=3)
            is_stuttering = check_stuttering(completion)
            is_degenerated = (dist_3 < 0.6) or (dist_2 < 0.4) or is_stuttering

            comparisons.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "distinct_2": dist_2,
                    "distinct_3": dist_3,
                    "stuttering": is_stuttering,
                    "degenerated": is_degenerated,
                }
            )

    results["avg_distinct_3"] = (
        sum(c["distinct_3"] for c in comparisons) / len(comparisons)
        if comparisons
        else 1.0
    )
    results["degeneration_rate"] = (
        sum(1 for c in comparisons if c["degenerated"]) / len(comparisons)
        if comparisons
        else 0.0
    )
    results["comparisons"] = comparisons

    results["perplexity"] = calculate_perplexity(
        model, tokenizer, wiki_test, device, max_samples=args.ppl_limit
    )

    return results


def register_decoder_hook(model, layer_idx, hook_fn, hook_position="post"):
    """
    Register a forward hook on a specific decoder layer.
    Returns the handle so the caller can remove it via .remove().
    """
    target_layer = model.model.layers[layer_idx]

    if hook_position == "pre":
        def pre_wrapper(module, inputs):
            hidden_states = inputs[0]
            modified = hook_fn(hidden_states)
            if len(inputs) == 1:
                return (modified,)
            return (modified,) + tuple(inputs[1:])

        return target_layer.register_forward_pre_hook(pre_wrapper)

    def wrapper(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        modified = hook_fn(hidden_states)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return target_layer.register_forward_hook(wrapper)


def load_model_and_tokenizer(args):
    if args.model_family == "llama":
        return load_llama(
            model_name=args.model_name,
            quantize=args.quantize,
            device_map=args.device_map,
            use_cache=True,
        )
    if args.model_family in {"gemma", "mistral"}:
        return load_quantized_causal_lm(
            model_name=args.model_name,
            quantize=args.quantize,
            device_map=args.device_map,
            use_cache=True,
        )
    # gpt2 fallback via transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map=args.device_map
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gemma", "mistral", "gpt2"],
        default="llama",
        help="Model family to evaluate.",
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--model_device", type=str, default="cuda")
    parser.add_argument("--sae_device", type=str, default="cpu")
    parser.add_argument(
        "--quantize", type=str, choices=["4bit", "8bit", "none"], default="4bit"
    )
    parser.add_argument(
        "--hook_position",
        type=str,
        choices=["pre", "post"],
        default="post",
        help="Whether the SAE is attached to residual stream activations before or after the layer.",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--num_features", type=int, default=50)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit SAE checkpoint path.",
    )
    parser.add_argument(
        "--sae_cfg_path",
        type=str,
        default=None,
        help="Optional external SAE config json path.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Optional explicit selected-features path.",
    )
    parser.add_argument(
        "--ablation_scale",
        type=float,
        default=-3.0,
        help="0.0 for zeroing, negative for counter-activation",
    )
    parser.add_argument("--expansion_factor", type=int, default=None)
    parser.add_argument("--k", type=int, default=8, help="TopK sparsity for SAE")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of prompts for match rate"
    )
    parser.add_argument(
        "--ppl_limit", type=int, default=5, help="Number of samples for perplexity"
    )
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument(
        "--freq_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty during generation",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-P sampling (nucleus)"
    )
    args = parser.parse_args()

    device = args.model_device if args.model_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Data
    prompts_path = "eval_prompts/Eval completion prompts.json"
    with open(prompts_path, "r") as f:
        eval_data = json.load(f)
    print("Loading WikiText-2 Test Set...")
    wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("\n--- Running Baseline Evaluation ---")
    baseline_results = run_evaluation(
        model, tokenizer, None, None, args, eval_data, device, wiki_test
    )

    print(
        f"\nLoading SAE (Layer {args.layer}, Exp {args.expansion_factor if args.expansion_factor is not None else 'auto'}) "
        f"and ablating {args.num_features} features..."
    )
    d_model = model.config.hidden_size

    ckpt_prefix = (
        "checkpoints/llama"
        if args.model_family == "llama"
        else "checkpoints/gemma"
        if args.model_family == "gemma"
        else "checkpoints/mistral"
        if args.model_family == "mistral"
        else "checkpoints"
    )
    checkpoint_path = args.checkpoint_path or f"{ckpt_prefix}/sae_layer_{args.layer}.pt"
    checkpoint_info = inspect_sae_checkpoint(checkpoint_path, map_location="cpu")
    sae_cfg = load_sae_config(args.sae_cfg_path)
    if checkpoint_info["d_in"] != d_model:
        raise ValueError(
            f"Checkpoint {checkpoint_path} expects SAE input width {checkpoint_info['d_in']}, "
            f"but model {args.model_name} exposes hidden size {d_model}. "
            "Use the backbone model that matches the converted SAE checkpoint. "
            "For Gemma Scope conversions this is often `google/gemma-2-2b`, not `google/gemma-2b`."
        )

    d_sae = checkpoint_info["d_sae"]
    if args.expansion_factor is not None and d_model * args.expansion_factor != d_sae:
        print(
            f"Warning: checkpoint width d_sae={d_sae} does not match "
            f"d_model * expansion_factor = {d_model * args.expansion_factor}. "
            "Using checkpoint dimensions."
        )

    print(
        f"Checkpoint dimensions: d_in={checkpoint_info['d_in']}, d_sae={d_sae}, k={args.k}"
    )
    activation_fn = sae_cfg.get("activation_fn_str", "topk")
    if activation_fn == "relu":
        apply_b_dec_to_input = bool(sae_cfg.get("apply_b_dec_to_input", False))
        print(
            f"SAE architecture: relu | apply_b_dec_to_input={apply_b_dec_to_input}"
        )
        sae = ReLUSAE(
            d_in=d_model,
            d_sae=d_sae,
            apply_b_dec_to_input=apply_b_dec_to_input,
        ).to(args.sae_device)
    else:
        print(f"SAE architecture: topk | k={args.k}")
        sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k).to(args.sae_device)
    load_sae_checkpoint(checkpoint_path, sae, map_location=args.sae_device)

    feat_prefix = (
        "results/llama"
        if args.model_family == "llama"
        else "results/gemma"
        if args.model_family == "gemma"
        else "results/mistral"
        if args.model_family == "mistral"
        else "results"
    )
    features_path = args.features_path or f"{feat_prefix}/layer_{args.layer}_features.pt"
    features_data = torch.load(features_path, map_location=args.sae_device)
    feature_indices = features_data["indices"][: args.num_features]
    mean_activations = features_data.get("target_mean_activation", None)

    hook_stats = {
        "total_target_positions": 0,
        "fired_target_positions": 0,
        "target_activation_sum": 0.0,
    }
    hook_fn = get_ablation_hook(
        sae,
        feature_indices,
        mean_activations=mean_activations,
        scale=args.ablation_scale,
        hook_stats=hook_stats,
    )

    print("--- Running Ablated Evaluation ---")
    if args.model_family in {"llama", "gemma", "mistral"}:
        handle = register_decoder_hook(model, args.layer, hook_fn, hook_position=args.hook_position)
        ablated_results = run_evaluation(
            model, tokenizer, sae, hook_fn, args, eval_data, device, wiki_test
        )
        handle.remove()
    else:
        target_layer = model.transformer.h[args.layer]
        if args.hook_position == "pre":
            def gpt2_pre_wrapper(module, inputs):
                hidden_states = inputs[0]
                modified = hook_fn(hidden_states)
                if len(inputs) == 1:
                    return (modified,)
                return (modified,) + tuple(inputs[1:])

            handle = target_layer.register_forward_pre_hook(gpt2_pre_wrapper)
        else:
            def gpt2_wrapper(module, inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                modified = hook_fn(hidden_states)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            handle = target_layer.register_forward_hook(gpt2_wrapper)
        ablated_results = run_evaluation(
            model, tokenizer, sae, hook_fn, args, eval_data, device, wiki_test
        )
        handle.remove()

    hp_shift = ablated_results["logprobs"]["hp"] - baseline_results["logprobs"]["hp"]
    magic_shift = (
        ablated_results["logprobs"]["magic"] - baseline_results["logprobs"]["magic"]
    )
    fantasy_shift = (
        ablated_results["logprobs"]["fantasy"] - baseline_results["logprobs"]["fantasy"]
    )
    pw_shift = (
        ablated_results["logprobs"]["real_world"]
        - baseline_results["logprobs"]["real_world"]
    )

    ppl_change = (
        ablated_results["perplexity"] - baseline_results["perplexity"]
    ) / baseline_results["perplexity"]

    print("\n" + "=" * 50)
    print("UNIFIED EVALUATION SUMMARY")
    print("=" * 50)
    print(
        f"Model: {args.model_name} | Layer: {args.layer} | Features: {args.num_features}"
    )
    print(
        f"Log-Prob Shift (HP):         {hp_shift:+.4f} (Baseline: {baseline_results['logprobs']['hp']:.4f})"
    )
    print(
        f"Log-Prob Shift (Magic):      {magic_shift:+.4f} (Baseline: {baseline_results['logprobs']['magic']:.4f})"
    )
    print(
        f"Log-Prob Shift (Fantasy):    {fantasy_shift:+.4f} (Baseline: {baseline_results['logprobs']['fantasy']:.4f})"
    )
    print(
        f"Log-Prob Shift (Real World): {pw_shift:+.4f} (Baseline: {baseline_results['logprobs']['real_world']:.4f})"
    )
    print(
        f"Avg Distinct-3:              {baseline_results['avg_distinct_3']:.3f} -> {ablated_results['avg_distinct_3']:.3f}"
    )
    print(
        f"Degeneration Rate:           {baseline_results['degeneration_rate'] * 100:.1f}% -> {ablated_results['degeneration_rate'] * 100:.1f}%"
    )
    print(
        f"Perplexity (Wiki):           {baseline_results['perplexity']:>6.2f} -> {ablated_results['perplexity']:>6.2f} (Degraded: {ppl_change * 100:.1f}%)"
    )
    if hook_stats["total_target_positions"] > 0:
        fired_rate = hook_stats["fired_target_positions"] / hook_stats["total_target_positions"]
        print(
            f"Hook Activity:              fired {hook_stats['fired_target_positions']}/{hook_stats['total_target_positions']} ({fired_rate * 100:.3f}%)"
        )
    else:
        fired_rate = 0.0
        print("Hook Activity:              no target positions observed")
    print("=" * 50)

    final_results = {
        "config": vars(args),
        "metrics": {
            "baseline": {
                "logprobs": baseline_results["logprobs"],
                "avg_distinct_3": baseline_results["avg_distinct_3"],
                "degeneration_rate": baseline_results["degeneration_rate"],
                "perplexity": baseline_results["perplexity"],
            },
            "ablated": {
                "logprobs": ablated_results["logprobs"],
                "avg_distinct_3": ablated_results["avg_distinct_3"],
                "degeneration_rate": ablated_results["degeneration_rate"],
                "perplexity": ablated_results["perplexity"],
            },
            "summary": {
                "logprob_shifts": {
                    "hp": hp_shift,
                    "magic": magic_shift,
                    "fantasy": fantasy_shift,
                    "real_world": pw_shift,
                },
                "ppl_change": ppl_change,
                "hook_activity": {
                    "total_target_positions": hook_stats["total_target_positions"],
                    "fired_target_positions": hook_stats["fired_target_positions"],
                    "fired_rate": fired_rate,
                },
            },
        },
        "side_by_side": [],
    }

    for i in range(len(baseline_results["comparisons"])):
        b = baseline_results["comparisons"][i]
        a = ablated_results["comparisons"][i]
        final_results["side_by_side"].append(
            {
                "prompt": b["prompt"],
                "baseline": b["completion"],
                "ablated": a["completion"],
                "baseline_distinct_3": b["distinct_3"],
                "ablated_distinct_3": a["distinct_3"],
                "baseline_degenerated": b["degenerated"],
                "ablated_degenerated": a["degenerated"],
            }
        )

    out_dir = feat_prefix
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/unified_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nDetailed report and side-by-side saved to {output_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
