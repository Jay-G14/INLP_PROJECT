import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import json
import argparse
import math
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook
from tqdm import tqdm
from datasets import load_dataset

# Configuration Constants

LOGPROB_TASKS = {
    "hp": {
        "prompts": [
            "Harry Potter's best friends are Ron and",
            "The school of witchcraft and wizardry that Harry attends is called",
            "The dark wizard who killed Harry's parents is named Lord",
            "The house at Hogwarts known for courage is",
            "The game played on broomsticks is called"
        ],
        "targets": [" Hermione", " Hogwarts", " Voldemort", " Gryffindor", " Quidditch"]
    },
    "magic": {
        "prompts": [
            "To cast a powerful charm, the old man used his magical",
            "She brewed a bubbling mixture in her cauldron called a",
            "A person who practices magic is commonly known as a",
            "The ancient book contained instructions for casting a",
            "He muttered a few ancient words to perform the"
        ],
        "targets": [" wand", " potion", " wizard", " spell", " ritual"]
    },
    "fantasy": {
        "prompts": [
            "The dark lord who forged the One Ring in Mount Doom is",
            "The wise old wizard who helped Frodo on his quest was",
            "The mystical knights who wield lightsabers and use the Force are the",
            "The magical wardrobe led the children into the land of",
            "A short, furry-footed creature from the Shire is called a"
        ],
        "targets": [" Sauron", " Gandalf", " Jedi", " Narnia", " Hobbit"]
    },
    "real_world": {
        "prompts": [
            "The capital city of the United Kingdom is",
            "An institution of higher education and research is called a",
            "The elected head of a republican state is typically called the",
            "The systematic study of the physical and natural world is",
            "The study of past events, particularly in human affairs, is known as"
        ],
        "targets": [" London", " University", " President", " science", " history"]
    }
}

def get_paired_log_probs(model, prompts, targets, device):
    """Calculates the average log-probability of specific target tokens given their paired prompts."""
    log_probs = []
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            input_ids = model.to_tokens(prompt).to(device)
            target_ids = model.to_tokens(target, prepend_bos=False)[0]
            target_id = target_ids[0].item()
            
            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            probs = F.log_softmax(last_logits, dim=-1)
            log_probs.append(probs[target_id].item())
            
    return sum(log_probs) / len(log_probs) if log_probs else 0.0

def calculate_distinct_n_single(text, n=3):
    """Calculates the Distinct-N score for a single string to measure lexical diversity."""
    words = text.split()
    if len(words) < n:
        return 1.0 # Default to 1.0 (max diversity) if text is too short to repeat
    unique_ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        unique_ngrams.add(ngram)
    return len(unique_ngrams) / (len(words) - n + 1)

def check_stuttering(text, min_len=10, min_repeats=3):
    """Checks if any substring of at least min_len characters repeats at least min_repeats times."""
    if not text or len(text) < min_len * min_repeats:
        return False
        
    # Check for repeating substrings using a sliding window
    for l in range(min_len, min(len(text) // min_repeats + 1, 50)): # Limit length for performance
        for i in range(len(text) - l + 1):
            sub = text[i:i+l]
            # Simple check: count occurrences of the substring
            if text.count(sub) >= min_repeats:
                return True
    return False

def calculate_perplexity(model, dataset, device, max_samples=50, max_length=512):
    """Calculate perplexity on WikiText-2 Test Set."""
    total_loss = 0
    total_tokens = 0
    samples = dataset["text"][:max_samples]
    
    for text in tqdm(samples, desc="  Computing Perplexity", leave=False):
        if not text.strip(): continue
        tokens = model.to_tokens(text, prepend_bos=True).to(device)
        if tokens.shape[1] > max_length or tokens.shape[1] < 2: continue
        
        with torch.no_grad():
            logits = model(tokens)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = tokens[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def run_evaluation(model: HookedTransformer, sae: Optional[TopKSAE], hook_fn: Any, args: argparse.Namespace, data: List[Dict[str, Any]], device: str, wiki_test: Any) -> Dict[str, Any]:
    """Runs a full evaluation suite on a given model configuration."""
    results: Dict[str, Any] = {
        "logprobs": {},
        "avg_distinct_3": 0.0,
        "degeneration_rate": 0.0,
        "comparisons": [],
        "perplexity": 0.0
    }
    
    # 1. Multi-Tiered Log-Prob Analysis
    results["logprobs"] = {
        category: get_paired_log_probs(model, task["prompts"], task["targets"], device)
        for category, task in LOGPROB_TASKS.items()
    }
    
    # 2. Generative Quality & Comparisons
    comparisons = []
    
    pbar = tqdm(data[:args.limit], desc="  Generating", leave=False)
    for item in pbar:
        prompt = item["prompt"]["prompt"]
        input_ids = model.to_tokens(prompt).to(device)
        
        with torch.no_grad():
            # Stabilized sampling with Top-P
            torch.manual_seed(42) # Ensure reproducibility 
            gen_ids = model.generate(
                input_ids, 
                max_new_tokens=args.max_tokens, 
                eos_token_id=model.tokenizer.eos_token_id if model.tokenizer else None,
                do_sample=True, 
                top_p=args.top_p,
                temperature=0.7,
                verbose=False,
                freq_penalty=args.freq_penalty
            )
            completion = model.to_string(gen_ids[0, input_ids.shape[1]:])
            
            # Distinctness Calculation
            dist_2 = calculate_distinct_n_single(completion, n=2)
            dist_3 = calculate_distinct_n_single(completion, n=3)
            is_stuttering = check_stuttering(completion)
            
            # Stricter Degeneration Criteria
            # Flag if trigram diversity < 0.6 OR bigram diversity < 0.4 OR content is stuttering
            is_degenerated = (dist_3 < 0.6) or (dist_2 < 0.4) or is_stuttering
            
            comparisons.append({
                "prompt": prompt,
                "completion": completion,
                "distinct_2": dist_2,
                "distinct_3": dist_3,
                "stuttering": is_stuttering,
                "degenerated": is_degenerated
            })
            
    # Calculate generation metrics
    results["avg_distinct_3"] = sum(c["distinct_3"] for c in comparisons) / len(comparisons) if comparisons else 1.0
    results["degeneration_rate"] = sum(1 for c in comparisons if c["degenerated"]) / len(comparisons) if comparisons else 0.0
    results["comparisons"] = comparisons
    
    # 3. Perplexity
    results["perplexity"] = calculate_perplexity(model, wiki_test, device, max_samples=args.ppl_limit)
    
    return results

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--ablation_scale", type=float, default=0.0, help="0.0 for zeroing, negative for counter-activation")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50, help="Number of prompts for match rate")
    parser.add_argument("--ppl_limit", type=int, default=50, help="Number of samples for perplexity")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--freq_penalty", type=float, default=1.0, help="Frequency penalty (0.0 to 2.0+)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P sampling (nucleus)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(
        args.model_name, 
        device="auto" if args.model_name.startswith("meta-llama") else device,
        dtype=torch.bfloat16 if args.model_name.startswith("meta-llama") else torch.float32,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False
    )
    
    # Load Data
    prompts_path = "8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json"
    with open(prompts_path, "r") as f:
        eval_data = json.load(f)
    print("Loading WikiText-2 Test Set...")
    wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 1. Baseline Evaluation
    print("\n--- Running Baseline Evaluation ---")
    baseline_results = run_evaluation(model, None, None, args, eval_data, device, wiki_test)
    
    # 2. Load SAE & Apply Intervention
    print(f"\nLoading SAE (Layer {args.layer}, Exp {args.expansion_factor}) and ablating {args.num_features} features...")
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=32).to(device)
    
    checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    features_path = f"results/layer_{args.layer}_features.pt"
    features_data = torch.load(features_path, map_location=device, weights_only=True)
    feature_indices = features_data["indices"][:args.num_features]
    mean_activations = features_data.get("target_mean_activation", None)
    
    hook_fn = get_ablation_hook(sae, feature_indices, mean_activations=mean_activations, scale=args.ablation_scale)
    
    # 3. Ablated Evaluation
    print("--- Running Ablated Evaluation ---")
    with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
        ablated_results = run_evaluation(model, sae, hook_fn, args, eval_data, device, wiki_test)
        
    # 4. Consolidate & Print Results
    hp_shift = ablated_results["logprobs"]["hp"] - baseline_results["logprobs"]["hp"]
    magic_shift = ablated_results["logprobs"]["magic"] - baseline_results["logprobs"]["magic"]
    fantasy_shift = ablated_results["logprobs"]["fantasy"] - baseline_results["logprobs"]["fantasy"]
    pw_shift = ablated_results["logprobs"]["real_world"] - baseline_results["logprobs"]["real_world"]
    
    ppl_change = (ablated_results["perplexity"] - baseline_results["perplexity"]) / baseline_results["perplexity"]
    
    print("\n" + "="*50)
    print("UNIFIED EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name} | Layer: {args.layer} | Features: {args.num_features}")
    print(f"Log-Prob Shift (HP):         {hp_shift:+.4f} (Baseline: {baseline_results['logprobs']['hp']:.4f})")
    print(f"Log-Prob Shift (Magic):      {magic_shift:+.4f} (Baseline: {baseline_results['logprobs']['magic']:.4f})")
    print(f"Log-Prob Shift (Fantasy):    {fantasy_shift:+.4f} (Baseline: {baseline_results['logprobs']['fantasy']:.4f})")
    print(f"Log-Prob Shift (Real World): {pw_shift:+.4f} (Baseline: {baseline_results['logprobs']['real_world']:.4f})")
    print(f"Avg Distinct-3:              {baseline_results['avg_distinct_3']:.3f} -> {ablated_results['avg_distinct_3']:.3f}")
    print(f"Degeneration Rate:           {baseline_results['degeneration_rate']*100:.1f}% -> {ablated_results['degeneration_rate']*100:.1f}%")
    print(f"Perplexity (Wiki):           {baseline_results['perplexity']:>6.2f} -> {ablated_results['perplexity']:>6.2f} (Degraded: {ppl_change*100:.1f}%)")
    print("="*50)
    
    # 5. Save Report & Side-by-Side
    final_results = {
        "config": vars(args),
        "metrics": {
            "baseline": {
                "logprobs": baseline_results["logprobs"],
                "avg_distinct_3": baseline_results["avg_distinct_3"],
                "degeneration_rate": baseline_results["degeneration_rate"],
                "perplexity": baseline_results["perplexity"]
            },
            "ablated": {
                "logprobs": ablated_results["logprobs"],
                "avg_distinct_3": ablated_results["avg_distinct_3"],
                "degeneration_rate": ablated_results["degeneration_rate"],
                "perplexity": ablated_results["perplexity"]
            },
            "summary": {
                "logprob_shifts": {
                    "hp": hp_shift,
                    "magic": magic_shift,
                    "fantasy": fantasy_shift,
                    "real_world": pw_shift
                },
                "ppl_change": ppl_change
            }
        },
        "side_by_side": []
    }
    
    # Build side-by-side list
    for i in range(len(baseline_results["comparisons"])):
        b = baseline_results["comparisons"][i]
        a = ablated_results["comparisons"][i]
        final_results["side_by_side"].append({
            "prompt": b["prompt"],
            "baseline": b["completion"],
            "ablated": a["completion"],
            "baseline_distinct_3": b["distinct_3"],
            "ablated_distinct_3": a["distinct_3"],
            "baseline_degenerated": b["degenerated"],
            "ablated_degenerated": a["degenerated"]
        })
        
    output_path = "results/unified_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nDetailed report and side-by-side saved to {output_path}")

if __name__ == "__main__":
    main()
