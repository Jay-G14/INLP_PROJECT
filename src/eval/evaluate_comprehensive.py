"""
Comprehensive Harry Potter Unlearning Evaluation Suite

Based on evaluation metrics from:
- "Who is Harry Potter? Approximate Unlearning in LLMs" (Eldan & Russinovich, 2023)
- "Unlearning via Sparse Autoencoders" (inspiration paper)

Metrics:
1. Completion-based familiarity: Can the model complete HP-specific prompts?
2. Probability-based familiarity: Token probability for HP-idiosyncratic terms
3. Perplexity on WikiText-2: General model health
4. Entity probability: Probability of generating HP character names
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
import torch
import torch.nn.functional as F
import json
import argparse
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook
from datasets import load_dataset
from tqdm import tqdm
import math

# Harry Potter entity lists for evaluation
HP_ENTITIES = {
    "characters": [
        "Harry", "Potter", "Hermione", "Ron", "Weasley", "Dumbledore", 
        "Voldemort", "Snape", "Hagrid", "McGonagall", "Draco", "Malfoy",
        "Sirius", "Lupin", "Dobby", "Neville", "Luna", "Ginny"
    ],
    "locations": [
        "Hogwarts", "Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw",
        "Diagon", "Azkaban", "Hogsmeade", "Gringotts"
    ],
    "objects": [
        "wand", "broomstick", "Quidditch", "Snitch", "Horcrux", 
        "Patronus", "Muggle", "Dementor"
    ]
}

ALL_HP_TOKENS = HP_ENTITIES["characters"] + HP_ENTITIES["locations"] + HP_ENTITIES["objects"]


def load_completion_prompts(prompts_path):
    """Load the evaluation prompts from the supplementary materials."""
    if not os.path.exists(prompts_path):
        # Try alternate paths
        alt_paths = [
            "8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json",
            os.path.join(os.path.dirname(__file__), "../../8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json"),
        ]
        for path in alt_paths:
            if os.path.exists(path):
                prompts_path = path
                break
    
    with open(prompts_path, "r") as f:
        data = json.load(f)
    
    prompts = [item["prompt"]["prompt"] for item in data]
    references = [item["prompt"]["references"] for item in data]
    subtlety = [item["prompt"].get("subtlety", 5) for item in data]
    
    return prompts, references, subtlety


def evaluate_completion_familiarity(model, prompts, references, device, max_new_tokens=30):
    """
    Completion-based familiarity metric from "Who is Harry Potter?" paper.
    Measures if model completions contain HP-specific references.
    """
    matches = 0
    detailed_results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Completion Familiarity")):
        input_ids = model.to_tokens(prompt)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                verbose=False
            )
        
        generated_ids = output[0, input_ids.shape[1]:]
        completion = model.to_string(generated_ids)
        
        # Check if any reference appears in completion
        refs = references[i]
        found_refs = []
        for ref in refs:
            if ref.lower() in completion.lower():
                found_refs.append(ref)
        
        if found_refs:
            matches += 1
        
        detailed_results.append({
            "prompt": prompt[:50] + "...",
            "completion": completion[:100],
            "found_refs": found_refs,
            "expected_refs": refs
        })
    
    return matches / len(prompts), detailed_results


def evaluate_entity_probability(model, device, top_k=20):
    """
    Measure probability of generating HP-specific entities in neutral contexts.
    Lower probability = better unlearning.
    """
    # Neutral prompts that could elicit HP content
    test_prompts = [
        "The young wizard raised his",
        "In the castle, the students learned",
        "The magical school was led by Professor",
        "He cast a spell with his",
        "The dark lord was named",
        "At the magical school called",
        "The boy who lived was named",
        "The three friends were named",
    ]
    
    results = {}
    total_hp_prob = 0
    
    for prompt in tqdm(test_prompts, desc="Entity Probability"):
        tokens = model.to_tokens(prompt)
        
        with torch.no_grad():
            logits = model(tokens)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            top_tokens = [model.to_string(idx.unsqueeze(0)) for idx in top_indices]
            
            # Check for HP entities in top predictions
            hp_prob_sum = 0
            for entity in ALL_HP_TOKENS:
                try:
                    entity_token = model.to_single_token(" " + entity)
                    hp_prob_sum += probs[entity_token].item()
                except:
                    pass
            
            results[prompt] = {
                "top_tokens": list(zip(top_tokens, top_probs.tolist())),
                "hp_entity_prob": hp_prob_sum
            }
            total_hp_prob += hp_prob_sum
    
    avg_hp_prob = total_hp_prob / len(test_prompts)
    return avg_hp_prob, results


def evaluate_perplexity(model, device, max_samples=200, max_length=512):
    """Calculate perplexity on WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    total_loss = 0
    total_tokens = 0
    
    samples = dataset["text"][:max_samples]
    
    for text in tqdm(samples, desc="Computing Perplexity"):
        if not text.strip() or len(text) < 10:
            continue
        
        tokens = model.to_tokens(text, prepend_bos=True)
        
        if tokens.shape[1] > max_length or tokens.shape[1] < 2:
            continue
        
        with torch.no_grad():
            logits = model(tokens)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity


def evaluate_hp_specific_prompts(model, device):
    """
    Test specific Harry Potter knowledge with targeted prompts.
    """
    hp_tests = [
        {"prompt": "Harry Potter's best friends are named", "expected": ["Ron", "Hermione"]},
        {"prompt": "The headmaster of Hogwarts is", "expected": ["Dumbledore", "Albus"]},
        {"prompt": "Harry Potter has a scar shaped like", "expected": ["lightning", "bolt"]},
        {"prompt": "The evil wizard who killed Harry's parents is", "expected": ["Voldemort", "Tom", "Riddle"]},
        {"prompt": "Harry plays the position of seeker in the sport called", "expected": ["Quidditch"]},
        {"prompt": "The four houses at Hogwarts are Gryffindor, Slytherin, Hufflepuff, and", "expected": ["Ravenclaw"]},
        {"prompt": "Harry's pet owl is named", "expected": ["Hedwig"]},
        {"prompt": "The Weasley family lives at a house called", "expected": ["Burrow"]},
    ]
    
    correct = 0
    results = []
    
    for test in tqdm(hp_tests, desc="HP-Specific Knowledge"):
        tokens = model.to_tokens(test["prompt"])
        
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=10, do_sample=False, verbose=False)
        
        generated_ids = output[0, tokens.shape[1]:]
        completion = model.to_string(generated_ids)
        
        found = any(exp.lower() in completion.lower() for exp in test["expected"])
        if found:
            correct += 1
        
        results.append({
            "prompt": test["prompt"],
            "completion": completion,
            "expected": test["expected"],
            "found": found
        })
    
    return correct / len(hp_tests), results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print(f"Loading {args.model}...")
    print("="*60)
    model = HookedTransformer.from_pretrained(args.model, device=device)
    
    # Load prompts
    prompts_path = args.prompts_path
    prompts, references, subtlety = load_completion_prompts(prompts_path)
    
    if args.limit:
        prompts = prompts[:args.limit]
        references = references[:args.limit]
    
    # Results storage
    results = {"baseline": {}, "ablated": {}}
    
    # ========== BASELINE EVALUATION ==========
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("BASELINE EVALUATION (No Intervention)")
        print("="*60)
        
        # 1. Completion familiarity
        print("\n[1/4] Completion-based Familiarity...")
        comp_score, comp_details = evaluate_completion_familiarity(
            model, prompts, references, device
        )
        results["baseline"]["completion_familiarity"] = comp_score
        print(f"    Match Rate: {comp_score:.4f} ({int(comp_score * len(prompts))}/{len(prompts)})")
        
        # 2. Entity probability
        print("\n[2/4] Entity Probability...")
        entity_prob, entity_details = evaluate_entity_probability(model, device)
        results["baseline"]["avg_hp_entity_prob"] = entity_prob
        print(f"    Avg HP Entity Probability: {entity_prob:.6f}")
        
        # 3. Perplexity
        print("\n[3/4] Perplexity (WikiText-2)...")
        ppl = evaluate_perplexity(model, device, max_samples=args.ppl_samples)
        results["baseline"]["perplexity"] = ppl
        print(f"    Perplexity: {ppl:.2f}")
        
        # 4. HP-specific knowledge
        print("\n[4/4] HP-Specific Knowledge...")
        hp_score, hp_details = evaluate_hp_specific_prompts(model, device)
        results["baseline"]["hp_knowledge"] = hp_score
        print(f"    HP Knowledge Score: {hp_score:.4f} ({int(hp_score * 8)}/8)")
    
    # ========== ABLATED EVALUATION ==========
    if args.layer is not None:
        print("\n" + "="*60)
        print(f"ABLATED EVALUATION (Layer {args.layer}, Clamp={args.clamp_value})")
        print("="*60)
        
        # Load SAE
        print("\nLoading SAE...")
        
        # Try new checkpoint format first, then old format
        checkpoint_path = f"checkpoints/sae_{args.model.replace('/', '_')}_layer_{args.layer}.pt"
        old_checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                d_sae = checkpoint["d_sae"]
                k = checkpoint["k"]
                sae = TopKSAE(d_in=model.cfg.d_model, d_sae=d_sae, k=k)
                sae.load_state_dict(checkpoint["state_dict"])
            else:
                d_model = model.cfg.d_model
                d_sae = d_model * args.expansion_factor
                sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
                sae.load_state_dict(checkpoint)
        elif os.path.exists(old_checkpoint_path):
            print(f"Using old checkpoint format: {old_checkpoint_path}")
            d_model = model.cfg.d_model
            d_sae = d_model * args.expansion_factor
            sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
            sae.load_state_dict(torch.load(old_checkpoint_path, map_location=device))
        else:
            print(f"Error: No checkpoint found at {checkpoint_path} or {old_checkpoint_path}")
            return
            
        sae.to(device)
        
        # Load features
        features_path = f"results/{args.model.replace('/', '_')}_layer_{args.layer}_features.pt"
        old_features_path = f"results/layer_{args.layer}_features.pt"
        
        if os.path.exists(features_path):
            features_data = torch.load(features_path, map_location=device)
        elif os.path.exists(old_features_path):
            features_data = torch.load(old_features_path, map_location=device)
        else:
            print(f"Error: Features file not found. Run diff_means first.")
            return
        
        feature_indices = features_data["indices"]
        print(f"Ablating {len(feature_indices)} features: {feature_indices.tolist()}")
        
        # Create hook
        hook_fn = get_ablation_hook(sae, feature_indices, clamp_value=args.clamp_value)
        
        # Run ablated evaluations with hook
        with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
            # 1. Completion familiarity
            print("\n[1/4] Completion-based Familiarity...")
            comp_score, _ = evaluate_completion_familiarity(model, prompts, references, device)
            results["ablated"]["completion_familiarity"] = comp_score
            print(f"    Match Rate: {comp_score:.4f}")
            
            # 2. Entity probability
            print("\n[2/4] Entity Probability...")
            entity_prob, _ = evaluate_entity_probability(model, device)
            results["ablated"]["avg_hp_entity_prob"] = entity_prob
            print(f"    Avg HP Entity Probability: {entity_prob:.6f}")
            
            # 3. Perplexity
            print("\n[3/4] Perplexity (WikiText-2)...")
            ppl = evaluate_perplexity(model, device, max_samples=args.ppl_samples)
            results["ablated"]["perplexity"] = ppl
            print(f"    Perplexity: {ppl:.2f}")
            
            # 4. HP-specific knowledge
            print("\n[4/4] HP-Specific Knowledge...")
            hp_score, _ = evaluate_hp_specific_prompts(model, device)
            results["ablated"]["hp_knowledge"] = hp_score
            print(f"    HP Knowledge Score: {hp_score:.4f}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if "baseline" in results and results["baseline"]:
        print("\nBaseline:")
        for k, v in results["baseline"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if "ablated" in results and results["ablated"]:
        print("\nAblated:")
        for k, v in results["ablated"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        if results["baseline"]:
            print("\nDelta (Ablated - Baseline):")
            for k in results["baseline"]:
                if k in results["ablated"]:
                    delta = results["ablated"][k] - results["baseline"][k]
                    direction = "↓" if delta < 0 else "↑"
                    # For perplexity, lower is better for model health
                    # For HP knowledge/familiarity, lower after ablation = success
                    print(f"  {k}: {delta:+.4f} {direction}")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive HP Unlearning Evaluation")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model to use (gpt2-small, gemma-2b, etc.)")
    parser.add_argument("--prompts_path", type=str, 
                        default="8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json")
    parser.add_argument("--layer", type=int, default=None, help="SAE layer for ablation")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--clamp_value", type=float, default=-20.0)
    parser.add_argument("--limit", type=int, default=50, help="Limit completion prompts")
    parser.add_argument("--ppl_samples", type=int, default=200, help="Samples for perplexity")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    
    args = parser.parse_args()
    main(args)
