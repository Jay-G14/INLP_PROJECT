import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
import torch
torch.set_default_dtype(torch.float16)
import json
import argparse
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook
from tqdm import tqdm

def calculate_score(model, prompts, references, device):
    matches = 0
    total = len(prompts)
    
    print(f"Evaluating {total} prompts...")
    # Batching to speed up
    # But generation is tricky with batching if lengths differ.
    # We will do sequential for simplicity and safety, or batch if consistent.
    
    for i, prompt in enumerate(tqdm(prompts)):
        input_ids = model.to_tokens(prompt)
        
        # Generate
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=20, do_sample=False, verbose=False)
            
        # Extract completion
        # output is [1, seq_len + 20]
        generated_ids = output[0, input_ids.shape[1]:]
        completion = model.to_string(generated_ids)
        
        # Check references
        refs = references[i]
        found = False
        for ref in refs:
            if ref.lower() in completion.lower():
                found = True
                break
        
        if found:
            matches += 1
            
    return matches / total

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Prompts
    prompts_path = "8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json"
    if not os.path.exists(prompts_path):
         # Try locating it
         prompts_path = "/home/jay/inlp/8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json"
         
    with open(prompts_path, "r") as f:
        data = json.load(f)
        
    # Data is a list of dicts or list of list?
    # View file showed: [ { "prompt": { ... } }, ... ]
    prompts = [item["prompt"]["prompt"] for item in data]
    references = [item["prompt"]["references"] for item in data]
    
    # Subset for speed if needed
    if args.limit:
        prompts = prompts[:args.limit]
        references = references[:args.limit]
    
    # 2. Load Model
    print(f"Loading Model: {args.model} on CPU first to save memory...")
    model = HookedTransformer.from_pretrained(
        args.model,
        device="cpu",
        torch_dtype=torch.float16,
    )

    # Move model to the specified device
    if device == "cuda":
        print("Moving model to CUDA...")
        torch.cuda.empty_cache()
        model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Baseline Eval
    if not args.skip_baseline:
        print("Evaluating Baseline...")
        baseline_score = calculate_score(model, prompts, references, device)
        print(f"Baseline Match Rate: {baseline_score:.4f}")
    
    # 3. Load SAE and Features
    print(f"Loading SAE for Layer {args.layer}...")
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
    
    checkpoint_path = f"checkpoints/sae_{args.model.replace('/', '_')}_layer_{args.layer}.pt"
    # Fallback to older format if newer doesn't exist
    if not os.path.exists(checkpoint_path):
        old_checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
        if os.path.exists(old_checkpoint_path):
            checkpoint_path = old_checkpoint_path
        else:
            print(f"Error: Checkpoint {checkpoint_path} not found. Train SAE first.")
            return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sae.load_state_dict(checkpoint["state_dict"])
    else:
        sae.load_state_dict(checkpoint)
        
    sae.to(device)
    # sae.eval() # model doesn't have eval/train specific behavior probably
    
    features_path = f"results/layer_{args.layer}_features.pt"
    feature_indices = []
    if os.path.exists(features_path):
        features_data = torch.load(features_path, map_location=device)
        feature_indices = features_data["indices"]
    else:
        print(f"Warning: Features file {features_path} not found. No ablation unless manually specified.")
        
    if len(feature_indices) > 0:
        print(f"Ablating {len(feature_indices)} features: {feature_indices}")
        
        # 4. Apply Hook
        hook_fn = get_ablation_hook(sae, feature_indices, clamp_value=args.clamp_value)
        
        # 5. Evaluate with Ablation
        print("Evaluating Ablated Model...")
        # We use a context manager to apply the hook temporarily
        with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
            ablated_score = calculate_score(model, prompts, references, device)
            
        print(f"Ablated Match Rate: {ablated_score:.4f}")
        
        if not args.skip_baseline:
            print(f"Reduction: {baseline_score - ablated_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small", help="Model to use")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--limit", type=int, default=50, help="Limit number of prompts for speed")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--clamp_value", type=float, default=-20.0, help="Negative clamping value for feature intervention")
    args = parser.parse_args()
    evaluate(args)
