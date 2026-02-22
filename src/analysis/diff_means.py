import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
torch.set_default_dtype(torch.float16)
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse
import einops
import os
from tqdm import tqdm

def analyze(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print(f"Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    
    # 2. Load SAE
    print(f"Loading SAE for Layer {args.layer}...")
    
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
            # Old format - just state dict
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
    sae.eval()
    
    # 3. Load Data
    print("Loading Target Corpus (Harry Potter)...")
    target_tokens = load_and_tokenize(args.target_corpus)
    target_tokens = target_tokens[:300000]
    
    print("Loading Neutral Corpus (WikiText)...")
    neutral_dataset = get_neutral_corpus(split="train")
    neutral_text = "\n".join(neutral_dataset["text"][:2000])
    neutral_tokens = model.tokenizer.encode(neutral_text)[:300000]
    
    # 4. Get Feature Statistics
    def get_feature_stats(tokens, batch_size=8):
        """Compute mean activation and sparsity for each feature."""
        feature_acts_sum = torch.zeros(sae.d_sae, device=device)
        feature_active_count = torch.zeros(sae.d_sae, device=device)
        total_tokens = 0
        
        token_batches = [tokens[i:i+128] for i in range(0, len(tokens), 128)]
        
        clean_batches = []
        for i in range(0, len(token_batches), batch_size):
            batch = token_batches[i:i+batch_size]
            if len(batch) > 0:
                try:
                    tensor_batch = torch.tensor(batch).to(device)
                    if tensor_batch.ndim == 1:
                        tensor_batch = tensor_batch.unsqueeze(0)
                    clean_batches.append(tensor_batch)
                except:
                    pass
        
        print(f"Processing {len(clean_batches)} batches...")
        with torch.no_grad():
            for batch in tqdm(clean_batches):
                _, cache = model.run_with_cache(batch, stop_at_layer=args.layer + 1)
                acts = cache[f"blocks.{args.layer}.hook_resid_post"]
                features = sae.encode(acts)
                
                features = einops.rearrange(features, "b s d -> (b s) d")
                feature_acts_sum += features.sum(dim=0)
                feature_active_count += (features > 0).float().sum(dim=0)
                total_tokens += features.shape[0]
                
        mean_activation = feature_acts_sum / total_tokens
        sparsity = feature_active_count / total_tokens
        
        return mean_activation, sparsity
    
    print("Computing Target Feature Statistics...")
    target_mean, target_sparsity = get_feature_stats(target_tokens)
    
    print("Computing Neutral Feature Statistics...")
    neutral_mean, neutral_sparsity = get_feature_stats(neutral_tokens)
    
    if args.method == "sparsity":
        print(f"\nUsing SPARSITY-based feature selection (retain_threshold={args.retain_threshold})")
        
        # Features that rarely fire on neutral corpus but fire on target
        valid_mask = neutral_sparsity < args.retain_threshold
        
        # Among valid features, sort by target sparsity (descending)
        target_sparsity_masked = torch.where(valid_mask, target_sparsity, torch.tensor(-1.0, device=device))
        top_vals, top_inds = torch.topk(target_sparsity_masked, k=args.num_features)
        
        print("\nTop Features (sparsity-based):")
        print(f"{'Feature':<10} {'Target Sparsity':<18} {'Neutral Sparsity':<18}")
        print("-" * 46)
        for i in range(args.num_features):
            idx = top_inds[i].item()
            print(f"{idx:<10} {target_sparsity[idx].item():.6f}         {neutral_sparsity[idx].item():.6f}")
    else:
        print("\nUsing DIFFERENCE-IN-MEANS feature selection")
        diff = target_mean - neutral_mean
        top_vals, top_inds = torch.topk(diff, k=args.num_features)
        
        print("\nTop Features (difference-in-means):")
        for i in range(args.num_features):
            print(f"Feature {top_inds[i].item()}: Diff = {top_vals[i].item():.4f}")
        
    # Save results
    save_path = f"results/{args.model.replace('/', '_')}_layer_{args.layer}_features.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "indices": top_inds, 
        "values": top_vals,
        "method": args.method,
        "model": args.model,
        "layer": args.layer,
        "target_sparsity": target_sparsity[top_inds] if args.method == "sparsity" else None,
        "neutral_sparsity": neutral_sparsity[top_inds] if args.method == "sparsity" else None,
    }, save_path)
    print(f"\nSaved selected features to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model to use (gpt2-small, gemma-2b, etc.)")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--method", type=str, default="sparsity", choices=["sparsity", "diff_means"])
    parser.add_argument("--retain_threshold", type=float, default=0.01)
    args = parser.parse_args()
    analyze(args)


