import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
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
    print("Loading Model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # 2. Load SAE
    print(f"Loading SAE for Layer {args.layer}...")
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
    
    checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sae.to(device)
    sae.eval()
    
    # 3. Load Data
    print("Loading Target Corpus...")
    target_tokens = load_and_tokenize(args.target_corpus)
    # Truncate for speed
    target_tokens = target_tokens[:300000] # Use more tokens for better estimate
    
    print("Loading Neutral Corpus...")
    # Using WikiText-2
    neutral_dataset = get_neutral_corpus(split="train")
    # Tokenize neutral
    # Just grab first 300k tokens
    neutral_text = "\n".join(neutral_dataset["text"][:2000]) # approximate
    neutral_tokens = model.tokenizer.encode(neutral_text)[:300000]
    
    # 4. Get Activations and Feature Activations
    def get_feature_activations(tokens, batch_size=8):
        feature_acts_sum = torch.zeros(sae.d_sae, device=device)
        count = 0
        
        # Batching
        num_batches = len(tokens) // 128 // batch_size
        token_batches = [tokens[i:i+128] for i in range(0, len(tokens), 128)]
        # This simple batching might drop last few
        
        # Ensure we have clean batches
        clean_batches = []
        for i in range(0, len(token_batches), batch_size):
            batch = token_batches[i:i+batch_size]
            if len(batch) > 0:
                # Pad if needed or just drop
                # Simple: stack
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
                # Run model
                _, cache = model.run_with_cache(batch, stop_at_layer=args.layer + 1)
                acts = cache[f"blocks.{args.layer}.hook_resid_post"]
                features = sae.encode(acts) # [batch, seq, d_sae]
                
                # Sum over batch and seq
                # We want mean activation probability or mean activation magnitude?
                # "Difference-in-Means" usually means mean activation value.
                features = einops.rearrange(features, "b s d -> (b s) d")
                feature_acts_sum += features.sum(dim=0)
                count += features.shape[0]
                
        return feature_acts_sum / count

    print("Computing Target Feature Activations...")
    target_mean = get_feature_activations(target_tokens)
    
    print("Computing Neutral Feature Activations...")
    neutral_mean = get_feature_activations(neutral_tokens)
    
    # 5. Difference in Means
    diff = target_mean - neutral_mean
    
    # 6. Select Top Features
    top_vals, top_inds = torch.topk(diff, k=args.num_features)
    
    print("Top Features specific to Target:")
    for i in range(args.num_features):
        print(f"Feature {top_inds[i].item()}: Diff = {top_vals[i].item():.4f}")
        
    # Save results
    save_path = f"results/layer_{args.layer}_features.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"indices": top_inds, "values": top_vals}, save_path)
    print(f"Saved selected features to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=10)
    args = parser.parse_args()
    analyze(args)
