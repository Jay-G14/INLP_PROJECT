import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse
import einops
from tqdm import tqdm

def tokenize_in_chunks(model, text, chunk_size=10000):
    """Tokenize large text without hitting sequence length limits."""
    words = text.split()
    all_tokens = []
    print(f"  Tokenizing {len(words)} words in chunks of {chunk_size}...")
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        tokens = model.to_tokens(chunk).squeeze(0).tolist()
        if i == 0:
            all_tokens.extend(tokens)
        else:
            all_tokens.extend(tokens[1:])  # skip BOS except first chunk
    return all_tokens

def analyze(args):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print(f"Loading {args.model_name}...")
    if args.model_name.startswith("meta-llama"):
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = HookedTransformer.from_pretrained(
            args.model_name,
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )
    else:
        model = HookedTransformer.from_pretrained(
            args.model_name, 
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )
    
    # 2. Load SAE
    print(f"Loading SAE for Layer {args.layer}...")
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k).to(device)
    
    checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    sae.to(device)
    sae.eval()
    
    # 3. Load Data
    print("Loading Target Corpus (Harry Potter)...")
    with open(args.target_corpus, 'r', encoding='utf-8') as f:
        target_text = f.read()
    
    target_tokens = tokenize_in_chunks(model, target_text)
    target_tokens = target_tokens[:args.max_tokens]
    print(f"  Target tokens after chunking: {len(target_tokens):,}")
    
    print("Loading Neutral Corpus (Wiki + Fiction)...")
    neutral_list = get_neutral_corpus(split="train")
    if isinstance(neutral_list, list):
        neutral_text = "\n".join([t for t in neutral_list if t.strip()])
    else:
        neutral_text = "\n".join([t for t in neutral_list["text"] if t.strip()])
        
    neutral_tokens = tokenize_in_chunks(model, neutral_text)
    neutral_tokens = neutral_tokens[:args.max_tokens]
    print(f"  Neutral tokens after chunking: {len(neutral_tokens):,}")

    # Verification block
    print(f"\n--- Corpus Size Check ---")
    print(f"Target tokens : {len(target_tokens):,}")
    print(f"Neutral tokens: {len(neutral_tokens):,}")

    target_batches = len(target_tokens) // (8 * 128)
    neutral_batches = len(neutral_tokens) // (8 * 128)
    print(f"Target batches : {target_batches}")
    print(f"Neutral batches: {neutral_batches}")

    if neutral_batches < target_batches * 0.5:
        print("⚠️  WARNING: Neutral corpus is less than 50% the size of target!")
        print("    Results will be unreliable. Increase neutral corpus size.")
    print("-------------------------\n")

    # Match neutral corpus size to target for fair comparison
    min_tokens = min(len(target_tokens), len(neutral_tokens))
    target_tokens = target_tokens[:min_tokens]
    neutral_tokens = neutral_tokens[:min_tokens]
    print(f"Using {min_tokens:,} tokens for each corpus (balanced)")
    print(f"Target tokens  : {len(target_tokens):,}")
    print(f"Neutral tokens : {len(neutral_tokens):,}")
    print(f"Target batches : {len(target_tokens) // (8 * 128)}")
    print(f"Neutral batches: {len(neutral_tokens) // (8 * 128)}")
    
    # 4. Compute Feature Activation Statistics
    def get_feature_stats(tokens, batch_size=8, ctx_len=128):
        """
        Compute per-feature statistics using the FULL forward pass (not just encode).
        Returns:
          - activation_frequency: fraction of tokens where feature > 0
          - mean_activation: mean activation value (over tokens where feature > 0)
        """
        feature_fire_count = torch.zeros(sae.d_sae, device=device)
        feature_act_sum = torch.zeros(sae.d_sae, device=device)
        total_tokens_seen = 0
        
        # Flatten and filter out PAD tokens (if needed) but keep everything sequential
        if isinstance(tokens, list):
            tokens = tokens  # Usually 1D list
        elif tokens.ndim > 1:
            tokens = tokens.flatten().tolist()
            
        print(f"  Got {len(tokens)} total tokens for stats...")
        
        # Create batches
        token_chunks = [tokens[i:i+ctx_len] for i in range(0, len(tokens) - ctx_len + 1, ctx_len)]
        
        clean_batches = []
        for i in range(0, len(token_chunks), batch_size):
            batch = token_chunks[i:i+batch_size]
            if len(batch) > 0:
                try:
                    tensor_batch = torch.tensor(batch)
                    if tensor_batch.ndim == 1:
                        tensor_batch = tensor_batch.unsqueeze(0)
                    if tensor_batch.shape[-1] == ctx_len:
                        clean_batches.append(tensor_batch)
                except:
                    pass
        
        print(f"  Constructed {len(clean_batches)} clean batches of shape {[b.shape for b in clean_batches[:1]]}...")
        with torch.no_grad():
            for batch in tqdm(clean_batches, desc="  Features"):
                # Move batch tokens to device 
                device_to_use = next(model.parameters()).device
                batch = batch.to(device_to_use)
            
                # Run model to get activations
                act_name = f"blocks.{args.layer}.hook_resid_post"
                _, cache = model.run_with_cache(
                    batch, 
                    names_filter=lambda name: name == act_name,
                    stop_at_layer=args.layer + 1
                )
                acts = cache[act_name]
                
                # Move acts back to SAE processing device
                acts = acts.to(device)
                
                # Use FULL forward pass (not just encode) to get post-TopK sparse activations
                _, z_sparse = sae(acts)
                
                # Clear unused values manually
                del cache, acts
                
                # z_sparse shape: [batch, seq, d_sae]
                
                z_flat = einops.rearrange(z_sparse, "b s d -> (b s) d")
                
                # Activation frequency: how often does each feature fire (> 0)?
                fired = (z_flat > 0).float()
                feature_fire_count += fired.sum(dim=0)
                
                # Mean activation (sum of activations for computing mean later)
                feature_act_sum += z_flat.sum(dim=0)
                
                total_tokens_seen += z_flat.shape[0]
                
                del z_sparse, z_flat, fired
                torch.cuda.empty_cache()
                
        activation_freq = feature_fire_count / max(1, total_tokens_seen)
        mean_activation = feature_act_sum / (feature_fire_count + 1e-8)  # mean over firing tokens only
        
        return activation_freq, mean_activation, total_tokens_seen

    print("\nComputing Target Feature Statistics...")
    target_freq, target_mean_act, target_count = get_feature_stats(target_tokens)
    
    print("\nComputing Neutral Feature Statistics...")
    neutral_freq, neutral_mean_act, neutral_count = get_feature_stats(neutral_tokens)
    
    # 5. Feature Metrics calculation
    # Compute Specificity Ratio: how much more likely is it to fire on target?
    # Use a small epsilon to avoid div by zero
    specificity_ratio = target_freq / (neutral_freq + 1e-6)
    
    # Filter: avoid syntactic/common features that fire too often in general
    # Generic features break the model's grammar if ablated.
    # Stricter neutral_freq filter for high expansion SAEs
    max_neutral_freq = args.max_neutral_freq
    max_freq = args.max_freq
    
    # valid_mask requires minimum frequency on target and maximum on neutral
    # PLUS a minimum specificity ratio
    valid_mask = (target_freq > args.min_freq) & \
                 (neutral_freq < max_neutral_freq) & \
                 (target_freq < max_freq) & \
                 (specificity_ratio > args.min_ratio)
    
    # Metric: freq_diff is good, but let's also weight by specificity
    # We want features that are BOTH high-diff AND high-ratio
    freq_diff = target_freq - neutral_freq
    score = freq_diff * torch.log1p(specificity_ratio)
    
    # Sort selection
    if args.sort_by == "ratio":
        selection_metric = specificity_ratio.clone()
    elif args.sort_by == "score":
        selection_metric = score.clone()
    else:
        selection_metric = freq_diff.clone()
    
    selection_metric[~valid_mask] = -float('inf')
    
    # 6. Select Top Features
    top_vals, top_inds = torch.topk(selection_metric, k=args.num_features)
    
    print(f"\n{'='*70}")
    print(f"Top {args.num_features} Harry Potter-specific features (by score):")
    print(f"{'Index':>10} | {'Score':>10} | {'Target Freq':>12} | {'Neut Freq':>12} | {'Ratio':>10}")
    print("-" * 70)
    for i in range(args.num_features):
        idx = top_inds[i].item()
        val = top_vals[i].item()
        t_f = target_freq[idx].item()
        n_f = neutral_freq[idx].item()
        ratio = specificity_ratio[idx].item()
        print(f"{idx:10} | {val:10.4f} | {t_f:12.4f} | {n_f:12.4f} | {ratio:10.1f}")
        
    # 7. Save results (indices, frequency data, and mean activations for ablation)
    save_path = f"results/layer_{args.layer}_features.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "indices": top_inds,
        "freq_diff": top_vals,
        "target_freq": target_freq,
        "neutral_freq": neutral_freq,
        "target_mean_activation": target_mean_act,
        "neutral_mean_activation": neutral_mean_act,
    }, save_path)
    print(f"\nSaved selected features to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name to load")
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=300000, help="Max tokens from each corpus")
    parser.add_argument("--min_freq", type=float, default=0.001, help="Min activation frequency on target")
    parser.add_argument("--max_neutral_freq", type=float, default=0.005, help="Max activation frequency on neutral corpus")
    parser.add_argument("--max_freq", type=float, default=0.2, help="Max activation frequency overall")
    parser.add_argument("--min_ratio", type=float, default=20.0, help="Minimum specificity ratio (target/neutral)")
    parser.add_argument("--sort_by", type=str, choices=["ratio", "score", "diff"], default="score")
    args = parser.parse_args()
    analyze(args)
