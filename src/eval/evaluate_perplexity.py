import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
import torch
torch.set_default_dtype(torch.float16)
import argparse
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook
from datasets import load_dataset
from tqdm import tqdm
import math

def calculate_perplexity(model, dataset, device, max_samples=None, max_length=512):
    """Calculate perplexity on WikiText-2."""
    total_loss = 0
    total_tokens = 0
    
    samples = dataset["text"] if max_samples is None else dataset["text"][:max_samples]
    
    for text in tqdm(samples, desc="Computing Perplexity"):
        if not text.strip():  # Skip empty lines
            continue
            
        # Tokenize
        tokens = model.to_tokens(text, prepend_bos=True)
        
        # Skip if too long or too short
        if tokens.shape[1] > max_length or tokens.shape[1] < 2:
            continue
        
        with torch.no_grad():
            logits = model(tokens)  # [1, seq_len, vocab_size]
            
            # Calculate loss (cross-entropy)
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                shift_logits, 
                shift_labels, 
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    # Perplexity = exp(average negative log likelihood)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading {args.model} on CPU first to save memory...")
    model = HookedTransformer.from_pretrained(
        args.model, 
        device="cpu",
        dtype=torch.float16,
    )

    # Move model to the specified device
    if device == "cuda":
        print("Moving model to CUDA...")
        torch.cuda.empty_cache()
        model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load WikiText-2 dataset
    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Baseline evaluation
    if not args.skip_baseline:
        print("Evaluating Baseline Perplexity...")
        baseline_ppl = calculate_perplexity(model, dataset, device, max_samples=args.limit)
        print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    
    # Load SAE and features for ablation
    if args.layer is not None:
        print(f"Loading SAE for Layer {args.layer}...")
        d_model = model.cfg.d_model
        d_sae = d_model * args.expansion_factor
        sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
        
        checkpoint_path = f"checkpoints/sae_{args.model.replace('/', '_')}_layer_{args.layer}.pt"
        if not os.path.exists(checkpoint_path):
            old_checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
            if os.path.exists(old_checkpoint_path):
                checkpoint_path = old_checkpoint_path
            else:
                print(f"Error: Checkpoint {checkpoint_path} not found.")
                return
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sae.load_state_dict(checkpoint["state_dict"])
        else:
            sae.load_state_dict(checkpoint)
        sae.to(device)
        
        features_path = f"results/{args.model.replace('/', '_')}_layer_{args.layer}_features.pt"
        if os.path.exists(features_path):
            features_data = torch.load(features_path, map_location=device)
            feature_indices = features_data["indices"]
            print(f"Ablating {len(feature_indices)} features")
            
            # Apply hook and evaluate
            hook_fn = get_ablation_hook(sae, feature_indices, clamp_value=args.clamp_value)
            
            print("Evaluating Ablated Model Perplexity...")
            with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
                ablated_ppl = calculate_perplexity(model, dataset, device, max_samples=args.limit)
                
            print(f"Ablated Perplexity: {ablated_ppl:.2f}")
            
            if not args.skip_baseline:
                print(f"Difference: {ablated_ppl - baseline_ppl:.2f}")
                print(f"Relative Change: {((ablated_ppl - baseline_ppl) / baseline_ppl * 100):.2f}%")
        else:
            print(f"Warning: Features file {features_path} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small", help="Model to use")
    parser.add_argument("--layer", type=int, default=10, help="Layer to ablate")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--clamp_value", type=float, default=-20.0, help="Negative clamping value for feature intervention")
    args = parser.parse_args()
    main(args)
