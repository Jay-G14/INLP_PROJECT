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

def evaluate_mmlu(model, dataset, device, max_samples=None):
    """Evaluate model on MMLU multiple choice questions."""
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    for item in tqdm(samples, desc="Evaluating MMLU"):
        question = item['question']
        choices = item['choices']
        answer_idx = item['answer']
        
        # Format as multiple choice
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # Get logits for A, B, C, D tokens
        with torch.no_grad():
            tokens = model.to_tokens(prompt)
            logits = model(tokens)  # [1, seq_len, vocab_size]
            
            # Get logits for last position
            last_logits = logits[0, -1, :]
            
            # Get probabilities for A, B, C, D
            choice_tokens = [model.to_single_token(" " + chr(65+i)) for i in range(len(choices))]
            choice_logits = last_logits[choice_tokens]
            pred_idx = torch.argmax(choice_logits).item()
            
            if pred_idx == answer_idx:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print("Loading Model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Load MMLU dataset (using a subset for faster evaluation)
    print("Loading MMLU dataset...")
    # Using "abstract_algebra" as a representative task
    # You can change this to other subjects or load multiple
    dataset = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    
    # Baseline evaluation
    if not args.skip_baseline:
        print("Evaluating Baseline on MMLU...")
        baseline_acc = evaluate_mmlu(model, dataset, device, max_samples=args.limit)
        print(f"Baseline MMLU Accuracy: {baseline_acc:.4f}")
    
    # Load SAE and features for ablation
    if args.layer is not None:
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
        
        features_path = f"results/layer_{args.layer}_features.pt"
        if os.path.exists(features_path):
            features_data = torch.load(features_path, map_location=device)
            feature_indices = features_data["indices"]
            print(f"Ablating {len(feature_indices)} features")
            
            # Apply hook and evaluate
            hook_fn = get_ablation_hook(sae, feature_indices, clamp_value=args.clamp_value)
            
            print("Evaluating Ablated Model on MMLU...")
            with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
                ablated_acc = evaluate_mmlu(model, dataset, device, max_samples=args.limit)
                
            print(f"Ablated MMLU Accuracy: {ablated_acc:.4f}")
            
            if not args.skip_baseline:
                print(f"Difference: {ablated_acc - baseline_acc:.4f}")
        else:
            print(f"Warning: Features file {features_path} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=10, help="Layer to ablate")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--clamp_value", type=float, default=-20.0, help="Negative clamping value for feature intervention")
    args = parser.parse_args()
    main(args)
