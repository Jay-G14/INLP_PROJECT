import os

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
torch.set_default_dtype(torch.float16)
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
import os
from src.sae.model import TopKSAE
from src.sae.trainer import SAETrainer
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse

# Supported models
SUPPORTED_MODELS = {
    "gpt2-small": {"d_model": 768, "n_layers": 12},
    "gpt2-medium": {"d_model": 1024, "n_layers": 24},
    "gemma-2b": {"d_model": 2048, "n_layers": 18},
    "gemma-2-2b": {"d_model": 2304, "n_layers": 26},
}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading {args.model} on CPU first to save memory...")
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
    
    # 2. Load Data - CRITICAL: Train on NEUTRAL corpus to learn general features!
    print("Loading Data...")
    
    if args.use_neutral_corpus:
        # This is the correct approach from the inspiration paper
        # Train SAE on general text to learn general features
        print("Using NEUTRAL corpus (OpenWebText) for SAE training...")
        print("This learns general features that can later be compared to HP corpus.")
        
        from datasets import load_dataset
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        # Collect tokens
        tokenizer = model.tokenizer
        all_tokens = []
        target_tokens = args.num_tokens
        
        print(f"Collecting {target_tokens} tokens from OpenWebText...")
        for example in dataset:
            tokens = tokenizer.encode(example["text"])
            all_tokens.extend(tokens)
            if len(all_tokens) >= target_tokens:
                break
        
        all_tokens = all_tokens[:target_tokens]
        print(f"Collected {len(all_tokens)} tokens")
        
        # Chunk into context length
        ctx_len = 128
        num_chunks = len(all_tokens) // ctx_len
        all_tokens = all_tokens[:num_chunks * ctx_len]
        data_tensor = torch.tensor(all_tokens).view(-1, ctx_len)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        # Original approach (not recommended for unlearning task)
        print("WARNING: Training on target corpus. This won't work for unlearning!")
        print("Use --use_neutral_corpus for proper feature learning.")
        tokens = load_and_tokenize(args.target_corpus, model_name=args.model)
        ctx_len = 128
        num_chunks = len(tokens) // ctx_len
        tokens = tokens[:num_chunks * ctx_len]
        data_tensor = torch.tensor(tokens).view(-1, ctx_len)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize SAE
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    k = args.k
    
    print(f"Initializing TopK SAE for Layer {args.layer} with d_sae={d_sae}, k={k}...")
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=k)
    
    # 4. Train
    print("Starting training...")
    trainer = SAETrainer(sae, model, data_loader, layer=args.layer, lr=args.lr, device=device)
    trainer.train(num_epochs=args.epochs)
    
    # 5. Save
    save_path = f"checkpoints/sae_{args.model.replace('/', '_')}_layer_{args.layer}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "model": args.model,
        "layer": args.layer,
        "d_sae": d_sae,
        "k": k,
    }, save_path)
    print(f"Saved SAE to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small", 
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Model to use")
    parser.add_argument("--layer", type=int, default=8, help="Layer to train SAE on")
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    parser.add_argument("--use_neutral_corpus", action="store_true", default=True,
                        help="Train on neutral corpus (OpenWebText) - RECOMMENDED")
    parser.add_argument("--no_neutral_corpus", action="store_false", dest="use_neutral_corpus",
                        help="Train on target corpus instead (not recommended)")
    parser.add_argument("--num_tokens", type=int, default=5_000_000,
                        help="Number of tokens to train on from neutral corpus")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    
    args = parser.parse_args()
    main(args)

