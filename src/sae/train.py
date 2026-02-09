import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
import os
from src.sae.model import TopKSAE
from src.sae.trainer import SAETrainer
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # 2. Load Data
    print("Loading Data...")
    if args.target_corpus:
        # Load processed target corpus
        tokens = load_and_tokenize(args.target_corpus)
        # Create dataset
        # Chunk tokens into context length (e.g. 128 or 1024)
        # For SAE training, usually we just need a stream of activations.
        # But data loader needs batches.
        # Let's chunk into 128 for speed/memory.
        ctx_len = 128
        num_chunks = len(tokens) // ctx_len
        tokens = tokens[:num_chunks * ctx_len]
        data_tensor = torch.tensor(tokens).view(-1, ctx_len)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        # Load neutral corpus (wikitext)
        # Use simple implementation for now
        dataset = get_neutral_corpus(split="train")
        # Tokenize
        tokenizer = model.tokenizer
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128, return_tensors="pt", padding="max_length")
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_datasets.set_format(type="torch", columns=["input_ids"])
        data_loader = DataLoader(tokenized_datasets, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize SAE
    # SAE Config
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
    save_path = f"checkpoints/sae_layer_{args.layer}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sae.state_dict(), save_path)
    print(f"Saved SAE to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8, help="Layer to train SAE on")
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt", help="Path to target corpus")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--expansion_factor", type=int, default=16, help="Expansion factor for SAE")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    
    args = parser.parse_args()
    main(args)
