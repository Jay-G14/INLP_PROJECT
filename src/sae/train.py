import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.sae.trainer import SAETrainer
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse

# Helper function to prevent silent truncation over large datasets
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

def main(args):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
            device="cpu", # device is handled by device_map
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
    
    # 2. Load Data — SAE should be trained on GENERAL data, not just target corpus
    # This is critical: the SAE learns a general-purpose feature dictionary.
    # HP-specific features are identified LATER via difference-in-means.
    print("Loading Training Data...")
    
    ctx_len = 128
    all_tokens = []
    
    # Load neutral corpus (WikiText-2) — this is the primary training data
    print("Loading WikiText-2 (neutral corpus)...")
    neutral_dataset = get_neutral_corpus(split="train")
    if isinstance(neutral_dataset, list):
        neutral_text = "\n".join([t for t in neutral_dataset if t.strip()])
    else:
        neutral_text = "\n".join([t for t in neutral_dataset["text"] if t.strip()])
    
    # Use model.to_tokens logically over chunks to prevent silent context window truncation
    neutral_tokens = tokenize_in_chunks(model, neutral_text)
    
    # Cap maximum tokens parsed to speed up debugs vs real runs cleanly if max_tokens is passed
    if hasattr(args, 'max_tokens') and args.max_tokens:
        neutral_tokens = neutral_tokens[:args.max_tokens]
    
    print(f"  WikiText-2 tokens: {len(neutral_tokens):,}")
    all_tokens.extend(neutral_tokens)
    
    # Optionally also include target corpus to ensure SAE can represent those features too
    if args.include_target and args.target_corpus and os.path.exists(args.target_corpus):
        print(f"Loading target corpus: {args.target_corpus}...")
        with open(args.target_corpus, 'r', encoding='utf-8') as f:
            target_text = f.read()
            
        target_tokens = tokenize_in_chunks(model, target_text)
        
        if hasattr(args, 'max_tokens') and args.max_tokens:
            target_tokens = target_tokens[:args.max_tokens]
            
        print(f"  Target corpus tokens: {len(target_tokens):,}")
        all_tokens.extend(target_tokens)
    
    print(f"Total tokens: {len(all_tokens)}")
    
    # Chunk into context windows
    num_chunks = len(all_tokens) // ctx_len
    all_tokens = all_tokens[:num_chunks * ctx_len]
    data_tensor = torch.tensor(all_tokens).view(-1, ctx_len)
    
    print(f"Created {data_tensor.shape[0]} chunks of length {ctx_len}")
    
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize SAE
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    k = args.k
    
    print(f"Initializing TopK SAE for Layer {args.layer}:")
    print(f"  d_model={d_model}, d_sae={d_sae}, k={k}")
    print(f"  Expansion factor: {args.expansion_factor}x")
    
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=k).to(device)
    
    # 4. Train
    print(f"\nStarting training for {args.epochs} epoch(s)...")
    
    # --- Diagnostic: Activation Scale & Data Info ---
    print(f"\n--- Diagnostic ---")
    print(f"Actual token count being used: {len(all_tokens):,}")
    print(f"Number of batches per epoch: {len(data_loader)}")
    
    sample_batch = next(iter(data_loader))[0]
    device_to_use = getattr(model, "hf_device_map", {}).get("", device) 
    if isinstance(device_to_use, dict):
        device_to_use = next(iter(model.parameters())).device
        
    with torch.no_grad():
        _, cache = model.run_with_cache(
            sample_batch.to(device_to_use),
            names_filter=f"blocks.{args.layer}.hook_resid_post"
        )
        acts = cache[f"blocks.{args.layer}.hook_resid_post"]
        print(f"Activation shape: {acts.shape}")
        print(f"Activation mean: {acts.mean().item():.4f}")
        print(f"Activation std: {acts.std().item():.4f}")
        print(f"Activation variance (vital for rel MSE): {acts.var().item():.4f}")
    print(f"------------------\n")

    trainer = SAETrainer(
        sae, model, data_loader, 
        layer=args.layer, 
        lr=args.lr, 
        device=device,
        aux_loss_weight=args.aux_loss_weight,       # ← add
        dead_neuron_window=args.dead_neuron_window 
    )
    trainer.train(num_epochs=args.epochs)
    
    # 5. Save
    save_path = f"checkpoints/sae_layer_{args.layer}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sae.state_dict(), save_path)
    print(f"\nSaved SAE to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8, help="Layer to train SAE on")
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt", 
                        help="Path to target corpus (optional, mixed in if --include_target)")
    parser.add_argument("--include_target", action="store_true", default=True,
                        help="Include target corpus in training data alongside WikiText")
    parser.add_argument("--no_include_target", dest="include_target", action="store_false",
                        help="Train SAE on WikiText only")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name to load")
    parser.add_argument("--expansion_factor", type=int, default=8, help="Expansion factor for SAE")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--max_tokens", type=int, default=500000, help="Maximum number of tokens to parse for training phase")
    
    args = parser.parse_args()
    main(args)
