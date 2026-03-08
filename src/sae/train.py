import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer

from src.data.preprocess import get_neutral_corpus, load_and_tokenize
from src.sae.model import TopKSAE
from src.sae.trainer import SAETrainer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    # SAE is trained on GENERAL data, not just target corpus
    # the SAE learns a general-purpose feature dictionary.
    # HP-specific features are identified LATER via difference-in-means.
    print("Loading Training Data...")

    ctx_len = 128
    all_tokens = []

    # Load neutral corpus (WikiText-2) — this is the primary training data
    print("Loading WikiText-2 (neutral corpus)...")
    neutral_dataset = get_neutral_corpus(split="train")
    tokenizer = model.tokenizer
    neutral_text = "\n".join([t for t in neutral_dataset["text"] if t.strip()])
    neutral_tokens = tokenizer.encode(neutral_text)
    print(f"  WikiText-2 tokens: {len(neutral_tokens)}")
    all_tokens.extend(neutral_tokens)

    #  include target corpus to ensure SAE can represent those features too
    if (
        args.include_target
        and args.target_corpus
        and os.path.exists(args.target_corpus)
    ):
        print(f"Loading target corpus: {args.target_corpus}...")
        target_tokens = load_and_tokenize(args.target_corpus)
        print(f"  Target corpus tokens: {len(target_tokens)}")
        all_tokens.extend(target_tokens)

    print(f"Total tokens: {len(all_tokens)}")

    # Chunk into context windows
    num_chunks = len(all_tokens) // ctx_len
    all_tokens = all_tokens[: num_chunks * ctx_len]
    data_tensor = torch.tensor(all_tokens).view(-1, ctx_len)

    print(f"Created {data_tensor.shape[0]} chunks of length {ctx_len}")

    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize SAE
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    k = args.k

    print(f"Initializing TopK SAE for Layer {args.layer}:")
    print(f"  d_model={d_model}, d_sae={d_sae}, k={k}")
    print(f"  Expansion factor: {args.expansion_factor}x")

    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=k)

    # train
    print(f"\nStarting training for {args.epochs} epoch(s)...")
    trainer = SAETrainer(
        sae, model, data_loader, layer=args.layer, lr=args.lr, device=device
    )
    trainer.train(num_epochs=args.epochs)
    save_path = f"checkpoints/sae_layer_{args.layer}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sae.state_dict(), save_path)
    print(f"\nSaved SAE to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8, help="Layer to train SAE on")
    parser.add_argument(
        "--target_corpus",
        type=str,
        default="src/data/target_corpus.txt",
        help="Path to target corpus (optional, mixed in if --include_target)",
    )
    parser.add_argument(
        "--include_target",
        action="store_true",
        default=True,
        help="Include target corpus in training data alongside WikiText",
    )
    parser.add_argument(
        "--no_include_target",
        dest="include_target",
        action="store_false",
        help="Train SAE on WikiText only",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--model_name", type=str, default="gpt2-medium", help="Model name to load"
    )
    parser.add_argument(
        "--expansion_factor", type=int, default=16, help="Expansion factor for SAE"
    )
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")

    args = parser.parse_args()
    main(args)
