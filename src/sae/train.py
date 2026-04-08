import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformer_lens import HookedTransformer

from src.data.preprocess import (
    get_tokenizer,
    iter_target_documents,
    iter_tinystories_documents,
    iter_token_chunks,
    iter_wikitext_documents,
)
from src.models.llama_loader import load_llama
from src.sae.checkpoints import load_sae_checkpoint, save_training_checkpoint
from src.sae.model import TopKSAE
from src.sae.trainer import SAETrainer


class TokenChunkIterableDataset(IterableDataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def _iter_documents(self):
        print("Streaming WikiText-2 (neutral corpus)...")
        for text in iter_wikitext_documents(split="train"):
            yield text

        print("Streaming TinyStories...")
        for text in iter_tinystories_documents(limit=self.args.tiny_limit):
            yield text

        if (
            self.args.include_target
            and self.args.target_corpus
            and os.path.exists(self.args.target_corpus)
        ):
            print(f"Streaming target corpus: {self.args.target_corpus}...")
            for text in iter_target_documents(self.args.target_corpus):
                yield text

    def __iter__(self):
        for token_chunk in iter_token_chunks(
            self._iter_documents(),
            self.tokenizer,
            ctx_len=self.args.ctx_len,
            max_tokens=self.args.limit,
        ):
            yield torch.tensor(token_chunk, dtype=torch.long)


def build_data_loader(tokenizer, args):
    dataset = TokenChunkIterableDataset(tokenizer, args)
    return DataLoader(dataset, batch_size=args.batch_size)


def get_model_and_tokenizer(args, model_device):
    if args.model_family == "llama":
        model, tokenizer = load_llama(
            model_name=args.model_name,
            quantize=args.quantize,
            device_map=args.device_map,
            use_cache=False,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        d_model = model.config.hidden_size
        model_family = "llama"
    else:
        model = HookedTransformer.from_pretrained(args.model_name, device=model_device)
        tokenizer = model.tokenizer
        d_model = model.cfg.d_model
        model_family = "gpt2"
    return model, tokenizer, d_model, model_family


def main(args):
    has_cuda = torch.cuda.is_available()
    model_device = args.model_device if args.model_device != "auto" else ("cuda" if has_cuda else "cpu")
    sae_device = args.sae_device if args.sae_device != "auto" else ("cpu" if args.model_family == "llama" else model_device)

    print(f"Using model device: {model_device}")
    print(f"Using SAE device: {sae_device}")
    print(f"Model family: {args.model_family} | model: {args.model_name}")

    model, tokenizer, d_model, model_family = get_model_and_tokenizer(args, model_device)
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model_name)

    data_loader = build_data_loader(tokenizer, args)

    d_sae = d_model * args.expansion_factor
    print(f"Initializing TopK SAE for Layer {args.layer}:")
    print(f"  d_model={d_model}, d_sae={d_sae}, k={args.k}")
    print(f"  Expansion factor: {args.expansion_factor}x")

    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)

    prefix = "checkpoints/llama" if args.model_family == "llama" else "checkpoints"
    os.makedirs(prefix, exist_ok=True)
    checkpoint_path = args.resume_from or f"{prefix}/sae_layer_{args.layer}.pt"

    trainer = SAETrainer(
        sae,
        model,
        data_loader,
        layer=args.layer,
        lr=args.lr,
        model_device=model_device,
        sae_device=sae_device,
        model_family=model_family,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=checkpoint_path,
        checkpoint_args=vars(args),
    )

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0

    if args.resume_from:
        metadata = load_sae_checkpoint(
            args.resume_from,
            trainer.sae,
            optimizer=trainer.optimizer,
            map_location=sae_device,
        )
        start_epoch = metadata["epoch"]
        start_step_in_epoch = metadata["step_in_epoch"]
        global_step = metadata["global_step"]
        print(
            f"Resumed from {args.resume_from} at epoch={start_epoch + 1}, step_in_epoch={start_step_in_epoch}, global_step={global_step}"
        )

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    final_state = trainer.train(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        save_every_steps=args.save_every_steps,
        start_epoch=start_epoch,
        start_step_in_epoch=start_step_in_epoch,
        global_step=global_step,
    )

    save_training_checkpoint(
        checkpoint_path,
        trainer.sae,
        optimizer=trainer.optimizer,
        epoch=final_state["epoch"],
        step_in_epoch=final_state["step_in_epoch"],
        global_step=final_state["global_step"],
        args=vars(args),
    )
    print(f"\nSaved SAE checkpoint to {checkpoint_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gpt2"],
        default="llama",
        help="Model family to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HF model name.",
    )
    parser.add_argument("--device_map", type=str, default="auto", help="Device map.")
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        help="Device used for the backbone model activations.",
    )
    parser.add_argument(
        "--sae_device",
        type=str,
        default="cpu",
        help="Device used for SAE training (cpu or cuda).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization mode for Llama.",
    )
    parser.add_argument("--layer", type=int, default=15, help="Layer to train SAE on")
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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--expansion_factor", type=int, default=4, help="Expansion factor for SAE"
    )
    parser.add_argument("--k", type=int, default=8, help="TopK sparsity")
    parser.add_argument("--ctx_len", type=int, default=128, help="Context length")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional total token limit across streamed corpora.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum number of training steps.",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=500,
        help="How often to save a resumable checkpoint.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a saved training checkpoint.",
    )
    parser.add_argument(
        "--tiny_limit",
        type=int,
        default=2000,
        help="How many TinyStories examples to stream into the neutral corpus.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory savings.",
    )

    args = parser.parse_args()
    main(args)

