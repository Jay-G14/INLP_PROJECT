import os
from typing import Iterable, Iterator, List, Optional

import datasets
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


def clean_text(input_path, output_path):
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved processed text to {output_path}. Length: {len(full_text)} chars.")
    return full_text


def get_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def count_tokens(text_list, tokenizer):
    total_tokens = 0
    for text in text_list:
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    return total_tokens


def print_table(rows, headers):
    col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def format_row(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, col_widths))

    print("\n" + format_row(headers))
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(format_row(row))
    print()


def get_neutral_corpus(split="train", model_name="gpt2", tiny_limit=2000):
    tokenizer = get_tokenizer(model_name)

    print("Loading WikiText-2...")
    wiki_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    wiki_text = [text for text in wiki_dataset["text"] if text.strip()]

    print("Loading TinyStories...")
    fiction_text = []
    try:
        fiction_ds = datasets.load_dataset(
            "roneneldan/TinyStories", split="train", streaming=False
        )
        it = iter(fiction_ds)
        for _ in range(tiny_limit):
            fiction_text.append(next(it)["text"])
    except Exception as exc:
        print(f"Warning: TinyStories failed ({exc})")

    print("Counting tokens...")
    wiki_tokens = count_tokens(wiki_text, tokenizer)
    fiction_tokens = count_tokens(fiction_text, tokenizer)
    total = wiki_tokens + fiction_tokens

    rows = [
        ["WikiText-2", len(wiki_text), wiki_tokens, f"{(wiki_tokens / total) if total else 0:.3f}"],
        ["TinyStories", len(fiction_text), fiction_tokens, f"{(fiction_tokens / total) if total else 0:.3f}"],
        ["Total", len(wiki_text) + len(fiction_text), total, "1.000"],
    ]
    print_table(rows, headers=["Dataset", "Documents", "Tokens", "Sampling Ratio"])
    return wiki_text + fiction_text


def load_and_tokenize(file_path, model_name="gpt2"):
    print(f"Loading and tokenizing {file_path}...")
    tokenizer = get_tokenizer(model_name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text, add_special_tokens=False)
    print_table([["Target Corpus", len(tokens)]], headers=["Corpus", "Tokens"])
    return tokens


def iter_wikitext_documents(split="train") -> Iterator[str]:
    dataset = datasets.load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=split, streaming=True
    )
    for row in dataset:
        text = row["text"]
        if text and text.strip():
            yield text


def iter_tinystories_documents(limit=2000) -> Iterator[str]:
    try:
        dataset = datasets.load_dataset(
            "roneneldan/TinyStories", split="train", streaming=True
        )
        for idx, row in enumerate(dataset):
            if idx >= limit:
                break
            text = row["text"]
            if text and text.strip():
                yield text
    except Exception as exc:
        print(f"Warning: TinyStories streaming failed ({exc})")


def iter_target_documents(file_path: str) -> Iterator[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def iter_token_chunks(
    text_iter: Iterable[str],
    tokenizer,
    ctx_len: int,
    max_tokens: Optional[int] = None,
) -> Iterator[List[int]]:
    token_buffer: List[int] = []
    total_tokens = 0

    for text in text_iter:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue

        if max_tokens is not None:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            token_ids = token_ids[:remaining]

        total_tokens += len(token_ids)
        token_buffer.extend(token_ids)

        while len(token_buffer) >= ctx_len:
            yield token_buffer[:ctx_len]
            token_buffer = token_buffer[ctx_len:]

        if max_tokens is not None and total_tokens >= max_tokens:
            break


if __name__ == "__main__":
    input_file = "Harry_Potter_all_books_preprocessed.txt"
    output_file = "src/data/target_corpus.txt"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_text(input_file, output_file)
    load_and_tokenize(output_file)
    get_neutral_corpus()

