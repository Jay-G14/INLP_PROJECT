import os
import re

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
import datasets
from transformers import AutoTokenizer, GPT2Tokenizer

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def clean_text(input_path, output_path):
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    target_text = full_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(target_text)

    print(f"Saved processed text to {output_path}. Length: {len(target_text)} chars.")
    return target_text


def count_tokens(text_list, tokenizer):
    total_tokens = 0
    for t in text_list:
        total_tokens += len(tokenizer.encode(t))
    return total_tokens


def print_table(rows, headers):
    col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def format_row(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, col_widths))

    print("\n" + format_row(headers))
    print("-+-".join("-" * w for w in col_widths))

    for r in rows:
        print(format_row(r))
    print()


def get_neutral_corpus(split="train", model_name="gpt2"):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print("Loading WikiText-2...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    wiki_text = [t for t in dataset["text"] if t.strip()]

    print("Loading TinyStories...")
    fiction_text = []
    try:
        fiction_ds = datasets.load_dataset(
            "roneneldan/TinyStories", split="train", streaming=False
        )

        it = iter(fiction_ds)
        for _ in range(2000):
            fiction_text.append(next(it)["text"])

    except Exception as e:
        print(f"Warning: TinyStories failed ({e})")

    print("Counting tokens...")

    wiki_tokens = count_tokens(wiki_text, tokenizer)
    fiction_tokens = count_tokens(fiction_text, tokenizer)

    total = wiki_tokens + fiction_tokens

    ratio_wiki = wiki_tokens / total if total > 0 else 0
    ratio_fiction = fiction_tokens / total if total > 0 else 0

    rows = [
        ["WikiText-2", len(wiki_text), wiki_tokens, f"{ratio_wiki:.3f}"],
        ["TinyStories", len(fiction_text), fiction_tokens, f"{ratio_fiction:.3f}"],
        ["Total", len(wiki_text) + len(fiction_text), total, "1.000"],
    ]

    print_table(rows, headers=["Dataset", "Documents", "Tokens", "Sampling Ratio"])

    return wiki_text + fiction_text


def load_and_tokenize(file_path, model_name="gpt2"):
    print(f"Loading and tokenizing {file_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    rows = [["Target Corpus", len(tokens)]]

    print_table(rows, headers=["Corpus", "Tokens"])

    return tokens


if __name__ == "__main__":
    input_file = "Harry_Potter_all_books_preprocessed.txt"
    output_file = "src/data/target_corpus.txt"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    clean_text(input_file, output_file)

    load_and_tokenize(output_file)

    get_neutral_corpus()
