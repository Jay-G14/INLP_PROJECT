import re
import os
import datasets
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def clean_text(input_path, output_path):
    """
    Reads the input file and saves the cleaned text (first ~3 books).
    """
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # The file seems to be preprocessed (no newlines?), or maybe it has them.
    # We'll just take the first portion.
    # Estimate: Book 1-3 is approx 270k words.
    # Avg word length 5 chars + 1 space = 6.
    # 270k * 6 = 1.62M chars.
    # Let's be generous and take 1.8M chars to be safe, or just cut by book if possible.
    target_text = full_text
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(target_text)
        
    print(f"Saved processed text to {output_path}. Length: {len(target_text)} chars.")
    return target_text

def get_neutral_corpus(split="train"):
    # Load a subset of Wikitext-2
    print("Loading neutral corpus (wikitext-2)...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    wiki_text = [t for t in dataset["text"] if t.strip()]
    
    # Add a fictional component to broaden the baseline
    print("Loading fictional neutral corpus (TinyStories)...")
    try:
        fiction_ds = datasets.load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        fiction_text = []
        it = iter(fiction_ds)
        for _ in range(2000): # Take 2000 samples
            fiction_text.append(next(it)["text"])
        return wiki_text + fiction_text
    except Exception as e:
        print(f"Warning: Could not load TinyStories ({e}). Falling back to WikiText-2 only.")
        return wiki_text

def load_and_tokenize(file_path, model_name="meta-llama/Llama-2-7b-hf"):
    print(f"Loading and tokenizing {file_path}...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Loaded {len(tokens)} tokens.")
    return tokens

if __name__ == "__main__":
    input_file = "Harry_Potter_all_books_preprocessed.txt"
    output_file = "src/data/target_corpus.txt"
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    clean_text(input_file, output_file)
    get_neutral_corpus() # Just to verify it loads
