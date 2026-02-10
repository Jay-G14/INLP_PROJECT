import re
import os
import datasets
from transformers import AutoTokenizer

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def clean_text(input_path, output_path):
    """
    Reads the input file and saves the cleaned text (first ~3 books).
    """
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Book 4 Start: "Harry Potter and the Goblet of Fire"
    # Try to find "THE VILLAGE OF LITTLE HANGLETON" (Chapter 1 of Book 4)
    book4_start = full_text.find("THE VILLAGE OF LITTLE HANGLETON")
    if book4_start == -1:
        book4_start = full_text.find("Harry Potter and the Goblet of Fire")
    
    if book4_start != -1:
        print(f"Found Book 4 start at index {book4_start}. Truncating there.")
        target_text = full_text[:book4_start]
    else:
        print("Could not find Book 4 start. Using length heuristic (1.6M chars).")
        target_text = full_text[:1600000]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(target_text)
        
    print(f"Saved processed text to {output_path}. Length: {len(target_text)} chars.")
    return target_text

def get_neutral_corpus(split="train", num_samples=1000):
    # Load a subset of Wikitext-2
    print("Loading neutral corpus (wikitext-2)...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return dataset

def load_and_tokenize(file_path, model_name="gpt2"):
    print(f"Loading and tokenizing {file_path} using {model_name} tokenizer...")
    
    # Handle relative paths - make them relative to project root
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    print(f"Loaded {len(tokens)} tokens.")
    return tokens

if __name__ == "__main__":
    # Use paths relative to project structure
    input_file = os.path.join(PROJECT_ROOT, "Harry_Potter_all_books_preprocessed.txt")
    output_file = os.path.join(SCRIPT_DIR, "target_corpus.txt")
    
    clean_text(input_file, output_file)
    get_neutral_corpus()  # Just to verify it loads

