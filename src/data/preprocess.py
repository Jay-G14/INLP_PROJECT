import re
import os
import datasets
from transformers import GPT2Tokenizer

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
    # Searching for "Harry Potter and the Goblet of Fire" (Book 4) title might work.
    
    # Book 4 Start: "Harry Potter and the Goblet of Fire"
    # Note: The text seems to be stripped of some formatting. 
    # Let's try to find "THE VILLAGE OF LITTLE HANGLETON" (Chapter 1 of Book 4).
    # Or just "The Riddle House".
    
    book4_start = full_text.find("THE VILLAGE OF LITTLE HANGLETON")
    if book4_start == -1:
        # Try finding Book 4 title
        book4_start = full_text.find("Harry Potter and the Goblet of Fire")
    
    if book4_start != -1:
        print(f"Found Book 4 start at index {book4_start}. Truncating there.")
        target_text = full_text[:book4_start]
    else:
        print("Could not find Book 4 start. Using length heuristic (1.6M chars).")
        target_text = full_text[:1600000] # Fallback
    
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
    print(f"Loading and tokenizing {file_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Tokenize
    # This might be slow for large files without chunking, but for 1.6MB it should be fine.
    tokens = tokenizer.encode(text)
    print(f"Loaded {len(tokens)} tokens.")
    return tokens

if __name__ == "__main__":
    input_file = "/home/jay/inlp/Harry_Potter_all_books_preprocessed.txt"
    output_file = "/home/jay/inlp/src/data/target_corpus.txt"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    clean_text(input_file, output_file)
    get_neutral_corpus() # Just to verify it loads
