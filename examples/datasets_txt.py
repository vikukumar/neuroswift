import os
from datasets import load_dataset

def generate_training_file(filename="examples/data/neuroswift_corpus.txt", target_lines=100000):
    print("Downloading WikiText dataset (this may take a moment)...")
    # We use wikitext-103, a standard dataset for training language models
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    lines_written = 0
    
    print(f"Generating {filename} with {target_lines} lines...")
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item["text"].strip()
            
            # Skip empty lines or standard wikipedia headers to keep data dense
            if text and len(text) > 20 and not text.startswith('='):
                f.write(text + "\n")
                lines_written += 1
                
                # Print progress
                if lines_written % 10000 == 0:
                    print(f"Progress: {lines_written}/{target_lines} lines written...")
                    
            if lines_written >= target_lines:
                break
                
    print(f"\nSuccess! '{filename}' has been created with {lines_written} lines.")
    print(f"File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    generate_training_file()