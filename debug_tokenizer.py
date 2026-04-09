import logging
from neuroswift.tokenizer import HybridTokenizer
import os

logging.basicConfig(level=logging.INFO)

# 1. Test Simple Encoding
vocab_dir = "artifacts/neurollm4"
if not os.path.exists(vocab_dir):
    print(f"Error: {vocab_dir} does not exist")
else:
    try:
        tokenizer = HybridTokenizer.from_pretrained(vocab_dir)
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        test_text = "Who founded Amazon?\nJeff Bezos"
        ids = tokenizer.encode(test_text)
        print(f"Text: {test_text}")
        print(f"IDs: {ids}")
        print(f"Decoded: {tokenizer.decode(ids)}")
        
        if not ids:
            print("CRITICAL: Tokenizer returned EMPTY list for valid text!")
    except Exception as e:
        print(f"Error loading/testing tokenizer: {e}")

# 2. Check sanitize_text
from neuroswift.data_pipeline import sanitize_text
print(f"Sanitized: '{sanitize_text('Who founded Amazon?')}'")
