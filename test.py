# test_llm_fixed.py

from __future__ import annotations
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import torch

from neuroswift.layers import auto_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("NeuroSwift.Test")


# =========================
# FIX: VOCAB TYPE HANDLER
# =========================
def _fix_vocab_types(vocab: dict):
    fixed = {}
    for k, v in vocab.items():
        try:
            fixed[int(k)] = v
        except:
            fixed[k] = v
    return fixed


# =========================
# LOAD MODEL + TOKENIZER
# =========================
def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    from neuroswift.tokenizer import HybridTokenizer
    from neuroswift.model import NeuroSwiftLM

    # v35: Standardized loading matching train_small_llm.py
    tokenizer = HybridTokenizer.from_texts([], vocab_size=12000, output_dir=model_dir)
    
    # Health Check 1: Vocab Size
    if tokenizer.vocab_size < 100:
        logger.warning(f"Tokenizer at {model_dir} appears corrupted (vocab={tokenizer.vocab_size}).")
    
    # Health Check 2: Decoding Sanity
    test_str = "NeuroSwift Tokenizer Check"
    try:
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded)
        if not decoded or len(decoded) < 5:
            logger.error("Tokenizer decoding sanity check failed!")
    except Exception as e:
        logger.error(f"Tokenizer health check crashed: {e}")

    model = NeuroSwiftLM.from_pretrained(model_dir, device=device)
    return model, tokenizer


# =========================
# ARGUMENTS
# =========================
def parse_args():
    p = ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True)

    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.2)

    return p.parse_args()


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    device = auto_device()

    print("\n" + "═" * 60)
    print("NeuroSwift Inference (FIXED)")
    print(f"Model: {args.model_dir}")
    print(f"Device: {device}")
    print("═" * 60)

    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    prompt_text = f"instruction: {args.prompt}\nresponse:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

    with torch.no_grad():
        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            adapt_during_generation=False,
        )

    answer_ids = gen_ids[0, prompt_ids.size(1):].tolist()

    # =========================
    # DEBUG TOKEN STREAM
    # =========================
    print("\nDEBUG TOKENS:")
    for i in answer_ids[:30]:
        try:
            print(i, "->", tokenizer.decode([i]))
        except:
            print(i, "-> [decode error]")

    # =========================
    # FINAL OUTPUT
    # =========================
    answer = tokenizer.decode(answer_ids).strip()

    print("\nANSWER:")
    print(answer)
    print()


if __name__ == "__main__":
    main()