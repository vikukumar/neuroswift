"""
neuroswift.benchmark
====================
God-level performance and accuracy benchmarking for NeuroSwift models.
Measures speed (TPS), Memory, and Reasoning Accuracy.
"""
from __future__ import annotations

import time
import logging
import torch
from pathlib import Path
from typing import Optional, Any

from .model import NeuroSwiftLM
from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)


class NeuroSwiftBenchmark:
    """
    Inbuilt benchmark suite for NeuroSwift models.
    """
    def __init__(self, model: NeuroSwiftLM, tokenizer: WordTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def run_speed_test(self, prompt: str = "Explain the future of AI.", n_tokens: int = 128) -> dict[str, float]:
        """Measure tokens per second on current device."""
        logger.info(f"Running speed benchmark on {self.device}...")
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        # Warmup
        self.model.generate(input_ids, max_new_tokens=10)
        
        start = time.perf_counter()
        # Actual test
        generated = self.model.generate(input_ids, max_new_tokens=n_tokens)
        end = time.perf_counter()
        
        duration = end - start
        tps = n_tokens / duration
        memory_mb = 0.0
        if self.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        return {
            "tokens_per_second": tps,
            "latency_ms": duration * 1000,
            "memory_mb": memory_mb,
            "device": str(self.device)
        }

    def run_accuracy_test(self, test_data: list[dict[str, str]]) -> dict[str, float]:
        """Measure exact match and reasoning score on a test set."""
        score = 0.0
        count = 0
        for item in test_data:
            prompt, expected = item["prompt"], item["response"]
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            gen_ids = self.model.generate(input_ids, max_new_tokens=64)
            actual = self.tokenizer.decode(gen_ids[0].tolist()).strip()
            
            # Trivial Exact Match + fuzzy contain
            if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                score += 1.0
            count += 1
            
        return {"accuracy": score / max(1, count), "count": count}

    def run_all(self, test_data: Optional[list[dict[str, str]]] = None) -> None:
        """Run full benchmark report."""
        speed = self.run_speed_test()
        print("\n" + "="*40)
        print(" NEUROSWIFT GOD-LEVEL BENCHMARK ")
        print("="*40)
        print(f" DEVICE:    {speed['device']}")
        print(f" SPEED:     {speed['tokens_per_second']:.2f} t/s")
        print(f" LATENCY:   {speed['latency_ms']:.2f} ms")
        if speed['memory_mb'] > 0:
            print(f" GPU MEM:   {speed['memory_mb']:.2f} MB")
        
        if test_data:
            acc = self.run_accuracy_test(test_data)
            print(f" ACCURACY:  {acc['accuracy']*100:.1f}% ({acc['count']} samples)")
        print("="*40 + "\n")
