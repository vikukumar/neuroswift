"""
neuroswift.streaming
====================
SSD-backed data streaming for God-level throughput on low-RAM systems.

Uses numpy.memmap to store tokenized datasets on disk and stream them 
directly into training loops.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)


class MmapDataset(IterableDataset):
    """
    God-level dataset that streams tokens from an SSD-backed memmap file.
    Ideal for multi-gigabyte datasets that don't fit in RAM.
    """
    def __init__(
        self,
        mmap_path: Path,
        seq_len: int = 128,
        dtype: np.dtype = np.uint16,
    ) -> None:
        super().__init__()
        self.mmap_path = mmap_path
        self.seq_len = seq_len
        self.dtype = dtype
        
        if not mmap_path.exists():
            raise FileNotFoundError(f"Mmap file not found: {mmap_path}")
            
        # Determine total tokens by file size
        filesize = os.path.getsize(mmap_path)
        self.total_tokens = filesize // np.dtype(dtype).itemsize
        self.num_samples = self.total_tokens // seq_len
        
        logger.info(f"Loaded MmapDataset: {self.num_samples:,} samples ({self.total_tokens:,} tokens)")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Open memmap in read-only mode
        data = np.memmap(self.mmap_path, dtype=self.dtype, mode="r")
        
        # Shuffle conceptually via random start offsets if multiple workers, 
        # but for simple God-level, we just stream.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_range = range(self.num_samples)
        else:
            # Split work across data loader workers
            per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))
            iter_range = range(worker_info.id * per_worker, min((worker_info.id + 1) * per_worker, self.num_samples))

        for i in iter_range:
            start = i * self.seq_len
            end = start + self.seq_len
            chunk = data[start:end].astype(np.int64)
            
            # Simple causal LM pair (input, target)
            # Actually, the memmap should store pre-tokenized sequences
            if len(chunk) < self.seq_len:
                continue
                
            input_ids = torch.from_numpy(chunk)
            # For causal LM, targets are shifted input_ids
            # (In god level, we assume the memmap contains [input_ids_with_one_extra_for_target])
            yield {
                "input_ids": input_ids,
            }

    @classmethod
    def from_pairs(
        cls,
        pairs: list[Any],
        tokenizer: WordTokenizer,
        mmap_path: Path,
        seq_len: int = 128,
    ) -> "MmapDataset":
        """Tokenize a list of TrainPairs and write them to a memmap file on disk."""
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Estimate total tokens
        logger.info(f"Tokenizing {len(pairs):,} pairs to disk...")
        all_tokens = []
        for pair in pairs:
            # Format: user: {prompt}\nassistant: {response}<eos>
            text = f"user: {pair.prompt}\nassistant: {pair.response}"
            ids = tokenizer.encode(text, add_eos=True)
            all_tokens.extend(ids)
            
        # Pad to multiple of seq_len
        pad_id = tokenizer.stoi[tokenizer.pad_token]
        rem = len(all_tokens) % seq_len
        if rem > 0:
            all_tokens.extend([pad_id] * (seq_len - rem))
            
        # Write to disk
        arr = np.array(all_tokens, dtype=np.uint16)
        fp = np.memmap(mmap_path, dtype="uint16", mode="w+", shape=arr.shape)
        fp[:] = arr[:]
        fp.flush()
        del fp
        
        return cls(mmap_path, seq_len=seq_len)
