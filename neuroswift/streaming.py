"""
neuroswift.streaming
====================
SSD-backed data streaming for God-level throughput on low-RAM systems.
Parallelized with ProcessPoolExecutor for 'Invincible Speed'.
"""
from __future__ import annotations

import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List

import numpy as np
import torch
from torch.utils.data import Dataset # Corrected from IterableDataset

from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)


def _tokenize_worker_mmap(args):
    """Parallel worker to tokenize texts for memmap."""
    pair, tokenizer = args
    # Consistent format for mmap: user: {prompt}\nassistant: {response}<eos>
    text = f"user: {pair.prompt}\nassistant: {pair.response}"
    return tokenizer.encode(text, add_eos=True)


class MmapDataset(Dataset): # Corrected from IterableDataset
    """
    God-level dataset that streams tokens from an SSD-backed memmap file.
    Ideal for multi-gigabyte datasets that don't fit in RAM.
    Supports index-based random access for high-speed shuffling.
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
            
        filesize = os.path.getsize(mmap_path)
        self.total_tokens = filesize // np.dtype(dtype).itemsize
        # One extra token for shifted targets
        self.num_samples = (self.total_tokens - 1) // seq_len
        
        self._data = None
        logger.info(f"Loaded MmapDataset: {self.num_samples:,} samples ({self.total_tokens:,} tokens)")

    def _get_data(self):
        if self._data is None:
            self._data = np.memmap(self.mmap_path, dtype=self.dtype, mode="r")
        return self._data

    def __len__(self) -> int:
        """Returns the total number of samples in the memmap dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides fast random access to the raw SSD-backed buffer."""
        data = self._get_data()
        start = index * self.seq_len
        end = start + self.seq_len + 1
        
        # Consistent shape for the model training pipeline
        if end > len(data):
            chunk = np.zeros(self.seq_len + 1, dtype=np.int64)
            available = data[start:]
            chunk[:len(available)] = available.astype(np.int64)
        else:
            chunk = data[start:end].astype(np.int64)
            
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])
        return input_ids, labels

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Maintains simple streaming support."""
        for i in range(self.num_samples):
            yield self[i]

    @classmethod
    def from_pairs(
        cls,
        pairs: List[Any],
        tokenizer: WordTokenizer,
        mmap_path: Path,
        seq_len: int = 128,
    ) -> "MmapDataset":
        """Tokenize pairs in parallel and write to SSD memmap for world-class throughput."""
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Parallel Tokenizing {len(pairs):,} pairs to SSD ({mmap_path.name})...")
        
        all_tokens = []
        if len(pairs) > 1000:
            num_procs = min(multiprocessing.cpu_count(), 16)
            worker_args = [(p, tokenizer) for p in pairs]
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                # Large chunksize to minimize IPC overhead for 5GB throughput
                token_lists = list(executor.map(_tokenize_worker_mmap, worker_args, chunksize=1000))
            for tlist in token_lists:
                all_tokens.extend(tlist)
        else:
            # Sequential for tiny datasets
            for pair in pairs:
                all_tokens.extend(_tokenize_worker_mmap((pair, tokenizer)))
            
        # Pad to (num_samples * seq_len) + 1 for shifted targets
        pad_id = tokenizer.stoi[tokenizer.pad_token]
        n_samples = int(np.ceil(len(all_tokens) / seq_len))
        target_len = (n_samples * seq_len) + 1
        
        if len(all_tokens) < target_len:
            all_tokens.extend([pad_id] * (target_len - len(all_tokens)))
            
        # High-speed disk dump
        logger.info(f"Writing {len(all_tokens):,} tokens to {mmap_path} ...")
        arr = np.array(all_tokens, dtype=np.uint16)
        fp = np.memmap(mmap_path, dtype="uint16", mode="w+", shape=arr.shape)
        fp[:] = arr[:]
        fp.flush()
        del fp
        
        return cls(mmap_path, seq_len=seq_len)
