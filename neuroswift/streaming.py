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
from torch.utils.data import IterableDataset

from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)


def _tokenize_worker_mmap(args):
    """Parallel worker to tokenize texts for memmap."""
    pair, tokenizer = args
    # Consistent format for mmap: user: {prompt}\nassistant: {response}<eos>
    text = f"user: {pair.prompt}\nassistant: {pair.response}"
    return tokenizer.encode(text, add_eos=True)


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
            
        filesize = os.path.getsize(mmap_path)
        self.total_tokens = filesize // np.dtype(dtype).itemsize
        self.num_samples = self.total_tokens // seq_len
        
        logger.info(f"Loaded MmapDataset: {self.num_samples:,} samples ({self.total_tokens:,} tokens)")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        data = np.memmap(self.mmap_path, dtype=self.dtype, mode="r")
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            iter_range = range(self.num_samples)
        else:
            per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))
            iter_range = range(worker_info.id * per_worker, min((worker_info.id + 1) * per_worker, self.num_samples))

        for i in iter_range:
            start = i * self.seq_len
            end = start + self.seq_len
            chunk = data[start:end].astype(np.int64)
            
            if len(chunk) < self.seq_len:
                continue
                
            input_ids = torch.from_numpy(chunk)
            yield {
                "input_ids": input_ids,
            }

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
            
        # Pad to multiple of seq_len
        pad_id = tokenizer.stoi[tokenizer.pad_token]
        rem = len(all_tokens) % seq_len
        if rem > 0:
            all_tokens.extend([pad_id] * (seq_len - rem))
            
        # High-speed disk dump
        logger.info(f"Writing {len(all_tokens):,} tokens to {mmap_path} ...")
        arr = np.array(all_tokens, dtype=np.uint16)
        fp = np.memmap(mmap_path, dtype="uint16", mode="w+", shape=arr.shape)
        fp[:] = arr[:]
        fp.flush()
        del fp
        
        return cls(mmap_path, seq_len=seq_len)
