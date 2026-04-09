"""
neuroswift.dataset_downloader
=============================
Auto-Intelligence V3 utility for automatic dataset fetching.
Optimized with Asyncio and ThreadPools for 'Invincible Speed'.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("NeuroSwift.Downloader")

def _check_cache_exists(repo_id: str, prefix: str = "") -> Optional[Path]:
    """
    Robust check if a dataset is already cached.
    Uses artifacts/datasets/<id> as the primary storage.
    """
    safe_name = repo_id.replace("/", "_")
    path = Path("artifacts/datasets") / f"{prefix}{safe_name}"
    
    # Check if folder exists and has files (indicating a successful previous download)
    if path.exists() and path.is_dir():
        # Check for any non-hidden files to confirm it's not just an empty scaffold
        if any(f for f in path.iterdir() if not f.name.startswith(".")):
            return path
    return None

def _download_hf_single(repo_id: str) -> Path:
    """Uses 'datasets' library to fetch data and export to JSONL for pipeline flow."""
    import json
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Dataset library NOT found. Please install: pip install datasets")
        raise

    safe_name = repo_id.replace("/", "_")
    output_dir = Path("artifacts/datasets") / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dataset.jsonl"
    
    # One-time download check
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return output_dir

    logger.info(f"Loading HF dataset '{repo_id}' via datasets library...")
    try:
        # load_dataset manages its own cache, but we export it to our local artifacts 
        # to ensure the existing pipeline's directory-based ingestion works perfectly.
        ds = load_dataset(repo_id)
        
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            if hasattr(ds, "keys"):
                for split_name in ds.keys():
                    for row in ds[split_name]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
            else:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        
        logger.info(f"Successfully cached {count:,} samples from '{repo_id}' to {jsonl_path}")
        return output_dir
    except Exception as e:
        logger.warning(f"Datasets library failed for {repo_id}: {e}. Falling back to snapshot_download.")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir
        )
        return output_dir

def _download_kaggle_single(dataset_id: str) -> Path:
    """Synchronous worker for Kaggle download."""
    import kaggle
    output_dir = Path("artifacts/datasets") / f"kaggle_{dataset_id.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assumes authenticate() was called once globally
    kaggle.api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
    return output_dir

async def _async_get_hf(repo_ids: List[str]) -> List[Path]:
    """Async coordinator for HF downloads."""
    loop = asyncio.get_running_loop()
    results = []
    
    # Filter out cached
    to_download = []
    for rid in repo_ids:
        cached = _check_cache_exists(rid)
        if cached:
            logger.info(f"HF Cache Hit: {rid}")
            results.append(cached)
        else:
            to_download.append(rid)
            
    if not to_download:
        return results

    logger.info(f"Parallel Downloading {len(to_download)} HF datasets...")
    # Use thread pool to avoid blocking the event loop
    with ThreadPoolExecutor(max_workers=min(4, len(to_download))) as pool:
        dl_tasks = [loop.run_in_executor(pool, _download_hf_single, rid) for rid in to_download]
        dl_results = await asyncio.gather(*dl_tasks)
        results.extend(dl_results)
        
    return results

async def _async_get_kaggle(dataset_ids: List[str]) -> List[Path]:
    """Async coordinator for Kaggle downloads."""
    import kaggle
    loop = asyncio.get_running_loop()
    results = []
    
    to_download = []
    for did in dataset_ids:
        cached = _check_cache_exists(did, prefix="kaggle_")
        if cached:
            logger.info(f"Kaggle Cache Hit: {did}")
            results.append(cached)
        else:
            to_download.append(did)
            
    if not to_download:
        return results

    # Re-auth once
    try:
        kaggle.api.authenticate()
    except Exception as e:
        logger.error(f"Kaggle Auth Failed: {e}")
        raise e

    logger.info(f"Batch Downloading {len(to_download)} Kaggle datasets...")
    # Kaggle API is not strictly thread-safe, so we use a small pool 
    # to avoid race conditions but still gain I/O overlap
    with ThreadPoolExecutor(max_workers=2) as pool:
        dl_tasks = [loop.run_in_executor(pool, _download_kaggle_single, did) for did in to_download]
        dl_results = await asyncio.gather(*dl_tasks)
        results.extend(dl_results)
        
    return results

def download_from_hf(repo_ids: str | list[str]) -> list[Path]:
    """High-level sync entry for HF downloads using asyncio internally."""
    ids = [repo_ids] if isinstance(repo_ids, str) else repo_ids
    return asyncio.run(_async_get_hf(ids))

def download_from_kaggle(dataset_ids: str | list[str]) -> list[Path]:
    """High-level sync entry for Kaggle downloads using asyncio internally."""
    ids = [dataset_ids] if isinstance(dataset_ids, str) else dataset_ids
    return asyncio.run(_async_get_kaggle(ids))
