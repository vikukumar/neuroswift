"""
neuroswift.data_pipeline
========================
Built-in data pre-processing pipeline for NeuroSwift training.

Runs automatically before every training or fine-tuning job to ensure
all data is clean, de-duplicated, well-formatted, and properly split
into train / validation sets.

Pipeline stages (in order)
--------------------------
1. **Ingest** — read all supported file types from any directory or file
2. **Extract** — convert every file into raw text / instruction pairs
3. **Normalize** — Unicode NFD→NFC, whitespace collapse, encoding repair
4. **Filter** — remove empty, too-short, too-long, or repetitive examples
5. **Deduplicate** — exact + near-duplicate removal (minhash fingerprint)
6. **Augment** — light synonym / word-order augmentation (optional)
7. **Format** — convert to unified ``{"prompt": ..., "response": ...}`` pairs
8. **Split** — deterministic train / validation split

Supports all source types:
  - .jsonl / .json — instruction pairs, ShareGPT, OpenAI chat format
  - .txt / .md / .rst — plain text chunked into overlapping windows
  - .csv / .tsv / .xlsx — tabular → QA pairs per row
  - .pdf → (text, if pdfminer installed) → windowed chunks
  - Images, audio, video → captioned via MultimodalSample.to_rag_text()
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainPair:
    """Unified instruction-response pair after full pipeline processing."""

    prompt: str
    response: str
    source: str = ""
    modality: str = "text"
    quality_score: float = 1.0  # 0.0–1.0

    def to_dict(self) -> dict[str, str]:
        return {"prompt": self.prompt, "response": self.response}


@dataclass
class PipelineStats:
    """Counters updated during pipeline processing."""

    raw_files: int = 0
    raw_pairs: int = 0
    after_normalize: int = 0
    after_filter: int = 0
    after_dedup: int = 0
    skipped_empty: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    skipped_dedup: int = 0
    train_pairs: int = 0
    val_pairs: int = 0

    def report(self) -> str:
        lines = [
            "-- Data Pipeline Report ---------------------------",
            f"  Raw files:        {self.raw_files:>6,}",
            f"  Raw pairs:        {self.raw_pairs:>6,}",
            f"  After normalize:  {self.after_normalize:>6,}",
            f"  After filter:     {self.after_filter:>6,}",
            f"    (too short:     {self.skipped_too_short:>6,})",
            f"    (too long:      {self.skipped_too_long:>6,})",
            f"    (empty:         {self.skipped_empty:>6,})",
            f"  After dedup:      {self.after_dedup:>6,}",
            f"    (dupes removed: {self.skipped_dedup:>6,})",
            f"  Train set:        {self.train_pairs:>6,}",
            f"  Val   set:        {self.val_pairs:>6,}",
            "---------------------------------------------------",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 + 2: Ingest & Extract
# ---------------------------------------------------------------------------

class UniversalSchemaMapper:
    """
    God-level schema mapper that automatically finds 'prompt' and 'response' 
    like fields in any dictionary using fuzzy matching and heuristics.
    """
    _PROMPT_HINTS = {"prompt", "instruction", "input", "question", "human", "user", "query", "q", "title", "header", "topic"}
    _RESP_HINTS = {"response", "output", "answer", "assistant", "gpt", "model", "a", "body", "content", "text", "summary", "description"}

    @classmethod
    def map_obj(cls, obj: dict[str, Any], source: str = "") -> TrainPair | None:
        if not isinstance(obj, dict):
            return None
        
        # 1. Exact match pass
        for pk, rk in _INSTRUCTION_KEYS:
            p, r = str(obj.get(pk, "")).strip(), str(obj.get(rk, "")).strip()
            if p and r: return TrainPair(prompt=p, response=r, source=source)

        # 2. Fuzzy match pass
        keys = list(obj.keys())
        p_key, r_key = None, None
        
        # Heuristic: longest text is usually the response, second longest or 'question' like is prompt
        sorted_by_len = sorted([k for k in keys if isinstance(obj[k], str)], key=lambda k: len(str(obj[k])), reverse=True)
        
        if not sorted_by_len: return None

        # Look for indicators
        for k in sorted_by_len:
            lk = k.lower()
            if any(hint in lk for hint in cls._RESP_HINTS) and not r_key:
                r_key = k
            elif any(hint in lk for hint in cls._PROMPT_HINTS) and not p_key:
                p_key = k

        # Fallback: take longest as response, second longest as prompt if no hints found
        if not r_key: r_key = sorted_by_len[0]
        if not p_key and len(sorted_by_len) > 1: p_key = sorted_by_len[1]

        if p_key and r_key and p_key != r_key:
            p, r = str(obj[p_key]).strip(), str(obj[r_key]).strip()
            if len(p) > 5 and len(r) > 10:
                return TrainPair(prompt=p, response=r, source=source)
        
        return None


def _extract_pair_from_dict(obj: dict[str, Any], source: str = "") -> TrainPair | None:
    """Delegates to UniversalSchemaMapper."""
    return UniversalSchemaMapper.map_obj(obj, source=source)


class WebScraper:
    """Simple inbuilt web scraper for URL ingestion."""
    @staticmethod
    def scrape(url: str) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts/styles
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return ""


def _read_jsonl(path: Path, cap_bytes: int = 50 * 1024 * 1024) -> Iterator[TrainPair]:
    """Stream pairs from a JSONL file, capped at cap_bytes to avoid OOM."""
    read_bytes = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                read_bytes += len(line.encode())
                if read_bytes > cap_bytes:
                    logger.debug(f"JSONL cap reached at {cap_bytes // 1024 // 1024} MB for {path}")
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pair = _extract_pair_from_dict(obj, source=str(path))
                    if pair:
                        yield pair
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning(f"Error reading JSONL {path}: {exc}")


def _read_json(path: Path) -> Iterator[TrainPair]:
    """Read a JSON file (list of objects or single object)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 20 * 1024 * 1024:
            text = text[:20 * 1024 * 1024]
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    pair = _extract_pair_from_dict(item, source=str(path))
                    if pair:
                        yield pair
        elif isinstance(obj, dict):
            pair = _extract_pair_from_dict(obj, source=str(path))
            if pair:
                yield pair
    except Exception as exc:
        logger.warning(f"Error reading JSON {path}: {exc}")


def _text_to_pairs(text: str, source: str, chunk_words: int = 150, stride_words: int = 75) -> Iterator[TrainPair]:
    """Chunk plain text into overlapping windows and generate summary-style pairs."""
    words = text.split()
    if not words:
        return
    source_name = Path(source).name if "/" in source or "\\" in source else source
    for start in range(0, max(1, len(words) - chunk_words // 2), stride_words):
        chunk = " ".join(words[start: start + chunk_words])
        if len(chunk) < 40:
            continue
        # Make a simple comprehension prompt
        prompt = f"Summarize the following text from {source_name}:\n{chunk[:400]}"
        response = chunk[:800]
        yield TrainPair(prompt=prompt, response=response, source=source)


def _read_text(path: Path, cap_bytes: int = 5 * 1024 * 1024) -> Iterator[TrainPair]:
    """Read plain text, markdown, RST files; chunk into pairs."""
    try:
        raw = path.read_bytes()[:cap_bytes].decode("utf-8", errors="replace")
        yield from _text_to_pairs(raw, source=str(path))
    except Exception as exc:
        logger.warning(f"Error reading text {path}: {exc}")


def _read_csv(path: Path) -> Iterator[TrainPair]:
    """Read CSV/TSV and generate QA pairs from each non-header row."""
    import csv
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            for row in reader:
                vals = [str(v).strip() for v in row.values() if str(v).strip()]
                keys = list(row.keys())
                if not vals:
                    continue
                # If first col looks like prompt and second like response
                if len(keys) >= 2:
                    p = str(row.get(keys[0], "")).strip()
                    r = str(row.get(keys[1], "")).strip()
                    if p and r:
                        yield TrainPair(prompt=p, response=r, source=str(path), modality="table")
                        continue
                # Otherwise: field: value summary
                row_text = "; ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
                if row_text:
                    yield TrainPair(
                        prompt=f"Describe this record from {path.name}",
                        response=row_text,
                        source=str(path),
                        modality="table",
                    )
    except Exception as exc:
        logger.warning(f"Error reading CSV {path}: {exc}")


def _read_xlsx(path: Path) -> Iterator[TrainPair]:
    """Read Excel and generate pairs per row."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            headers: list[str] = []
            for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
                cells = [str(c).strip() if c is not None else "" for c in row]
                if row_idx == 0:
                    headers = cells
                    continue
                if not any(cells):
                    continue
                if headers and len(headers) >= 2:
                    p = cells[0] if cells else ""
                    r = cells[1] if len(cells) > 1 else ""
                    if p and r:
                        yield TrainPair(prompt=p, response=r, source=str(path), modality="table")
                        continue
                row_text = "; ".join(
                    f"{h}: {c}" for h, c in zip(headers, cells) if c
                ) if headers else "; ".join(c for c in cells if c)
                if row_text:
                    yield TrainPair(
                        prompt=f"Describe this Excel record from {path.name}",
                        response=row_text,
                        source=str(path),
                        modality="table",
                    )
    except ImportError:
        logger.warning("openpyxl not installed, skipping .xlsx files.")
    except Exception as exc:
        logger.warning(f"Error reading XLSX {path}: {exc}")


def _read_multimodal(path: Path) -> Iterator[TrainPair]:
    """Delegate image/audio/video to DatasetFolderReader → caption pairs."""
    try:
        from .ingest import DatasetFolderReader
        reader = DatasetFolderReader()
        sample_or_list = reader.read_path(path)
        if sample_or_list is None:
            return
        samples = sample_or_list if isinstance(sample_or_list, list) else [sample_or_list]
        for sample in samples:
            rag_text = sample.to_rag_text()
            words = rag_text.split()
            summary = " ".join(words[:150])
            if len(summary) < 20:
                continue
            name = Path(sample.source).name if "://" not in sample.source else sample.source
            yield TrainPair(
                prompt=f"Describe the {sample.modality} content of {name}.",
                response=summary,
                source=str(path),
                modality=sample.modality,
            )
            yield TrainPair(
                prompt=f"What is in the file {name}?",
                response=f"The {sample.modality} file {name} contains: {summary}",
                source=str(path),
                modality=sample.modality,
            )
    except Exception as exc:
        logger.debug(f"Multimodal read skipped for {path}: {exc}")


_EXT_READERS: dict[str, Any] = {
    ".jsonl": _read_jsonl,
    ".json": _read_json,
    ".txt": _read_text,
    ".md": _read_text,
    ".rst": _read_text,
    ".csv": _read_csv,
    ".tsv": _read_csv,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
    ".png": _read_multimodal,
    ".jpg": _read_multimodal,
    ".jpeg": _read_multimodal,
    ".webp": _read_multimodal,
    ".bmp": _read_multimodal,
    ".wav": _read_multimodal,
    ".flac": _read_multimodal,
    ".mp3": _read_multimodal,
    ".ogg": _read_multimodal,
    ".mp4": _read_multimodal,
    ".mov": _read_multimodal,
    ".avi": _read_multimodal,
    ".mkv": _read_multimodal,
}


def ingest_path(path: Path) -> Iterator[TrainPair]:
    """Dispatch a single file path to the appropriate reader."""
    ext = path.suffix.lower()
    reader_fn = _EXT_READERS.get(ext)
    if reader_fn is None:
        return
    yield from reader_fn(path)


def ingest_directory(
    data_dir: Path,
    max_pairs_per_file: int = 10_000,
    skip_exts: set[str] | None = None,
) -> tuple[list[TrainPair], PipelineStats]:
    """
    Recursively read all supported files from *data_dir*.

    Args:
        data_dir: Root folder to scan.
        max_pairs_per_file: Cap extracted pairs per file (prevents huge imbalance).
        skip_exts: Set of extensions to skip (e.g. ``{".wav"}``).

    Returns:
        ``(all_pairs, stats)`` where stats is a :class:`PipelineStats` snapshot.
    """
    stats = PipelineStats()
    skip_exts = skip_exts or set()
    raw: list[TrainPair] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in skip_exts:
            continue
        if path.suffix.lower() not in _EXT_READERS:
            continue

        stats.raw_files += 1
        file_pairs = 0
        try:
            for pair in ingest_path(path):
                raw.append(pair)
                stats.raw_pairs += 1
                file_pairs += 1
                if file_pairs >= max_pairs_per_file:
                    logger.debug(f"Pair cap ({max_pairs_per_file}) reached for {path}")
                    break
        except Exception as exc:
            logger.warning(f"Failed reading {path}: {exc}")

    return raw, stats


def ingest_file(path: Path, max_pairs: int = 50_000) -> tuple[list[TrainPair], PipelineStats]:
    """Read a single file and return pairs + stats."""
    stats = PipelineStats()
    raw: list[TrainPair] = []
    stats.raw_files = 1
    for pair in ingest_path(path):
        raw.append(pair)
        stats.raw_pairs += 1
        if max_pairs and stats.raw_pairs >= max_pairs:
            break
    return raw, stats


# ---------------------------------------------------------------------------
# Stage 3: Normalize
# ---------------------------------------------------------------------------

_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Unicode NFC, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    text = text.strip()
    return text


def normalize_pair(pair: TrainPair) -> TrainPair:
    return TrainPair(
        prompt=normalize_text(pair.prompt),
        response=normalize_text(pair.response),
        source=pair.source,
        modality=pair.modality,
        quality_score=pair.quality_score,
    )


# ---------------------------------------------------------------------------
# Stage 4: Filter
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def _is_repetitive(text: str, n: int = 4, threshold: float = 0.35) -> bool:
    """True if the most common n-gram makes up > threshold of all n-grams."""
    words = text.lower().split()
    if len(words) < n * 3:
        return False
    from collections import Counter
    ngrams = [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False
    most_common_count = Counter(ngrams).most_common(1)[0][1]
    return most_common_count / len(ngrams) > threshold


def filter_pair(
    pair: TrainPair,
    min_prompt_words: int = 2,
    max_prompt_words: int = 512,
    min_response_words: int = 3,
    max_response_words: int = 1024,
    stats: PipelineStats | None = None,
) -> bool:
    """Return True if the pair passes all quality filters."""
    if not pair.prompt or not pair.response:
        if stats:
            stats.skipped_empty += 1
        return False

    pw = _word_count(pair.prompt)
    rw = _word_count(pair.response)

    if pw < min_prompt_words or rw < min_response_words:
        if stats:
            stats.skipped_too_short += 1
        return False

    if pw > max_prompt_words or rw > max_response_words:
        if stats:
            stats.skipped_too_long += 1
        return False

    if _is_repetitive(pair.response):
        if stats:
            stats.skipped_too_short += 1
        return False

    return True


# ---------------------------------------------------------------------------
# Stage 5: Deduplicate
# ---------------------------------------------------------------------------


def _fingerprint(text: str) -> str:
    """Quick exact-dedup fingerprint: lowercased first 200 chars."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())[:200]
    return hashlib.md5(normalized.encode()).hexdigest()


def _simhash(text: str, bits: int = 64) -> int:
    """Trivial simhash over word unigrams for near-dup detection."""
    words = text.lower().split()[:200]
    vector = [0] * bits
    for w in words:
        h = int(hashlib.sha1(w.encode()).hexdigest(), 16)
        for i in range(bits):
            if (h >> i) & 1:
                vector[i] += 1
            else:
                vector[i] -= 1
    return sum(1 << i for i in range(bits) if vector[i] > 0)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate(
    pairs: list[TrainPair],
    exact: bool = True,
    near_dup: bool = True,
    near_threshold: int = 6,
    stats: PipelineStats | None = None,
) -> list[TrainPair]:
    """
    Remove duplicate and near-duplicate pairs.

    Args:
        pairs: Input pairs.
        exact: Perform exact deduplication (MD5 fingerprint on prompt+response).
        near_dup: Perform near-duplicate removal via simhash on response.
        near_threshold: Max hamming distance to classify as near-duplicate.
        stats: Optional stats counter.

    Returns:
        Deduplicated list.
    """
    seen_exact: set[str] = set()
    simhashes: list[int] = []
    out: list[TrainPair] = []
    removed = 0

    for pair in pairs:
        key = _fingerprint(pair.prompt + " " + pair.response)
        if exact and key in seen_exact:
            removed += 1
            continue
        seen_exact.add(key)

        if near_dup:
            sh = _simhash(pair.response)
            is_near = any(_hamming(sh, other) <= near_threshold for other in simhashes)
            if is_near:
                removed += 1
                continue
            simhashes.append(sh)

        out.append(pair)

    if stats:
        stats.skipped_dedup = removed
    return out


# ---------------------------------------------------------------------------
# Stage 6 (optional): Light augmentation
# ---------------------------------------------------------------------------

_AUGMENT_TEMPLATES = [
    "{prompt}",
    "Please answer: {prompt}",
    "Question: {prompt}",
    "Can you explain: {prompt}",
    "Tell me about: {prompt}",
    "Help me understand: {prompt}",
]


def augment_pairs(
    pairs: list[TrainPair],
    factor: float = 0.2,
    seed: int = 42,
) -> list[TrainPair]:
    """
    Light prompt-template augmentation — adds ``factor * len(pairs)`` new items.
    Randomly rephrases a subset of prompts using known templates.
    """
    rng = random.Random(seed)
    n_aug = int(len(pairs) * factor)
    if n_aug == 0:
        return pairs

    augmented: list[TrainPair] = list(pairs)
    candidates = rng.sample(pairs, min(n_aug, len(pairs)))
    for pair in candidates:
        tmpl = rng.choice(_AUGMENT_TEMPLATES[1:])  # skip identity
        new_prompt = tmpl.format(prompt=pair.prompt)
        augmented.append(
            TrainPair(
                prompt=new_prompt,
                response=pair.response,
                source=pair.source,
                modality=pair.modality,
                quality_score=pair.quality_score * 0.9,
            )
        )
    return augmented


# ---------------------------------------------------------------------------
# Stage 8: Train / Val Split
# ---------------------------------------------------------------------------


def split_train_val(
    pairs: list[TrainPair],
    val_fraction: float = 0.05,
    seed: int = 42,
    min_val: int = 50,
    max_val: int = 2000,
) -> tuple[list[TrainPair], list[TrainPair]]:
    """
    Shuffle and split pairs into train and validation sets.

    Args:
        pairs: All processed pairs.
        val_fraction: Fraction held out for validation.
        seed: Reproducible shuffle seed.
        min_val: Minimum validation examples (if dataset large enough).
        max_val: Cap on validation set size.

    Returns:
        ``(train_pairs, val_pairs)``
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_val = min(
        max_val,
        max(0 if len(pairs) < min_val * 2 else min_val, int(len(pairs) * val_fraction)),
    )
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# High-level pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    source: Path,
    *,
    max_total_pairs: int = 50_000,
    max_pairs_per_file: int = 10_000,
    min_prompt_words: int = 2,
    max_prompt_words: int = 512,
    min_response_words: int = 3,
    max_response_words: int = 1024,
    dedup_exact: bool = True,
    dedup_near: bool = True,
    near_threshold: int = 6,
    augment: bool = False,
    augment_factor: float = 0.15,
    val_fraction: float = 0.05,
    seed: int = 42,
    skip_exts: set[str] | None = None,
    verbose: bool = True,
) -> tuple[list[TrainPair], list[TrainPair], PipelineStats]:
    """
    Full data pipeline: ingest → normalize → filter → dedup → split.

    Args:
        source: Path to a file or directory.
        max_total_pairs: Hard cap on total pairs after all stages.
        max_pairs_per_file: Cap pairs per file to prevent one file dominating.
        min/max_prompt_words: Word-count filter bounds for prompts.
        min/max_response_words: Word-count filter bounds for responses.
        dedup_exact: Exact deduplication via MD5 fingerprint.
        dedup_near: Near-duplicate removal via simhash.
        near_threshold: Hamming distance threshold for near-dup (lower=stricter).
        augment: Lightly augment prompts.
        augment_factor: Fraction of pairs to augment.
        val_fraction: Fraction to hold out for validation.
        seed: Reproducibility seed.
        skip_exts: File extensions to skip (e.g. ``{".wav"}``).
        verbose: Print pipeline report to stdout.

    Returns:
        ``(train_pairs, val_pairs, stats)``
    """
    # 1+2. Ingest
    if source.is_dir():
        raw, stats = ingest_directory(
            source,
            max_pairs_per_file=max_pairs_per_file,
            skip_exts=skip_exts,
        )
    else:
        raw, stats = ingest_file(source, max_pairs=max_pairs_per_file)

    if verbose:
        logger.info(f"Ingested {stats.raw_pairs:,} raw pairs from {stats.raw_files} files")

    # 3. Normalize
    normed = [normalize_pair(p) for p in raw]
    stats.after_normalize = len(normed)

    # 4. Filter
    filtered = [p for p in normed if filter_pair(
        p,
        min_prompt_words=min_prompt_words,
        max_prompt_words=max_prompt_words,
        min_response_words=min_response_words,
        max_response_words=max_response_words,
        stats=stats,
    )]
    stats.after_filter = len(filtered)
    if verbose:
        logger.info(f"After filter: {stats.after_filter:,} pairs")

    # 5. Deduplicate
    deduped = deduplicate(
        filtered,
        exact=dedup_exact,
        near_dup=dedup_near,
        near_threshold=near_threshold,
        stats=stats,
    )
    stats.after_dedup = len(deduped)
    if verbose:
        logger.info(f"After dedup:  {stats.after_dedup:,} pairs (removed {stats.skipped_dedup:,} dupes)")

    # Hard cap
    if max_total_pairs > 0 and len(deduped) > max_total_pairs:
        rng = random.Random(seed)
        deduped = rng.sample(deduped, max_total_pairs)

    # 6. (Optional) Augment
    if augment and deduped:
        deduped = augment_pairs(deduped, factor=augment_factor, seed=seed)

    # 7. Split
    train_pairs, val_pairs = split_train_val(
        deduped,
        val_fraction=val_fraction,
        seed=seed,
    )
    stats.train_pairs = len(train_pairs)
    stats.val_pairs = len(val_pairs)

    if verbose:
        print(stats.report())

    return train_pairs, val_pairs, stats


__all__ = [
    "TrainPair",
    "PipelineStats",
    "ingest_path",
    "ingest_directory",
    "ingest_file",
    "normalize_text",
    "normalize_pair",
    "filter_pair",
    "deduplicate",
    "augment_pairs",
    "split_train_val",
    "run_pipeline",
]
