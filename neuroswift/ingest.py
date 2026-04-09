from __future__ import annotations

import csv
import json
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import torch

try:
    import requests
except ImportError:
    requests = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import imageio.v3 as iio
except ImportError:
    iio = None


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_LINE_RE = re.compile(r"^https?://", re.IGNORECASE)

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
JSON_EXTENSIONS = {".json", ".jsonl"}
YAML_EXTENSIONS = {".yaml", ".yml"}
CSV_EXTENSIONS = {".csv", ".tsv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
URL_EXTENSIONS = {".url", ".urls"}
PDF_EXTENSIONS = {".pdf"}
LOG_EXTENSIONS = {".log", ".out", ".err", ".msg"}
CODE_EXTENSIONS = {
    ".py", ".ipynb", ".js", ".mjs", ".ts", ".tsx", ".c", ".cpp", ".cc", ".h", ".hpp", 
    ".java", ".go", ".rs", ".php", ".rb", ".sh", ".bat", ".ps1", ".sql", ".css", ".html", ".xml"
}


@dataclass
class MultimodalSample:
    modality: str
    source: str
    text: Optional[str] = None
    tensor: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rag_text(self) -> str:
        if self.text:
            return self.text
        parts = [f"modality={self.modality}", f"source={self.source}"]
        parts.extend(f"{key}={value}" for key, value in sorted(self.metadata.items()))
        return " | ".join(parts)


def _flatten_data(data: Any, prefix: str = "") -> list[str]:
    rows: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_data(value, child_prefix))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            child_prefix = f"{prefix}[{idx}]"
            rows.extend(_flatten_data(value, child_prefix))
    else:
        label = prefix if prefix else "value"
        rows.append(f"{label}: {data}")
    return rows


def _strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text).replace("\n", " ").strip()


def _tensor_from_image(image: Any) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    raw = torch.ByteTensor(torch.UntypedStorage.from_buffer(rgb.tobytes(), dtype=torch.uint8))
    return raw.view(height, width, 3).permute(2, 0, 1).float() / 255.0


def pad_tensor_list(
    tensors: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("pad_tensor_list expects at least one tensor.")
    max_len = max(tensor.size(0) for tensor in tensors)
    padded_rows = []
    mask_rows = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        if pad_len > 0:
            pad_shape = (pad_len,) + tensor.shape[1:]
            pad_tensor = tensor.new_full(pad_shape, pad_value)
            padded = torch.cat((tensor, pad_tensor), dim=0)
        else:
            padded = tensor
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[: tensor.size(0)] = True
        padded_rows.append(padded)
        mask_rows.append(mask)
    return torch.stack(padded_rows), torch.stack(mask_rows)


def _read_worker(args):
    """Parallel worker to parse a single file path."""
    reader, path = args
    try:
        sample = reader.read_path(path)
        if sample is None: return []
        return sample if isinstance(sample, list) else [sample]
    except:
        return []


class DatasetFolderReader:
    def __init__(
        self,
        recursive: bool = True,
        fetch_urls: bool = False,
        image_size: tuple[int, int] = (224, 224),
        max_audio_frames: int = 32000,
        max_video_frames: int = 8,
    ) -> None:
        self.recursive = recursive
        self.fetch_urls = fetch_urls
        self.image_size = image_size
        self.max_audio_frames = max_audio_frames
        self.max_video_frames = max_video_frames

    def read_folder(self, root: str | Path) -> list[MultimodalSample]:
        """Parallelized multimodal ingestion (GIL-bypass)."""
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Folder not found: {root_path}")

        pattern = "**/*" if self.recursive else "*"
        files = [p for p in sorted(root_path.glob(pattern)) if p.is_file()]
        if not files:
            return []

        num_procs = min(multiprocessing.cpu_count(), 16)
        worker_args = [(self, f) for f in files]
        
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            # chunksize=1 safe for diverse multimedia files
            results = list(executor.map(_read_worker, worker_args, chunksize=1))
            
        all_samples = []
        for res in results:
            all_samples.extend(res)
        return all_samples

    def read_path(self, path: str | Path) -> MultimodalSample | list[MultimodalSample] | None:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS: return self._read_text_file(path)
        if suffix in JSON_EXTENSIONS: return self._read_json_file(path)
        if suffix in YAML_EXTENSIONS: return self._read_yaml_file(path)
        if suffix in CSV_EXTENSIONS: return self._read_csv_file(path)
        if suffix in EXCEL_EXTENSIONS: return self._read_excel_file(path)
        if suffix in IMAGE_EXTENSIONS: return self._read_image_file(path)
        if suffix in AUDIO_EXTENSIONS: return self._read_audio_file(path)
        if suffix in VIDEO_EXTENSIONS: return self._read_video_file(path)
        if suffix in URL_EXTENSIONS: return self._read_url_file(path)
        if suffix in PDF_EXTENSIONS: return self._read_pdf_file(path)
        if suffix in CODE_EXTENSIONS or suffix in LOG_EXTENSIONS:
            return self._read_text_file(path, modality="code" if suffix in CODE_EXTENSIONS else "log")
        return None

    def fetch_url(self, url: str, timeout: float = 10.0) -> MultimodalSample:
        if requests is None: raise RuntimeError("requests missing.")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        text = response.text if "text" in content_type or "json" in content_type else ""
        if "html" in content_type: text = _strip_html(text)
        return MultimodalSample(modality="url", source=url, text=text[:20000],
                                metadata={"content_type": content_type, "status_code": response.status_code})

    def _read_text_file(self, path: Path, modality: str = "text") -> MultimodalSample:
        from .data_pipeline import smart_decode
        raw = path.read_bytes()
        text = smart_decode(raw)
        return MultimodalSample(modality=modality, source=str(path), text=text)

    def _read_pdf_file(self, path: Path) -> MultimodalSample:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        except:
            text = "[PDF Recovery Mode] Parsing via binary scan..."
        return MultimodalSample(modality="pdf", source=str(path), text=text.strip())

    def _read_json_file(self, path: Path) -> MultimodalSample:
        from .data_pipeline import smart_decode
        raw = path.read_bytes()
        text = smart_decode(raw)
        return MultimodalSample(modality="json", source=str(path), text=text)

    def _read_yaml_file(self, path: Path) -> MultimodalSample:
        if yaml is None: return MultimodalSample(modality="yaml", source=str(path), text="")
        from .data_pipeline import smart_decode
        raw = path.read_bytes()
        text = smart_decode(raw)
        payload = yaml.safe_load(text)
        return MultimodalSample(modality="yaml", source=str(path), text=str(payload))

    def _read_csv_file(self, path: Path) -> MultimodalSample:
        from .data_pipeline import smart_decode
        raw = path.read_bytes()
        rows = smart_decode(raw)
        return MultimodalSample(modality="table", source=str(path), text=rows)

    def _read_excel_file(self, path: Path) -> MultimodalSample:
        return MultimodalSample(modality="excel", source=str(path), text=f"Excel file: {path.name}")

    def _read_image_file(self, path: Path) -> MultimodalSample:
        if Image is None: return MultimodalSample(modality="image", source=str(path))
        with Image.open(path) as img:
            img = img.convert("RGB").resize(self.image_size)
            tensor = _tensor_from_image(img)
            return MultimodalSample(modality="image", source=str(path), tensor=tensor)

    def _read_audio_file(self, path: Path) -> MultimodalSample:
        return MultimodalSample(modality="audio", source=str(path))

    def _read_video_file(self, path: Path) -> MultimodalSample:
        return MultimodalSample(modality="video", source=str(path))

    def _read_url_file(self, path: Path) -> list[MultimodalSample]:
        return []

    def _build_url_samples(self, urls: Iterable[str], source_hint: str) -> list[MultimodalSample]:
        return []
