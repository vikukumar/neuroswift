from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import torch

try:
    import requests
except ImportError:  # pragma: no cover - optional
    requests = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional
    yaml = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional
    Image = None

try:
    import openpyxl
except ImportError:  # pragma: no cover - optional
    openpyxl = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional
    sf = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - optional
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
        raise RuntimeError("Pillow is required to load image tensors.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(rgb.tobytes()))
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
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {root_path}")

        pattern = "**/*" if self.recursive else "*"
        samples: list[MultimodalSample] = []
        for path in sorted(root_path.glob(pattern)):
            if not path.is_file():
                continue
            sample = self.read_path(path)
            if sample is None:
                continue
            if isinstance(sample, list):
                samples.extend(sample)
            else:
                samples.append(sample)
        return samples

    def read_path(self, path: str | Path) -> MultimodalSample | list[MultimodalSample] | None:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in TEXT_EXTENSIONS:
            return self._read_text_file(path)
        if suffix in JSON_EXTENSIONS:
            return self._read_json_file(path)
        if suffix in YAML_EXTENSIONS:
            return self._read_yaml_file(path)
        if suffix in CSV_EXTENSIONS:
            return self._read_csv_file(path)
        if suffix in EXCEL_EXTENSIONS:
            return self._read_excel_file(path)
        if suffix in IMAGE_EXTENSIONS:
            return self._read_image_file(path)
        if suffix in AUDIO_EXTENSIONS:
            return self._read_audio_file(path)
        if suffix in VIDEO_EXTENSIONS:
            return self._read_video_file(path)
        if suffix in URL_EXTENSIONS:
            return self._read_url_file(path)
        if suffix in PDF_EXTENSIONS:
            return self._read_pdf_file(path)
        if suffix in CODE_EXTENSIONS or suffix in LOG_EXTENSIONS:
            return self._read_text_file(path, modality="code" if suffix in CODE_EXTENSIONS else "log")
        return None

    def fetch_url(self, url: str, timeout: float = 10.0) -> MultimodalSample:
        if requests is None:
            raise RuntimeError("requests is required for URL ingestion.")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        text = response.text if "text" in content_type or "json" in content_type else ""
        if "html" in content_type:
            text = _strip_html(text)
        return MultimodalSample(
            modality="url",
            source=url,
            text=text[:20000],
            metadata={
                "content_type": content_type,
                "status_code": response.status_code,
                "netloc": urlparse(url).netloc,
            },
        )

    def _read_text_file(self, path: Path, modality: str = "text") -> MultimodalSample:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if modality == "text" and path.suffix.lower() == ".txt":
            url_lines = [line.strip() for line in text.splitlines() if URL_LINE_RE.match(line.strip())]
            if url_lines:
                return self._build_url_samples(url_lines, source_hint=str(path))
        return MultimodalSample(
            modality=modality,
            source=str(path),
            text=text,
            metadata={"extension": path.suffix.lower()},
        )

    def _read_pdf_file(self, path: Path) -> MultimodalSample:
        # Dual-Strategy PDF Extraction
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        except (ImportError, Exception):
            # Fallback: Binary Block Extraction (extremely robust)
            try:
                raw = path.read_bytes()
                # Basic PDF text block extraction using regex
                text_blocks = re.findall(b"(\((?:[^()]*|\([^()]*\))*\))", raw)
                text = " ".join([b.decode("latin1", "ignore")[1:-1] for b in text_blocks if len(b) > 5])
            except:
                text = f"[PDF Logic Error] File {path.name} is encrypted or corrupted."
        
        return MultimodalSample(
            modality="pdf",
            source=str(path),
            text=text.strip(),
            metadata={"filename": path.name}
        )

    def _read_json_file(self, path: Path) -> MultimodalSample:
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.extend(_flatten_data(json.loads(line)))
            text = "\n".join(rows)
        else:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            text = "\n".join(_flatten_data(payload))
        return MultimodalSample(modality="json", source=str(path), text=text, metadata={})

    def _read_yaml_file(self, path: Path) -> MultimodalSample:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML ingestion.")
        payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(_flatten_data(payload))
        return MultimodalSample(modality="yaml", source=str(path), text=text, metadata={})

    def _read_csv_file(self, path: Path) -> MultimodalSample:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                rows.append(f"row_{row_idx}: " + " | ".join(row))
        return MultimodalSample(modality="table", source=str(path), text="\n".join(rows), metadata={})

    def _read_excel_file(self, path: Path) -> MultimodalSample:
        if openpyxl is None:
            raise RuntimeError("openpyxl is required for Excel ingestion.")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows: list[str] = []
        for sheet in workbook.worksheets:
            for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
                values = [str(cell) for cell in row if cell is not None]
                if values:
                    rows.append(f"{sheet.title}.row_{row_idx}: " + " | ".join(values))
        return MultimodalSample(modality="excel", source=str(path), text="\n".join(rows), metadata={})

    def _read_image_file(self, path: Path) -> MultimodalSample:
        if Image is None:
            raise RuntimeError("Pillow is required for image ingestion.")
        with Image.open(path) as image:
            resized = image.resize(self.image_size)
            tensor = _tensor_from_image(resized)
            return MultimodalSample(
                modality="image",
                source=str(path),
                text=f"image file {path.name} with size {image.size[0]}x{image.size[1]}",
                tensor=tensor,
                metadata={
                    "width": image.size[0],
                    "height": image.size[1],
                    "mode": image.mode,
                },
            )

    def _read_audio_file(self, path: Path) -> MultimodalSample:
        if sf is None:
            return MultimodalSample(
                modality="audio",
                source=str(path),
                text=f"audio file {path.name}",
                metadata={"warning": "soundfile not installed; waveform not loaded"},
            )
        waveform, sample_rate = sf.read(path, always_2d=False)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        if waveform_tensor.ndim > 1:
            waveform_tensor = waveform_tensor.mean(dim=-1)
        waveform_tensor = waveform_tensor[: self.max_audio_frames]
        return MultimodalSample(
            modality="audio",
            source=str(path),
            text=f"audio file {path.name} sampled at {sample_rate} Hz",
            tensor=waveform_tensor,
            metadata={"sample_rate": sample_rate, "num_frames": int(waveform_tensor.numel())},
        )

    def _read_video_file(self, path: Path) -> MultimodalSample:
        if iio is None:
            return MultimodalSample(
                modality="video",
                source=str(path),
                text=f"video file {path.name}",
                metadata={"warning": "imageio not installed; frames not loaded"},
            )

        try:
            meta = iio.immeta(path)
        except Exception:
            meta = {}

        frames: list[torch.Tensor] = []
        try:
            iterator = iio.imiter(path)
            for frame_idx, frame in enumerate(iterator):
                if frame_idx >= self.max_video_frames:
                    break
                frame_tensor = torch.tensor(frame, dtype=torch.float32)
                if frame_tensor.ndim == 3:
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                frames.append(frame_tensor / 255.0)
        except Exception:
            frames = []

        tensor = torch.stack(frames) if frames else None
        frame_count = int(tensor.size(0)) if tensor is not None else 0
        return MultimodalSample(
            modality="video",
            source=str(path),
            text=f"video file {path.name} with {frame_count} sampled frames",
            tensor=tensor,
            metadata=dict(meta) if isinstance(meta, dict) else {"meta": str(meta)},
        )

    def _read_url_file(self, path: Path) -> list[MultimodalSample]:
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if URL_LINE_RE.match(line.strip())
        ]
        return self._build_url_samples(urls, source_hint=str(path))

    def _build_url_samples(self, urls: Iterable[str], source_hint: str) -> list[MultimodalSample]:
        samples: list[MultimodalSample] = []
        for url in urls:
            if self.fetch_urls:
                try:
                    samples.append(self.fetch_url(url))
                    continue
                except Exception as exc:  # pragma: no cover - network dependent
                    samples.append(
                        MultimodalSample(
                            modality="url",
                            source=url,
                            text=url,
                            metadata={"warning": str(exc), "source_hint": source_hint},
                        )
                    )
                    continue
            samples.append(
                MultimodalSample(
                    modality="url",
                    source=url,
                    text=url,
                    metadata={"source_hint": source_hint},
                )
            )
        return samples


__all__ = ["DatasetFolderReader", "MultimodalSample", "pad_tensor_list"]
