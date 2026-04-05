"""
neuroswift.artifacts_io
=======================
Real artifact I/O for NeuroSwiftOmni generated outputs.

Saves generated tensors as:
  - images  → PNG  (via Pillow)
  - audio   → WAV  (via soundfile)
  - video   → PNG frames + optional MP4 (via imageio / ffmpeg)

All save functions return the Path of the saved file so callers can
report, stream, or post-process the result.
"""

from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Optional heavy deps – import lazily so the core lib stays lightweight
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PIL_Image
except ImportError:  # pragma: no cover
    _PIL_Image = None  # type: ignore[assignment]

try:
    import soundfile as _sf
except ImportError:  # pragma: no cover
    _sf = None  # type: ignore[assignment]

try:
    import imageio.v3 as _iio
    import numpy as _np
except ImportError:  # pragma: no cover
    _iio = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Default directories
# ---------------------------------------------------------------------------
_DEFAULT_ARTIFACT_ROOT = Path("artifacts") / "generated"

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OmniArtifact:
    """Holds paths of all saved artifacts from a single generation call."""

    prompt: str
    modality: str
    image_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    video_dir: Optional[Path] = None
    video_mp4_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    extra: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"prompt: {self.prompt!r}", f"modality: {self.modality}"]
        if self.image_path:
            parts.append(f"image → {self.image_path}")
        if self.audio_path:
            parts.append(f"audio → {self.audio_path}")
        if self.video_mp4_path:
            parts.append(f"video → {self.video_mp4_path}")
        elif self.video_dir:
            parts.append(f"video frames → {self.video_dir}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clamp_uint8(tensor: Tensor) -> "bytes":
    """Convert a [C, H, W] float32 tensor in [0, 1] to bytes."""
    arr = (tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).byte()
    return arr.permute(1, 2, 0).numpy().tobytes()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def save_image(
    image_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_image",
    idx: int = 0,
) -> Path:
    """Save a [3, H, W] float32 tensor as PNG.

    Args:
        image_tensor: Shape ``[3, H, W]``, values in ``[0, 1]``.
        out_dir: Directory to save into (created if missing).
        stem: Filename prefix.
        idx: Batch index, appended to filename.

    Returns:
        Path of the saved PNG file.

    Raises:
        RuntimeError: If Pillow is not installed.
    """
    if _PIL_Image is None:
        raise RuntimeError(
            "Pillow is required for saving images. Install with: pip install Pillow"
        )
    _ensure_dir(out_dir)

    if image_tensor.ndim == 4:
        # [B, C, H, W] – take single item
        image_tensor = image_tensor[idx]

    channels, height, width = image_tensor.shape
    if channels not in (1, 3, 4):
        raise ValueError(f"Expected 1/3/4 channel image, got shape {image_tensor.shape}")

    raw_bytes = _clamp_uint8(image_tensor if channels != 1 else image_tensor.expand(3, height, width))
    mode = "RGB" if channels >= 3 else "L"
    pil_img = _PIL_Image.frombytes(mode, (width, height), raw_bytes)

    filename = f"{stem}_{_timestamp()}_b{idx:02d}.png"
    out_path = out_dir / filename
    pil_img.save(out_path, format="PNG")
    return out_path


def save_image_batch(
    image_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_image",
) -> list[Path]:
    """Save a [B, 3, H, W] batch as individual PNGs."""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    return [
        save_image(image_tensor, out_dir=out_dir, stem=stem, idx=i)
        for i in range(image_tensor.size(0))
    ]


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------


def save_audio(
    audio_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_audio",
    sample_rate: int = 22050,
    idx: int = 0,
) -> Path:
    """Save a [samples] or [B, samples] float32 tensor as WAV.

    Args:
        audio_tensor: Shape ``[samples]`` or ``[B, samples]``, values in ``[-1, 1]``.
        out_dir: Directory to save into (created if missing).
        stem: Filename prefix.
        sample_rate: Sample rate in Hz (default 22 050).
        idx: Batch index if 2-D tensor.

    Returns:
        Path of the saved WAV file.

    Raises:
        RuntimeError: If soundfile is not installed.
    """
    if _sf is None:
        raise RuntimeError(
            "soundfile is required for saving audio. Install with: pip install soundfile"
        )
    _ensure_dir(out_dir)

    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor[idx]

    waveform = audio_tensor.detach().cpu().clamp(-1.0, 1.0).float().numpy()
    filename = f"{stem}_{_timestamp()}_b{idx:02d}.wav"
    out_path = out_dir / filename
    _sf.write(str(out_path), waveform, samplerate=sample_rate, subtype="PCM_16")
    return out_path


def save_audio_batch(
    audio_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_audio",
    sample_rate: int = 22050,
) -> list[Path]:
    """Save a [B, samples] batch as individual WAVs."""
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    return [
        save_audio(audio_tensor, out_dir=out_dir, stem=stem, sample_rate=sample_rate, idx=i)
        for i in range(audio_tensor.size(0))
    ]


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------


def save_video_frames(
    video_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_video",
    idx: int = 0,
    assemble_mp4: bool = True,
    fps: int = 8,
) -> tuple[Path, Optional[Path]]:
    """Save a [F, 3, H, W] or [B, F, 3, H, W] tensor as PNG frames + optional MP4.

    Args:
        video_tensor: Float tensor in ``[0, 1]``.
        out_dir: Directory to save into (created if missing).
        stem: Filename prefix.
        idx: Batch index if 5-D tensor.
        assemble_mp4: Try to write MP4 via imageio + ffmpeg.
        fps: Frames per second for MP4.

    Returns:
        Tuple of ``(frames_directory, mp4_path_or_None)``.
    """
    if _PIL_Image is None:
        raise RuntimeError("Pillow is required for saving video frames.")

    if video_tensor.ndim == 5:
        video_tensor = video_tensor[idx]  # [F, C, H, W]

    n_frames, channels, height, width = video_tensor.shape
    run_id = f"{stem}_{_timestamp()}_b{idx:02d}"
    frame_dir = _ensure_dir(out_dir / run_id)

    frame_paths: list[Path] = []
    for f_idx in range(n_frames):
        frame_path = save_image(
            video_tensor[f_idx],
            out_dir=frame_dir,
            stem=f"frame",
            idx=f_idx,
        )
        frame_paths.append(frame_path)

    mp4_path: Optional[Path] = None
    if assemble_mp4 and _iio is not None and _np is not None:
        try:
            mp4_path = out_dir / f"{run_id}.mp4"
            frames_np = [
                (_np.array(_PIL_Image.open(p).convert("RGB")))
                for p in frame_paths
            ]
            _iio.imwrite(
                str(mp4_path),
                frames_np,
                fps=fps,
                codec="libx264",
                quality=8,
            )
        except Exception as exc:
            warnings.warn(f"MP4 assembly failed (imageio/ffmpeg): {exc}. Frames saved as PNGs.")
            mp4_path = None

    return frame_dir, mp4_path


def save_video_batch(
    video_tensor: Tensor,
    out_dir: Path = _DEFAULT_ARTIFACT_ROOT,
    stem: str = "generated_video",
    assemble_mp4: bool = True,
    fps: int = 8,
) -> list[tuple[Path, Optional[Path]]]:
    """Save a [B, F, 3, H, W] batch."""
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)
    return [
        save_video_frames(video_tensor, out_dir=out_dir, stem=stem, idx=i, assemble_mp4=assemble_mp4, fps=fps)
        for i in range(video_tensor.size(0))
    ]


# ---------------------------------------------------------------------------
# High-level OmniArtifact saver
# ---------------------------------------------------------------------------


class ArtifactSaver:
    """Convenience wrapper that saves all modality outputs from one generation call."""

    def __init__(
        self,
        out_dir: Path | str = _DEFAULT_ARTIFACT_ROOT,
        sample_rate: int = 22050,
        video_fps: int = 8,
        assemble_mp4: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.assemble_mp4 = assemble_mp4

    def save(
        self,
        prompt: str,
        modality: str,
        *,
        image_tensor: Optional[Tensor] = None,
        audio_tensor: Optional[Tensor] = None,
        video_tensor: Optional[Tensor] = None,
        batch_idx: int = 0,
        save_metadata: bool = True,
    ) -> OmniArtifact:
        """Save whichever tensors are provided and return an OmniArtifact record."""
        artifact = OmniArtifact(prompt=prompt, modality=modality)

        if image_tensor is not None:
            try:
                artifact.image_path = save_image(
                    image_tensor, out_dir=self.out_dir / "images", idx=batch_idx
                )
            except RuntimeError as exc:
                warnings.warn(str(exc))

        if audio_tensor is not None:
            try:
                artifact.audio_path = save_audio(
                    audio_tensor,
                    out_dir=self.out_dir / "audio",
                    sample_rate=self.sample_rate,
                    idx=batch_idx,
                )
            except RuntimeError as exc:
                warnings.warn(str(exc))

        if video_tensor is not None:
            try:
                frame_dir, mp4_path = save_video_frames(
                    video_tensor,
                    out_dir=self.out_dir / "video",
                    idx=batch_idx,
                    assemble_mp4=self.assemble_mp4,
                    fps=self.video_fps,
                )
                artifact.video_dir = frame_dir
                artifact.video_mp4_path = mp4_path
            except RuntimeError as exc:
                warnings.warn(str(exc))

        if save_metadata:
            meta_dir = _ensure_dir(self.out_dir / "metadata")
            meta_payload = {
                "prompt": prompt,
                "modality": modality,
                "image_path": str(artifact.image_path) if artifact.image_path else None,
                "audio_path": str(artifact.audio_path) if artifact.audio_path else None,
                "video_dir": str(artifact.video_dir) if artifact.video_dir else None,
                "video_mp4": str(artifact.video_mp4_path) if artifact.video_mp4_path else None,
                "timestamp": _timestamp(),
            }
            meta_path = meta_dir / f"artifact_{_timestamp()}.json"
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            artifact.metadata_path = meta_path

        return artifact


__all__ = [
    "ArtifactSaver",
    "OmniArtifact",
    "save_audio",
    "save_audio_batch",
    "save_image",
    "save_image_batch",
    "save_video_frames",
    "save_video_batch",
]
