"""Split audio into fixed core windows with contextual padding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioChunk:
    index: int
    path: Path | None
    core_start: float
    core_end: float
    upload_start: float
    upload_end: float


def plan_audio_chunks(
    duration_seconds: float,
    *,
    core_seconds: float = 120.0,
    context_seconds: float = 5.0,
) -> list[AudioChunk]:
    """Create 2-minute core windows padded on both sides by default."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if core_seconds <= 0:
        raise ValueError("core_seconds must be positive")
    if context_seconds < 0:
        raise ValueError("context_seconds must be non-negative")

    chunks: list[AudioChunk] = []
    core_start = 0.0
    index = 1
    while core_start < duration_seconds:
        core_end = min(core_start + core_seconds, duration_seconds)
        chunks.append(
            AudioChunk(
                index=index,
                path=None,
                core_start=core_start,
                core_end=core_end,
                upload_start=max(0.0, core_start - context_seconds),
                upload_end=min(duration_seconds, core_end + context_seconds),
            )
        )
        core_start = core_end
        index += 1
    return chunks


def _load_audio_runtime() -> tuple[Any, Any]:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "audio chunking requires PyAV and NumPy; install the Groq dependencies "
            "with 'python -m pip install -e .[groq]'"
        ) from exc
    return av, np


def _decode_mono_16k(audio_path: Path) -> tuple[Any, int]:
    av, np = _load_audio_runtime()
    sample_rate = 16_000
    blocks = []
    with av.open(str(audio_path)) as source:
        if not source.streams.audio:
            raise ValueError(f"file has no audio stream: {audio_path}")
        stream = source.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in source.decode(stream):
            for converted in resampler.resample(frame):
                blocks.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            blocks.append(converted.to_ndarray())
    if not blocks:
        raise ValueError(f"audio file decoded to no samples: {audio_path}")
    return np.concatenate(blocks, axis=1), sample_rate


def _write_flac(path: Path, samples: Any, sample_rate: int) -> None:
    av, np = _load_audio_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="flac") as destination:
        stream = destination.add_stream("flac", rate=sample_rate)
        stream.codec_context.layout = "mono"
        frame_size = 4096
        for offset in range(0, samples.shape[1], frame_size):
            block = np.ascontiguousarray(samples[:, offset : offset + frame_size])
            frame = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = offset
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                destination.mux(packet)
        for packet in stream.encode(None):
            destination.mux(packet)


def split_audio(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    core_seconds: float = 120.0,
    context_seconds: float = 5.0,
) -> list[AudioChunk]:
    """Decode once, then write padded 16 kHz mono FLAC chunks."""
    audio_path = Path(audio_file).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    samples, sample_rate = _decode_mono_16k(audio_path)
    duration = samples.shape[1] / sample_rate
    planned = plan_audio_chunks(
        duration,
        core_seconds=core_seconds,
        context_seconds=context_seconds,
    )
    destination = Path(output_dir).resolve()
    result: list[AudioChunk] = []
    for chunk in planned:
        path = destination / f"chunk-{chunk.index:04d}.flac"
        start_sample = round(chunk.upload_start * sample_rate)
        end_sample = round(chunk.upload_end * sample_rate)
        _write_flac(path, samples[:, start_sample:end_sample], sample_rate)
        result.append(replace(chunk, path=path))
    return result
