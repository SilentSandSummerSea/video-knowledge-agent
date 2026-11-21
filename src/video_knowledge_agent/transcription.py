"""Timestamp-preserving audio transcription with faster-whisper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import format_timestamp, timestamp_url


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be transcribed."""


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: int
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True, slots=True)
class TranscriptArtifact:
    audio_path: Path
    json_path: Path
    markdown_path: Path
    language: str
    language_probability: float
    duration_seconds: float
    segment_count: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("audio_path", "json_path", "markdown_path"):
            result[key] = str(result[key])
        return result


def _load_model_factory() -> Callable[..., Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed; install the media dependencies with "
            "'python -m pip install -e .[media]'"
        ) from exc
    return WhisperModel


def _source_metadata(audio_path: Path) -> dict[str, Any]:
    info_path = audio_path.with_name("source.info.json")
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _make_segments(raw_segments: Iterable[Any]) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments, start=1):
        text = str(segment.text).strip()
        if not text:
            continue
        result.append(
            TranscriptSegment(
                id=index,
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=text,
            )
        )
    return result


def _write_outputs(
    *,
    output_dir: Path,
    audio_path: Path,
    source_url: str,
    title: str,
    model_name: str,
    language: str,
    language_probability: float,
    duration_seconds: float,
    segments: list[TranscriptSegment],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "transcript.json"
    markdown_path = output_dir / "transcript.md"

    payload = {
        "source_url": source_url,
        "title": title,
        "audio_path": str(audio_path),
        "model": model_name,
        "language": language,
        "language_probability": language_probability,
        "duration_seconds": duration_seconds,
        "segments": [asdict(segment) for segment in segments],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [f"# {title or audio_path.stem} — Transcript", ""]
    lines.append(f"- Language: `{language}`")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Duration: `{format_timestamp(duration_seconds)}`")
    lines.append("")
    for segment in segments:
        label = format_timestamp(segment.start_seconds)
        if source_url:
            label = f"[{label}]({timestamp_url(source_url, segment.start_seconds)})"
        lines.append(f"- {label} {segment.text}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def transcribe_audio(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "small",
    language: str | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
    model_dir: str | Path = ".models",
    model_factory: Callable[..., Any] | None = None,
) -> TranscriptArtifact:
    """Transcribe an audio file and emit timestamped JSON and Markdown."""
    audio_path = Path(audio_file).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    factory = model_factory or _load_model_factory()
    try:
        model = factory(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(Path(model_dir).resolve()),
        )
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        segments = _make_segments(raw_segments)
    except Exception as exc:
        raise TranscriptionError(f"transcription failed: {exc}") from exc

    metadata = _source_metadata(audio_path)
    detected_language = str(getattr(info, "language", None) or language or "unknown")
    language_probability = float(getattr(info, "language_probability", 0.0))
    duration = float(getattr(info, "duration", 0.0))
    destination = Path(output_dir).resolve()
    json_path, markdown_path = _write_outputs(
        output_dir=destination,
        audio_path=audio_path,
        source_url=str(metadata.get("webpage_url") or metadata.get("original_url") or ""),
        title=str(metadata.get("title") or ""),
        model_name=model_name,
        language=detected_language,
        language_probability=language_probability,
        duration_seconds=duration,
        segments=segments,
    )
    return TranscriptArtifact(
        audio_path=audio_path,
        json_path=json_path,
        markdown_path=markdown_path,
        language=detected_language,
        language_probability=language_probability,
        duration_seconds=duration,
        segment_count=len(segments),
    )
