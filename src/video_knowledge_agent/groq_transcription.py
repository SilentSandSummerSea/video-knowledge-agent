"""Cloud transcription through Groq's hosted Whisper API."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .audio_chunking import AudioChunk, split_audio
from .languages import normalize_language
from .project_env import project_secret
from .transcription import (
    TranscriptArtifact,
    TranscriptSegment,
    TranscriptionError,
    _source_metadata,
    _write_outputs,
)


FREE_TIER_FILE_LIMIT_BYTES = 25_000_000


def _load_client(api_key: str | None = None) -> Any:
    key = project_secret("GROQ_API_KEY", api_key)
    if not key:
        raise TranscriptionError(
            "GROQ_API_KEY is not set; create a key at "
            "https://console.groq.com/keys and store it in the project .env file"
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise TranscriptionError(
            "the Groq SDK is not installed; install it with "
            "'python -m pip install -e .[groq]'"
        ) from exc
    return Groq(api_key=key)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_segments(
    raw_segments: Any,
    *,
    time_offset: float = 0.0,
    core_start: float | None = None,
    core_end: float | None = None,
) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments or [], start=1):
        text = str(_value(segment, "text", "")).strip()
        start = float(_value(segment, "start", 0.0)) + time_offset
        end = float(_value(segment, "end", start - time_offset)) + time_offset
        if not text or end <= start:
            continue
        midpoint = (start + end) / 2
        if core_start is not None and midpoint < core_start:
            continue
        if core_end is not None and midpoint >= core_end:
            continue
        result.append(
            TranscriptSegment(
                id=index,
                start_seconds=start,
                end_seconds=end,
                text=text,
            )
        )
    return result


def _remove_overlapping_exact_duplicates(
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Drop a repeated boundary segment emitted by adjacent padded chunks."""
    result: list[TranscriptSegment] = []
    for segment in sorted(
        segments,
        key=lambda item: (item.start_seconds, item.end_seconds),
    ):
        if (
            result
            and segment.text == result[-1].text
            and segment.start_seconds < result[-1].end_seconds
        ):
            continue
        result.append(segment)
    return result


def _request_transcription(
    client: Any,
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
    prompt: str | None,
) -> Any:
    request: dict[str, Any] = {
        "model": model_name,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
        "temperature": 0.0,
    }
    if language:
        request["language"] = language
    if prompt:
        request["prompt"] = prompt
    with audio_path.open("rb") as audio:
        return client.audio.transcriptions.create(file=audio, **request)


def transcribe_audio_groq(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "whisper-large-v3-turbo",
    language: str | None = "auto",
    prompt: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    enforce_free_tier_limit: bool = True,
) -> TranscriptArtifact:
    """Upload audio to Groq and emit timestamped JSON and Markdown."""
    audio_path = Path(audio_file).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    if enforce_free_tier_limit and audio_path.stat().st_size > FREE_TIER_FILE_LIMIT_BYTES:
        size_mb = audio_path.stat().st_size / 1_000_000
        raise ValueError(
            f"audio file is {size_mb:.2f} MB, above Groq's 25 MB free-tier limit; "
            "split or compress it before uploading"
        )

    requested_language = normalize_language(language)
    api_language = None if requested_language == "auto" else requested_language
    groq_client = client or _load_client(api_key)
    try:
        response = _request_transcription(
            groq_client,
            audio_path,
            model_name=model_name,
            language=api_language,
            prompt=prompt,
        )
    except Exception as exc:
        raise TranscriptionError(f"Groq transcription failed: {exc}") from exc

    segments = _normalize_segments(_value(response, "segments", []))
    if not segments:
        raise TranscriptionError("Groq returned no timestamped transcript segments")

    duration = float(_value(response, "duration", segments[-1].end_seconds))
    detected_language = str(
        _value(response, "language", api_language)
        or api_language
        or "unknown"
    )
    metadata = _source_metadata(audio_path)
    destination = Path(output_dir).resolve()
    json_path, markdown_path = _write_outputs(
        output_dir=destination,
        audio_path=audio_path,
        source_url=str(metadata.get("webpage_url") or metadata.get("original_url") or ""),
        title=str(metadata.get("title") or ""),
        model_name=f"groq/{model_name}",
        language=detected_language,
        language_probability=0.0,
        duration_seconds=duration,
        segments=segments,
    )
    return TranscriptArtifact(
        audio_path=audio_path,
        json_path=json_path,
        markdown_path=markdown_path,
        language=detected_language,
        language_probability=0.0,
        duration_seconds=duration,
        segment_count=len(segments),
    )


def transcribe_audio_groq_chunked(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "whisper-large-v3-turbo",
    language: str | None = "auto",
    prompt: str | None = None,
    core_seconds: float = 120.0,
    context_seconds: float = 5.0,
    api_key: str | None = None,
    client: Any | None = None,
    chunker: Callable[..., list[AudioChunk]] = split_audio,
) -> TranscriptArtifact:
    """Transcribe padded chunks and merge them onto the original timeline."""
    audio_path = Path(audio_file).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    requested_language = normalize_language(language)
    api_language = None if requested_language == "auto" else requested_language
    groq_client = client or _load_client(api_key)
    destination = Path(output_dir).resolve()

    try:
        chunks = chunker(
            audio_path,
            destination / "chunks",
            core_seconds=core_seconds,
            context_seconds=context_seconds,
        )
    except Exception as exc:
        raise TranscriptionError(f"audio chunking failed: {exc}") from exc

    merged: list[TranscriptSegment] = []
    detected_languages: Counter[str] = Counter()
    for chunk in chunks:
        if chunk.path is None:
            raise TranscriptionError(f"chunk {chunk.index} has no audio file")
        try:
            response = _request_transcription(
                groq_client,
                chunk.path,
                model_name=model_name,
                language=api_language,
                prompt=prompt,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Groq transcription failed for chunk {chunk.index}: {exc}"
            ) from exc
        response_language = str(
            _value(response, "language", api_language)
            or api_language
            or "unknown"
        )
        if response_language != "unknown":
            detected_languages[response_language] += 1
        merged.extend(
            _normalize_segments(
                _value(response, "segments", []),
                time_offset=chunk.upload_start,
                core_start=chunk.core_start,
                core_end=chunk.core_end,
            )
        )

    if not merged:
        raise TranscriptionError("Groq returned no timestamped transcript segments")
    merged = _remove_overlapping_exact_duplicates(merged)
    merged = [replace(segment, id=index) for index, segment in enumerate(merged, start=1)]
    detected_language = (
        detected_languages.most_common(1)[0][0]
        if detected_languages
        else api_language or "unknown"
    )
    duration = chunks[-1].core_end
    metadata = _source_metadata(audio_path)
    json_path, markdown_path = _write_outputs(
        output_dir=destination,
        audio_path=audio_path,
        source_url=str(metadata.get("webpage_url") or metadata.get("original_url") or ""),
        title=str(metadata.get("title") or ""),
        model_name=f"groq/{model_name}",
        language=detected_language,
        language_probability=0.0,
        duration_seconds=duration,
        segments=merged,
    )
    return TranscriptArtifact(
        audio_path=audio_path,
        json_path=json_path,
        markdown_path=markdown_path,
        language=detected_language,
        language_probability=0.0,
        duration_seconds=duration,
        segment_count=len(merged),
    )
