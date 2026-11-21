"""Prepare audio from a media file already present on the user's machine.

This module deliberately has no URL resolver or network downloader. An optional
external acquisition tool can hand its downloaded file to this boundary.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


class LocalMediaError(RuntimeError):
    """A local media file cannot be read or its audio cannot be extracted."""


@dataclass(frozen=True, slots=True)
class LocalMediaArtifact:
    video_id: str
    source_url: str
    title: str
    author: str
    duration_seconds: float
    file_path: Path
    source_path: Path
    video_path: Path | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["file_path"] = str(self.file_path)
        result["source_path"] = str(self.source_path)
        result["video_path"] = str(self.video_path) if self.video_path else None
        return result


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    source = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _ffmpeg_executable(explicit: str | Path | None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    else:
        found = shutil.which("ffmpeg")
        if not found:
            try:
                import imageio_ffmpeg

                found = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, OSError):
                found = None
        if not found:
            raise LocalMediaError("FFmpeg is required; install .[media] or pass --ffmpeg-path")
        candidate = Path(found).resolve()
    if not candidate.is_file():
        raise LocalMediaError(f"FFmpeg executable does not exist: {candidate}")
    return candidate


def _validate_audio(path: Path) -> float:
    if not path.is_file() or path.stat().st_size == 0:
        raise LocalMediaError("extracted audio is missing or empty")
    try:
        import av
    except ImportError as exc:
        raise LocalMediaError("audio validation requires PyAV; install .[groq]") from exc
    try:
        with av.open(str(path)) as container:
            if not any(stream.type == "audio" for stream in container.streams):
                raise LocalMediaError("extracted file has no audio stream")
            duration = sum(
                frame.samples / frame.sample_rate
                for frame in container.decode(audio=0)
                if frame.sample_rate
            )
    except LocalMediaError:
        raise
    except Exception as exc:
        raise LocalMediaError(f"cannot decode extracted audio: {exc}") from exc
    if duration <= 0:
        raise LocalMediaError("extracted audio has no decodable duration")
    return duration


def _has_video_stream(path: Path) -> bool:
    try:
        import av
    except ImportError as exc:
        raise LocalMediaError("media inspection requires PyAV; install .[media]") from exc
    try:
        with av.open(str(path)) as container:
            return any(stream.type == "video" for stream in container.streams)
    except Exception as exc:
        raise LocalMediaError(f"cannot inspect local media streams: {exc}") from exc


def _source_details(path: Path) -> dict[str, object]:
    """Read optional provenance beside a local file without fetching a URL."""
    candidates = [path.with_suffix(".info.json"), path.parent / "acquisition.json"]
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def prepare_local_audio(
    media_file: str | Path,
    output_dir: str | Path = "data/raw",
    *,
    ffmpeg_path: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    validator: Callable[[Path], float] = _validate_audio,
    stream_probe: Callable[[Path], bool] = _has_video_stream,
) -> LocalMediaArtifact:
    """Extract mono 16 kHz FLAC from an existing local video or audio file."""
    source_path = Path(media_file).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ValueError(f"local media file does not exist or is empty: {source_path}")
    fingerprint = _fingerprint(source_path)
    video_id = f"local-{fingerprint[:16]}"
    details = _source_details(source_path)
    destination = Path(output_dir).resolve() / video_id
    audio_path = destination / "source.flac"
    metadata_path = destination / "source.info.json"
    if audio_path.is_file() and metadata_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached.get("source_fingerprint") == fingerprint:
                duration = validator(audio_path)
                has_video = cached.get("has_video")
                if not isinstance(has_video, bool):
                    has_video = stream_probe(source_path)
                return LocalMediaArtifact(
                    video_id, str(cached.get("webpage_url") or f"local-{'video' if has_video else 'audio'}://{video_id}"),
                    str(cached.get("title") or source_path.stem),
                    str(cached.get("uploader") or cached.get("channel") or ""),
                    duration, audio_path,
                    source_path, source_path if has_video else None,
                )
        except (OSError, ValueError, LocalMediaError, json.JSONDecodeError):
            pass

    has_video = stream_probe(source_path)
    source_url = str(details.get("webpage_url") or details.get("source_url") or "")
    if not source_url.startswith(("https://", "http://")):
        source_url = f"local-{'video' if has_video else 'audio'}://{video_id}"
    title = str(details.get("title") or source_path.stem)
    author = str(details.get("uploader") or details.get("channel") or "")
    ffmpeg = _ffmpeg_executable(ffmpeg_path)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_audio = destination / "source.tmp.flac"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "flac", "-f", "flac", str(temporary_audio),
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise LocalMediaError(f"FFmpeg audio extraction failed: {completed.stderr.strip()}")
        duration = validator(temporary_audio)
        temporary_audio.replace(audio_path)
        metadata = {
            "id": video_id,
            "title": title,
            "uploader": author,
            "description": str(details.get("description") or ""),
            "tags": details.get("tags") if isinstance(details.get("tags"), list) else [],
            "categories": details.get("categories") if isinstance(details.get("categories"), list) else [],
            "webpage_url": source_url,
            "source_fingerprint": fingerprint,
            "local_media_path": str(source_path),
            "local_video_path": str(source_path) if has_video else None,
            "has_video": has_video,
            "duration": duration,
        }
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_metadata.replace(metadata_path)
    finally:
        if temporary_audio.is_file():
            temporary_audio.unlink()
    return LocalMediaArtifact(
        video_id, source_url, title, author, duration,
        audio_path, source_path, source_path if has_video else None,
    )
