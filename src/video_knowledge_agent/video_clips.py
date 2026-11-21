"""Extract indexed visual evidence from a video already stored locally."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .audio import LocalMediaError, _ffmpeg_executable, _fingerprint
from .models import timestamp_url


class VideoClipError(RuntimeError):
    """A published Moment cannot be extracted from its local source."""


@dataclass(frozen=True, slots=True)
class VideoClipArtifact:
    video_id: str
    moment_id: str
    chapter_id: int
    source_url: str
    start_seconds: float
    end_seconds: float
    moment_start_seconds: float
    moment_end_seconds: float
    file_path: Path
    reused: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["file_path"] = str(self.file_path)
        return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoClipError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VideoClipError(f"expected JSON object in {path}")
    return payload


def _validate_clip(path: Path, expected_seconds: float) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoClipError("extracted clip is missing or empty")
    try:
        import av
    except ImportError as exc:
        raise VideoClipError("clip validation requires PyAV; install .[media]") from exc
    try:
        with av.open(str(path)) as container:
            if not any(stream.type == "video" for stream in container.streams):
                raise VideoClipError("extracted clip contains no video stream")
            duration = (
                float(container.duration / av.time_base)
                if container.duration is not None else None
            )
            if not any(True for _ in container.decode(video=0)):
                raise VideoClipError("extracted clip contains no decodable video frames")
    except VideoClipError:
        raise
    except Exception as exc:
        raise VideoClipError(f"extracted clip cannot be decoded: {exc}") from exc
    if duration is None or duration <= 0:
        raise VideoClipError("extracted clip duration is unavailable")
    if abs(duration - expected_seconds) > max(5.0, expected_seconds * 0.15):
        raise VideoClipError(f"extracted clip duration {duration:.1f}s differs from {expected_seconds:.1f}s")


def extract_moment_video(
    video_id: str,
    *,
    moment_id: str | None = None,
    at_seconds: float | None = None,
    knowledge_root: str | Path = "data/outputs/knowledge",
    work_root: str | Path = "data/work",
    output_root: str | Path = "data/outputs/clips",
    padding_seconds: float = 5.0,
    max_height: int = 720,
    ffmpeg_path: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    validator: Callable[[Path, float], None] = _validate_clip,
) -> VideoClipArtifact:
    """Extract one chapter-bounded Moment using only the ignored local source path."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("invalid video_id")
    if not 0 <= padding_seconds <= 60:
        raise ValueError("padding_seconds must be between 0 and 60")
    if not 144 <= max_height <= 2160:
        raise ValueError("max_height must be between 144 and 2160")
    directory = Path(knowledge_root).resolve() / video_id
    metadata = _read_json(directory / "metadata.json")
    if metadata.get("video_id") != video_id or metadata.get("moments_file") != "moments.json":
        raise VideoClipError("published video has no matching Moment data")
    payload = _read_json(directory / "moments.json")
    moments = payload.get("moments")
    if payload.get("video_id") != video_id or not isinstance(moments, list):
        raise VideoClipError("published Moment data does not match the video")
    if (moment_id is None) == (at_seconds is None):
        raise ValueError("provide exactly one of moment_id or at_seconds")
    if at_seconds is not None and at_seconds < 0:
        raise ValueError("at_seconds must be non-negative")
    matching = [
        item for item in moments
        if isinstance(item, dict) and (
            item.get("id") == moment_id if moment_id is not None else
            float(item["start_seconds"]) <= at_seconds < float(item["end_seconds"])
        )
    ]
    if len(matching) != 1:
        raise VideoClipError("no unique Moment matches the requested ID or timestamp")
    moment = matching[0]
    state = _read_json(Path(work_root).resolve() / video_id / "pipeline.json")
    if state.get("source_has_video") is False:
        raise VideoClipError("source is audio-only; a local video file is required for visual extraction")
    if state.get("video_id") != video_id or not state.get("requested_file"):
        raise VideoClipError("matching local source path is unavailable")
    video_path = Path(str(state["requested_file"])).resolve()
    if not video_path.is_file():
        raise VideoClipError(f"local source video is unavailable: {video_path}")
    source_url = str(metadata.get("source_url") or payload.get("source_url") or "")
    if not source_url:
        raise VideoClipError("published source reference is unavailable")
    moment_start = float(moment["start_seconds"])
    moment_end = float(moment["end_seconds"])
    if moment_start < 0 or moment_end <= moment_start:
        raise VideoClipError("invalid Moment time range")
    start = max(0.0, moment_start - padding_seconds)
    end = moment_end + padding_seconds
    audio_file = state.get("stages", {}).get("extract_audio", {}).get("audio_path")
    if audio_file:
        metadata_path = Path(str(audio_file)).resolve().parent / "source.info.json"
        if metadata_path.is_file():
            media_duration = float(_read_json(metadata_path).get("duration") or 0)
            if media_duration >= moment_end:
                end = min(end, media_duration)
    expected = end - start
    safe_id = str(moment["id"]).replace(":", "_")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_id):
        raise VideoClipError("invalid Moment ID")
    source_hash = str(payload.get("source_hash") or "unknown")
    source_fingerprint = _fingerprint(video_path)
    window = f"{round(start * 1000)}-{round(end * 1000)}-{max_height}p-{source_hash[:12]}-{source_fingerprint[:12]}"
    clip_dir = Path(output_root).resolve() / video_id / safe_id / window
    manifest_path = clip_dir / "clip.json"
    final_path = clip_dir / "clip.mp4"

    def artifact(reused: bool) -> VideoClipArtifact:
        return VideoClipArtifact(
            video_id, str(moment["id"]), int(moment["chapter_id"]),
            timestamp_url(source_url, moment_start), start, end,
            moment_start, moment_end, final_path, reused,
        )

    if manifest_path.is_file() and final_path.is_file():
        cached = _read_json(manifest_path)
        if (cached.get("source_hash") == source_hash and
                cached.get("source_fingerprint") == source_fingerprint and
                cached.get("file") == "clip.mp4"):
            try:
                validator(final_path, expected)
            except VideoClipError:
                pass
            else:
                return artifact(True)

    try:
        ffmpeg = _ffmpeg_executable(ffmpeg_path)
    except LocalMediaError as exc:
        raise VideoClipError(str(exc)) from exc
    clip_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=clip_dir) as scratch:
        temporary = Path(scratch) / "clip.mp4"
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(video_path), "-t", str(expected),
            "-vf", f"scale=-2:min(ih\\,{max_height})",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-c:a", "aac", "-movflags", "+faststart", str(temporary),
        ]
        completed = runner(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise VideoClipError(f"local clip extraction failed: {completed.stderr.strip()}")
        validator(temporary, expected)
        temporary.replace(final_path)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps({
        "schema_version": 2, "video_id": video_id, "moment_id": moment["id"],
        "source_hash": source_hash, "source_fingerprint": source_fingerprint,
        "start_seconds": start, "end_seconds": end, "file": "clip.mp4",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return artifact(False)
