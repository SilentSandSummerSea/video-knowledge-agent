"""Deterministic, chapter-bounded time windows for video retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import timestamp_url


class MomentError(RuntimeError):
    """Raised when timestamped source data cannot form complete moments."""


@dataclass(frozen=True, slots=True)
class MomentArtifact:
    json_path: Path
    video_id: str
    chapter_count: int
    moment_count: int
    segment_count: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["json_path"] = str(self.json_path)
        return result


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MomentError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MomentError(f"expected JSON object in {path}")
    return value


def _source_hash(chapters_path: Path, transcript_path: Path, max_seconds: float, max_gap_seconds: float) -> str:
    digest = hashlib.sha256()
    for path in (chapters_path, transcript_path):
        digest.update(path.read_bytes())
    digest.update(f"v1:{max_seconds}:{max_gap_seconds}".encode("ascii"))
    return digest.hexdigest()


def build_moments(
    chapters_file: str | Path,
    output_dir: str | Path,
    video_id: str,
    *,
    max_seconds: float = 45.0,
    max_gap_seconds: float = 8.0,
) -> MomentArtifact:
    """Group every source transcript segment once, without crossing chapters."""
    if max_seconds <= 0 or max_gap_seconds < 0:
        raise ValueError("moment duration must be positive and gap non-negative")
    chapters_path = Path(chapters_file).resolve()
    chapters_payload = _read_object(chapters_path)
    transcript_value = chapters_payload.get("input_transcript")
    if not transcript_value:
        raise MomentError("chapter data has no input transcript path")
    transcript_path = Path(str(transcript_value)).resolve()
    transcript_payload = _read_object(transcript_path)
    raw_segments = transcript_payload.get("segments")
    chapters = chapters_payload.get("chapters")
    if not isinstance(raw_segments, list) or not raw_segments or not isinstance(chapters, list) or not chapters:
        raise MomentError("chapters or transcript segments are missing")

    destination = Path(output_dir).resolve()
    output_path = destination / "moments.json"
    fingerprint = _source_hash(chapters_path, transcript_path, max_seconds, max_gap_seconds)
    if output_path.is_file():
        try:
            cached = _read_object(output_path)
            if cached.get("source_hash") == fingerprint and cached.get("video_id") == video_id:
                return MomentArtifact(
                    output_path, video_id, len(chapters), len(cached["moments"]),
                    len(raw_segments), True,
                )
        except (MomentError, KeyError, TypeError):
            pass

    by_id: dict[int, dict[str, Any]] = {}
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise MomentError("invalid transcript segment")
        try:
            segment_id = int(raw["id"])
            start = float(raw["start_seconds"])
            end = float(raw["end_seconds"])
        except (KeyError, ValueError, TypeError) as exc:
            raise MomentError(f"invalid transcript segment fields: {exc}") from exc
        text = str(raw.get("corrected_text") or raw.get("text") or raw.get("asr_text") or "").strip()
        if segment_id in by_id or start < 0 or end <= start or not text:
            raise MomentError(f"invalid or duplicate transcript segment {segment_id}")
        by_id[segment_id] = {
            "id": segment_id,
            "start_seconds": start,
            "end_seconds": end,
            "text": text,
            "needs_review": bool(raw.get("needs_review", False)),
        }

    source_url = str(chapters_payload.get("source_url") or transcript_payload.get("source_url") or "")
    if not source_url:
        raise MomentError("source URL is missing")
    moments: list[dict[str, Any]] = []
    assigned: set[int] = set()

    def append_moment(chapter: dict[str, Any], items: list[dict[str, Any]]) -> None:
        if not items:
            return
        first, last = items[0], items[-1]
        chapter_id = int(chapter["id"])
        number = sum(1 for item in moments if item["chapter_id"] == chapter_id) + 1
        moments.append({
            "id": f"{video_id}:c{chapter_id:02d}:m{number:03d}",
            "chapter_id": chapter_id,
            "chapter_title": str(chapter["title"]),
            "start_segment_id": first["id"],
            "end_segment_id": last["id"],
            "segment_ids": [item["id"] for item in items],
            "start_seconds": first["start_seconds"],
            "end_seconds": max(item["end_seconds"] for item in items),
            "text": " ".join(item["text"] for item in items),
            "needs_review": any(item["needs_review"] for item in items),
            "source_url": timestamp_url(source_url, first["start_seconds"]),
        })

    for chapter in chapters:
        first_id = int(chapter["start_segment_id"])
        last_id = int(chapter["end_segment_id"])
        if last_id < first_id:
            raise MomentError(f"reversed chapter {chapter['id']} segment bounds")
        group: list[dict[str, Any]] = []
        for segment_id in range(first_id, last_id + 1):
            if segment_id not in by_id or segment_id in assigned:
                raise MomentError(f"missing or repeated segment {segment_id} in chapters")
            item = by_id[segment_id]
            assigned.add(segment_id)
            if group and (
                item["end_seconds"] - group[0]["start_seconds"] > max_seconds
                or item["start_seconds"] - group[-1]["end_seconds"] > max_gap_seconds
            ):
                append_moment(chapter, group)
                group = []
            group.append(item)
        append_moment(chapter, group)
    if assigned != set(by_id):
        raise MomentError(f"chapters omit {len(set(by_id) - assigned)} transcript segments")

    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "source_url": source_url,
        "title": str(chapters_payload.get("title") or transcript_payload.get("title") or ""),
        "source_language": str(chapters_payload.get("source_language") or transcript_payload.get("language") or "unknown"),
        "input_transcript": str(transcript_path),
        "source_hash": fingerprint,
        "max_seconds": max_seconds,
        "max_gap_seconds": max_gap_seconds,
        "segment_count": len(by_id),
        "chapter_count": len(chapters),
        "moment_count": len(moments),
        "moments": moments,
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return MomentArtifact(output_path, video_id, len(chapters), len(moments), len(by_id), False)


def backfill_moments(
    work_root: str | Path,
    knowledge_root: str | Path,
    database_path: str | Path,
) -> dict[str, Any]:
    """Locally add moments to already published videos, then rebuild the index."""
    from .indexing import build_index
    from .publishing import PublishingError, publish_moments

    work = Path(work_root).resolve()
    knowledge = Path(knowledge_root).resolve()
    processed = 0
    reused = 0
    skipped: list[dict[str, str]] = []
    moment_count = 0
    for metadata_path in sorted(knowledge.glob("*/metadata.json")):
        video_id = metadata_path.parent.name
        chapters_path = work / video_id / "chapters" / "chapters.json"
        if not chapters_path.is_file():
            skipped.append({"video_id": video_id, "reason": "chapter data missing"})
            continue
        try:
            artifact = build_moments(chapters_path, work / video_id / "moments", video_id)
            publish_moments(artifact.json_path, metadata_path.parent)
        except (MomentError, PublishingError, OSError, ValueError) as exc:
            skipped.append({"video_id": video_id, "reason": str(exc)})
            continue
        processed += 1
        reused += int(artifact.reused)
        moment_count += artifact.moment_count
    indexed = build_index(knowledge, database_path) if processed else None
    return {
        "processed_videos": processed,
        "reused_moment_files": reused,
        "published_moments": moment_count,
        "skipped": skipped,
        "index": indexed.to_dict() if indexed else None,
    }
