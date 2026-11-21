"""Publish clean chapter documents into a dedicated knowledge-index input tree."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PublishingError(RuntimeError):
    """Raised when chapter artifacts cannot be published safely."""


@dataclass(frozen=True, slots=True)
class PublishedKnowledge:
    knowledge_dir: Path
    chapters_dir: Path
    metadata_path: Path
    chapter_count: int
    moments_path: Path | None = None
    moment_count: int = 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishingError(f"cannot read chapter metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise PublishingError("chapter metadata must be a JSON object")
    return value


def publish_knowledge(
    chapters_file: str | Path,
    sections_dir: str | Path,
    publish_root: str | Path,
    video_id: str,
    structured_knowledge_file: str | Path | None = None,
    moments_file: str | Path | None = None,
) -> PublishedKnowledge:
    """Publish only index-ready chapter Markdown and its metadata."""
    chapters_path = Path(chapters_file).resolve()
    source_sections = Path(sections_dir).resolve()
    if not source_sections.is_dir():
        raise PublishingError(f"chapter sections directory does not exist: {source_sections}")
    payload = _read_json(chapters_path)
    raw_chapters = payload.get("chapters")
    if not isinstance(raw_chapters, list) or not raw_chapters:
        raise PublishingError("chapter metadata contains no chapters")

    knowledge_dir = Path(publish_root).resolve() / video_id
    destination = knowledge_dir / "chapters"
    destination.mkdir(parents=True, exist_ok=True)
    source_files = sorted(source_sections.glob("*.md"))
    if len(source_files) != len(raw_chapters):
        raise PublishingError(
            f"chapter file count mismatch: files={len(source_files)}, metadata={len(raw_chapters)}"
        )

    metadata_path = knowledge_dir / "metadata.json"
    structured_target = knowledge_dir / "structured_knowledge.json"
    moments_target = knowledge_dir / "moments.json"
    moment_count = 0
    if structured_knowledge_file is not None:
        structured_source = Path(structured_knowledge_file).resolve()
        if not structured_source.is_file():
            raise PublishingError(
                f"structured knowledge file does not exist: {structured_source}"
            )
        shutil.copy2(structured_source, structured_target)
    if moments_file is not None:
        moments_source = Path(moments_file).resolve()
        moments_payload = _read_json(moments_source)
        if moments_payload.get("video_id") != video_id or not isinstance(moments_payload.get("moments"), list):
            raise PublishingError("moment file does not match the video")
        moment_count = len(moments_payload["moments"])
        if moments_source != moments_target.resolve():
            shutil.copy2(moments_source, moments_target)
    previous_files: set[str] = set()
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            previous_files = {
                str(item["file"])
                for item in previous.get("chapters", [])
                if isinstance(item, dict) and item.get("file")
            }
        except (OSError, json.JSONDecodeError, TypeError):
            previous_files = set()

    published_files: set[str] = set()
    chapter_records: list[dict[str, Any]] = []
    for item, source_file in zip(raw_chapters, source_files, strict=True):
        target = destination / source_file.name
        shutil.copy2(source_file, target)
        relative = target.relative_to(knowledge_dir).as_posix()
        published_files.add(relative)
        chapter_records.append(
            {
                "id": int(item["id"]),
                "title": str(item["title"]),
                "file": relative,
                "start_seconds": float(item["start_seconds"]),
                "end_seconds": float(item["end_seconds"]),
                "topics": list(item.get("topics", [])),
            }
        )

    for stale_relative in previous_files - published_files:
        stale = (knowledge_dir / stale_relative).resolve()
        if stale.parent == destination and stale.suffix.lower() == ".md" and stale.is_file():
            stale.unlink()

    metadata = {
        "schema_version": 1,
        "video_id": video_id,
        "title": str(payload.get("title") or ""),
        "source_url": str(payload.get("source_url") or ""),
        "source_language": str(payload.get("source_language") or "unknown"),
        "output_language": str(
            payload.get("output_language")
            or payload.get("source_language")
            or "unknown"
        ),
        "chapter_count": len(chapter_records),
        "structured_knowledge_file": (
            structured_target.name if structured_knowledge_file is not None else None
        ),
        "moments_file": moments_target.name if moments_file is not None else None,
        "moment_count": moment_count,
        "chapters": chapter_records,
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(metadata_path)
    return PublishedKnowledge(
        knowledge_dir=knowledge_dir,
        chapters_dir=destination,
        metadata_path=metadata_path,
        chapter_count=len(chapter_records),
        moments_path=moments_target if moments_file is not None else None,
        moment_count=moment_count,
    )


def publish_moments(moments_file: str | Path, knowledge_dir: str | Path) -> Path:
    """Add moments to an already published video without rewriting chapters."""
    directory = Path(knowledge_dir).resolve()
    metadata_path = directory / "metadata.json"
    metadata = _read_json(metadata_path)
    source = Path(moments_file).resolve()
    moments = _read_json(source)
    if moments.get("video_id") != metadata.get("video_id") or not isinstance(moments.get("moments"), list):
        raise PublishingError("moment file does not match published video")
    target = directory / "moments.json"
    if source != target.resolve():
        shutil.copy2(source, target)
    metadata["moments_file"] = target.name
    metadata["moment_count"] = len(moments["moments"])
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return target
