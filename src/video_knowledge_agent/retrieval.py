"""Attach nearby source transcript to timestamped knowledge search results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .indexing import SearchResult, _search_tokens, search_index
from .models import timestamp_url


_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_TRANSCRIPT_FILES = (
    "asr_review/proofread_reviewed.json",
    "proofread/proofread.json",
    "groq/transcript.json",
)


def _load_transcript(work_root: Path, video_id: str) -> tuple[list[dict[str, Any]], str | None]:
    if not _VIDEO_ID.fullmatch(video_id):
        return [], None
    for relative in _TRANSCRIPT_FILES:
        path = work_root / video_id / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_segments = payload.get("segments", [])
            if not isinstance(raw_segments, list):
                continue
            segments = []
            for item in raw_segments:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("corrected_text") or item.get("text") or item.get("asr_text") or "").strip()
                start = float(item["start_seconds"])
                end = float(item["end_seconds"])
                if text and 0 <= start < end:
                    segments.append({
                        "id": item.get("id"),
                        "start_seconds": start,
                        "end_seconds": end,
                        "text": text,
                        "needs_review": bool(item.get("needs_review", False)),
                    })
            if segments:
                segments.sort(key=lambda item: item["start_seconds"])
                return segments, relative
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return [], None


def _chapter_anchor(query: str, result: SearchResult, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    within = [
        item for item in segments
        if item["start_seconds"] < result.end_seconds
        and item["end_seconds"] > result.start_seconds
    ]
    query_tokens = {token for token in _search_tokens(query).split() if len(token) >= 2}
    if not query_tokens:
        return None
    scored = [
        (len(query_tokens.intersection(_search_tokens(item["text"]).split())), item)
        for item in within
    ]
    if not scored:
        return None
    best_score, best_item = max(scored, key=lambda pair: pair[0])
    return best_item if best_score else None


def _context_for_result(
    query: str,
    result: SearchResult,
    segments: list[dict[str, Any]],
    source_url: str,
    transcript_file: str | None,
    context_seconds: float,
) -> dict[str, Any]:
    if not segments:
        return {"precision": "index_only", "segments": [], "transcript_file": None}
    anchor = _chapter_anchor(query, result, segments) if result.kind == "chapter" else None
    if result.kind == "chapter" and anchor is None:
        return {"precision": "chapter_only", "segments": [], "transcript_file": transcript_file}
    start = anchor["start_seconds"] if anchor else result.start_seconds
    end = anchor["end_seconds"] if anchor else result.end_seconds
    nearby = [
        item for item in segments
        if item["start_seconds"] < end + context_seconds
        and item["end_seconds"] > max(0.0, start - context_seconds)
    ]
    if not nearby:
        return {"precision": "index_only", "segments": [], "transcript_file": transcript_file}
    # Preserve the exact evidence interval, then its immediate context. Very
    # long ASR segments can otherwise make the agent's evidence bundle huge.
    if len(nearby) > 18:
        center = (start + end) / 2
        nearby = sorted(
            nearby,
            key=lambda item: (
                0 if item["start_seconds"] < end and item["end_seconds"] > start else 1,
                abs((item["start_seconds"] + item["end_seconds"]) / 2 - center),
            ),
        )[:18]
        nearby.sort(key=lambda item: item["start_seconds"])
    return {
        "precision": "segment" if anchor else "moment" if result.kind == "moment" else "knowledge_evidence",
        "evidence_start_seconds": start,
        "evidence_end_seconds": end,
        "evidence_url": timestamp_url(source_url, start) if source_url else result.source_url,
        "context_start_seconds": nearby[0]["start_seconds"] if nearby else start,
        "context_end_seconds": nearby[-1]["end_seconds"] if nearby else end,
        "segments": nearby,
        "transcript_file": transcript_file,
    }


def search_with_context(
    query: str,
    database_path: str | Path,
    work_root: str | Path,
    knowledge_root: str | Path,
    *,
    limit: int = 8,
    context_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Search locally and attach source-language transcript near each hit."""
    if context_seconds < 0:
        raise ValueError("context_seconds must not be negative")
    results = search_index(query, database_path, limit=limit)
    work = Path(work_root).resolve()
    knowledge = Path(knowledge_root).resolve()
    video_cache: dict[str, tuple[list[dict[str, Any]], str | None, dict[str, Any]]] = {}
    enriched = []
    for result in results:
        if result.video_id not in video_cache:
            segments, transcript_file = _load_transcript(work, result.video_id)
            metadata: dict[str, Any] = {}
            if _VIDEO_ID.fullmatch(result.video_id):
                metadata_path = knowledge / result.video_id / "metadata.json"
                try:
                    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metadata = loaded
                except (OSError, ValueError, TypeError):
                    pass
            video_cache[result.video_id] = segments, transcript_file, metadata
        segments, transcript_file, metadata = video_cache[result.video_id]
        source_url = str(metadata.get("source_url") or result.source_url)
        item = result.to_dict()
        item["video_title"] = str(metadata.get("title") or result.video_id)
        item["context"] = _context_for_result(
            query, result, segments, source_url, transcript_file, context_seconds
        )
        enriched.append(item)
    return enriched
