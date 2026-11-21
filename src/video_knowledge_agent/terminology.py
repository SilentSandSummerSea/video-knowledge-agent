"""Build a conservative, evidence-backed terminology list for ASR and proofreading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’+.-]*|[A-Z]{2,})(?:[ \t]+(?:[A-Z][A-Za-z0-9'’+.-]*|of|the|and)){0,4}\b"
)


def _unique(values: Iterable[str], *, limit: int = 80) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n,;|/")
        key = cleaned.casefold()
        if not cleaned or len(cleaned) > 80 or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _read_media_metadata(audio_path: Path) -> dict[str, Any]:
    candidates = sorted(audio_path.parent.glob("*.info.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def extract_metadata_terms(audio_file: str | Path) -> list[str]:
    """Extract only terms whose spelling is present in source metadata."""
    metadata = _read_media_metadata(Path(audio_file).resolve())
    if not metadata:
        return []

    raw: list[str] = []
    for key in ("uploader", "channel", "creator"):
        value = metadata.get(key)
        if value:
            raw.append(str(value))
    for key in ("tags", "categories"):
        values = metadata.get(key)
        if isinstance(values, list):
            raw.extend(str(value) for value in values if value)

    evidence_text = "\n".join(
        str(metadata.get(key) or "") for key in ("title", "description")
    )
    for match in _CAPITALIZED_PHRASE.finditer(evidence_text):
        phrase = match.group(0)
        raw.append(phrase)
        words = phrase.split()
        # A possessive or long title can cause the regex to include adjacent
        # entities (for example "Epic Games' Jim Brown"). Keep the evidenced
        # two-word names as additional, not guessed, glossary candidates.
        raw.extend(" ".join(words[index : index + 2]).strip("'’") for index in range(len(words) - 1))
    return _unique(raw)


def merge_terms(*groups: Iterable[str]) -> list[str]:
    """Merge user and metadata terms while preserving first-seen spelling."""
    return _unique(value for group in groups for value in group)


def build_asr_prompt(terms: Iterable[str], *, max_chars: int = 896) -> str | None:
    """Keep prioritized whole terms within Groq's UTF-8 prompt limit."""
    selected: list[str] = []
    length = 0
    for term in terms:
        extra = len(term.encode("utf-8")) + (2 if selected else 0)
        if length + extra > max_chars:
            continue
        selected.append(term)
        length += extra
    return ", ".join(selected) or None
