"""Language option normalization shared by the ingestion stages."""

from __future__ import annotations

import re


_ALIASES = {
    "auto": "auto",
    "automatic": "auto",
    "自动": "auto",
    "zh": "zh",
    "zh-cn": "zh",
    "chinese": "zh",
    "中文": "zh",
    "en": "en",
    "english": "en",
    "英语": "en",
    "英文": "en",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "日语": "ja",
    "日文": "ja",
}

_SOURCE_ALIASES = {"source", "same", "original", "原文", "源语言", "原语言"}


def normalize_language(value: str | None, *, allow_auto: bool = True) -> str:
    """Return an ISO-style language code or ``auto`` for language detection."""
    raw = str(value or "auto").strip().lower().replace("_", "-")
    normalized = _ALIASES.get(raw, raw)
    if normalized == "auto":
        if allow_auto:
            return normalized
        raise ValueError("output language must be an explicit language code")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", normalized):
        raise ValueError(f"invalid language code: {value!r}")
    return normalized


def normalize_output_language(value: str | None) -> str:
    """Normalize an output language code, preserving the special ``source`` mode."""
    raw = str(value or "source").strip().lower().replace("_", "-")
    if raw in _SOURCE_ALIASES or raw == "auto":
        return "source"
    return normalize_language(raw, allow_auto=False)


def resolve_output_language(value: str | None, source_language: str | None) -> str:
    """Resolve ``source`` to a detected language when one is available."""
    requested = normalize_output_language(value)
    if requested != "source":
        return requested
    try:
        return normalize_language(source_language, allow_auto=False)
    except ValueError:
        return "source"
