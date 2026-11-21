"""Helpers for provider-compatible structured text responses."""

from __future__ import annotations

import json
from typing import Any


def loads_json_object(output_text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating Markdown fences or surrounding prose."""
    text = str(output_text).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("structured output must be a JSON object")
    return value
