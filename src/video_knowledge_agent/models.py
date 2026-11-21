"""Core domain models shared by ingest, extraction, and retrieval stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def format_timestamp(seconds: float) -> str:
    """Format a non-negative number of seconds as HH:MM:SS."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_url(source_url: str, seconds: float) -> str:
    """Return a source URL that opens near the requested timestamp."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    parts = urlsplit(source_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = str(int(seconds))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@dataclass(frozen=True, slots=True)
class VideoSource:
    id: str
    source_url: str
    platform: str
    title: str = ""
    author: str = ""
    published_at: str | None = None
    version: str | None = None
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    topic: str = ""
    entities: tuple[str, ...] = field(default_factory=tuple)
    knowledge_type: str = "statement"

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if not self.text.strip():
            raise ValueError("text must not be empty")

    def citation(self, source_url: str) -> str:
        label = format_timestamp(self.start_seconds)
        return f"[{label}]({timestamp_url(source_url, self.start_seconds)})"
