"""Topic-based chapter extraction from a timestamped transcript."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .costing import add_usage, empty_usage, estimate_deepseek_flash_cost, local_text_cost, response_usage
from .languages import resolve_output_language
from .models import format_timestamp, timestamp_url
from .proofreading import _load_deepseek_client
from .structured_output import loads_json_object


class ChapteringError(RuntimeError):
    """Raised when chapters cannot be generated or validated safely."""


@dataclass(frozen=True, slots=True)
class ChapterSourceSegment:
    id: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class Chapter:
    id: int
    title: str
    start_segment_id: int
    end_segment_id: int
    start_seconds: float
    end_seconds: float
    summary: str
    topics: tuple[str, ...]
    confidence: str


@dataclass(frozen=True, slots=True)
class ChapterArtifact:
    input_path: Path
    json_path: Path
    markdown_path: Path
    sections_dir: Path
    section_paths: tuple[Path, ...]
    provider: str
    model: str
    chapter_count: int
    segment_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("input_path", "json_path", "markdown_path", "sections_dir"):
            result[key] = str(result[key])
        result["section_paths"] = [str(path) for path in self.section_paths]
        return result


CHAPTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_segment_id": {"type": "integer"},
                    "end_segment_id": {"type": "integer"},
                    "summary": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "title",
                    "start_segment_id",
                    "end_segment_id",
                    "summary",
                    "topics",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是多语言视频内容章节编辑。输入转写的语言由 source_language 指定。请按主题变化将单个视频划分为连续章节，并输出 JSON。

规则：
1. 每个输入片段必须且只能属于一个章节；章节按顺序连续覆盖全部片段，不得遗漏或重叠。
2. 只能使用输入文本中的信息，不得根据常识补充视频未表达的内容。
3. 在主题、叙事阶段或论证目标明显变化处切分，不要机械地按固定时长切分。
4. 尽量避免在句子中间切分。过短且主题相同的部分应与相邻章节合并。
5. 标题应简洁具体；summary 用一到两句话概括该章；topics 提供二到五个短语。这些字段必须使用 output_language 指定的语言，专有名词可保留原文。
6. start_segment_id 和 end_segment_id 必须来自输入。不要输出时间戳，时间戳由程序计算。
7. 一条约 30 分钟的讲解视频通常划分为 4 到 10 章，但应以实际内容结构为准。
"""


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _load_source(path: Path) -> tuple[dict[str, Any], list[ChapterSourceSegment]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChapteringError(f"cannot read transcript JSON: {exc}") from exc
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ChapteringError("input contains no transcript segments")

    segments: list[ChapterSourceSegment] = []
    try:
        for item in raw_segments:
            text = str(item.get("corrected_text") or item.get("text") or "").strip()
            if not text:
                raise ValueError(f"segment {item.get('id')} has no text")
            segments.append(
                ChapterSourceSegment(
                    id=int(item["id"]),
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                    text=text,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ChapteringError(f"invalid transcript segment: {exc}") from exc
    if len({segment.id for segment in segments}) != len(segments):
        raise ChapteringError("input contains duplicate segment IDs")
    if any(
        segments[index].start_seconds < segments[index - 1].start_seconds
        for index in range(1, len(segments))
    ):
        raise ChapteringError("input segments are not in chronological order")
    return payload, segments


def _request_chapters(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    title: str,
    source_language: str,
    output_language: str,
    segments: list[ChapterSourceSegment],
    max_output_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    transcript = [
        {
            "id": segment.id,
            "start_seconds": segment.start_seconds,
            "text": segment.text,
        }
        for segment in segments
    ]
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "video_title": title,
                        "source_language": source_language,
                        "output_language": output_language,
                        "segments": transcript,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "video_chapters",
                "strict": True,
                "schema": CHAPTER_SCHEMA,
            }
        },
    )
    output_text = str(_value(response, "output_text", "") or "")
    if not output_text:
        raise ChapteringError("text model returned no chapter output")
    try:
        chapters = list(loads_json_object(output_text)["chapters"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ChapteringError(f"invalid chapter output: {exc}") from exc

    return chapters, response_usage(response)


def _validate_chapters(
    raw_chapters: list[dict[str, Any]],
    segments: list[ChapterSourceSegment],
) -> list[Chapter]:
    if not raw_chapters:
        raise ChapteringError("DeepSeek returned no chapters")
    positions = {segment.id: index for index, segment in enumerate(segments)}
    expected_start = 0
    result: list[Chapter] = []
    for chapter_id, item in enumerate(raw_chapters, start=1):
        try:
            start_id = int(item["start_segment_id"])
            end_id = int(item["end_segment_id"])
            start_index = positions[start_id]
            end_index = positions[end_id]
        except (KeyError, TypeError, ValueError) as exc:
            raise ChapteringError(
                f"chapter {chapter_id} has an invalid segment boundary: {exc}"
            ) from exc
        if start_index != expected_start:
            raise ChapteringError(
                f"chapter {chapter_id} must start at segment {segments[expected_start].id}, "
                f"not {start_id}"
            )
        if end_index < start_index:
            raise ChapteringError(f"chapter {chapter_id} has a reversed boundary")
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        confidence = str(item.get("confidence", ""))
        topics = tuple(
            dict.fromkeys(
                str(topic).strip()
                for topic in item.get("topics", [])
                if str(topic).strip()
            )
        )
        if not title or not summary:
            raise ChapteringError(f"chapter {chapter_id} has an empty title or summary")
        if not topics:
            raise ChapteringError(f"chapter {chapter_id} has no topics")
        if confidence not in {"high", "medium", "low"}:
            raise ChapteringError(f"chapter {chapter_id} has invalid confidence")
        result.append(
            Chapter(
                id=chapter_id,
                title=title,
                start_segment_id=start_id,
                end_segment_id=end_id,
                start_seconds=segments[start_index].start_seconds,
                end_seconds=segments[end_index].end_seconds,
                summary=summary,
                topics=topics,
                confidence=confidence,
            )
        )
        expected_start = end_index + 1
    if expected_start != len(segments):
        raise ChapteringError(
            f"chapters stop at segment {segments[expected_start - 1].id}; "
            f"expected final segment {segments[-1].id}"
        )
    return result


def _write_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    source: dict[str, Any],
    provider: str,
    model: str,
    reasoning_effort: str,
    output_language: str,
    chapters: list[Chapter],
    segment_count: int,
    usage: dict[str, int],
) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "chapters.json"
    markdown_path = output_dir / "chapters.md"
    payload = {
        "source_url": source.get("source_url", ""),
        "title": source.get("title", ""),
        "input_transcript": str(input_path),
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "source_language": source.get("language", "unknown"),
        "output_language": output_language,
        "segment_count": segment_count,
        "usage": usage,
        "cost_estimate": estimate_deepseek_flash_cost(usage) if provider == "deepseek" else local_text_cost(model),
        "chapters": [asdict(chapter) for chapter in chapters],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    title = str(source.get("title") or input_path.stem)
    source_url = str(source.get("source_url") or "")
    lines = [f"# {title} — Chapters", ""]
    lines.append(f"- Model: `{model}` (`{reasoning_effort}` reasoning)")
    lines.append(f"- Output language: `{output_language}`")
    lines.append(f"- Chapters: `{len(chapters)}`")
    lines.append("")
    for chapter in chapters:
        start = format_timestamp(chapter.start_seconds)
        end = format_timestamp(chapter.end_seconds)
        time_range = f"{start}–{end}"
        if source_url:
            time_range = f"[{time_range}]({timestamp_url(source_url, chapter.start_seconds)})"
        lines.append(f"## {chapter.id}. {chapter.title}")
        lines.append("")
        lines.append(f"- Time: {time_range}")
        lines.append(f"- Topics: {', '.join(chapter.topics)}")
        lines.append(f"- Confidence: `{chapter.confidence}`")
        lines.append("")
        lines.append(chapter.summary)
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    sections_dir, section_paths = export_chapter_sections(json_path)
    return json_path, markdown_path, sections_dir, section_paths


def _safe_filename(value: str) -> str:
    safe = "".join("_" if character in '<>:"/\\|?*' else character for character in value)
    safe = " ".join(safe.split()).rstrip(" .")
    return (safe or "未命名章节")[:80]


def _paragraphs(texts: list[str], target_length: int = 1000) -> list[str]:
    paragraphs: list[str] = []
    current = ""
    for text in texts:
        candidate = current + text.strip()
        if current and len(candidate) > target_length:
            paragraphs.append(current)
            current = text.strip()
        else:
            current = candidate
    if current:
        paragraphs.append(current)
    return paragraphs


def export_chapter_sections(
    chapters_file: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Create one clean Markdown retrieval document per chapter, without API calls."""
    chapters_path = Path(chapters_file).resolve()
    try:
        chapters_payload = json.loads(chapters_path.read_text(encoding="utf-8"))
        input_path = Path(chapters_payload["input_transcript"]).resolve()
        _, segments = _load_source(input_path)
        raw_chapters = chapters_payload["chapters"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ChapteringError(f"cannot export chapter files: {exc}") from exc
    segment_by_id = {segment.id: segment for segment in segments}
    sections_dir = chapters_path.parent / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    previous_section_paths = set(sections_dir.glob("*.md"))
    section_paths: list[Path] = []
    source_url = str(chapters_payload.get("source_url") or "")
    video_title = str(chapters_payload.get("title") or input_path.stem)

    for item in raw_chapters:
        try:
            chapter_id = int(item["id"])
            title = str(item["title"]).strip()
            start_id = int(item["start_segment_id"])
            end_id = int(item["end_segment_id"])
            start_seconds = float(item["start_seconds"])
            end_seconds = float(item["end_seconds"])
            chapter_segments = [
                segment_by_id[segment_id]
                for segment_id in range(start_id, end_id + 1)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ChapteringError(f"invalid chapter export data: {exc}") from exc
        filename = f"{chapter_id:02d}-{_safe_filename(title)}.md"
        path = sections_dir / filename
        lines = [f"# {title}", "", f"> 所属视频：{video_title}", "", "## 章节摘要", ""]
        lines.append(str(item.get("summary") or "").strip())
        lines.extend(["", "## 主题", ""])
        topics = [str(topic).strip() for topic in item.get("topics", []) if str(topic).strip()]
        lines.append("、".join(topics))
        lines.extend(["", "## 转写正文", ""])
        texts = [segment.text for segment in chapter_segments]
        for paragraph in _paragraphs(texts):
            lines.extend([paragraph, ""])
        start = format_timestamp(start_seconds)
        end = format_timestamp(end_seconds)
        label = f"{start}–{end}"
        if source_url:
            label = f"[{label}]({timestamp_url(source_url, start_seconds)})"
        lines.extend(["## 来源", "", f"原视频：{label}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        section_paths.append(path)
    for stale_path in previous_section_paths - set(section_paths):
        stale_path.unlink()
    return sections_dir, tuple(section_paths)


def extract_chapters(
    transcript_file: str | Path,
    output_dir: str | Path,
    *,
    model: str | None = None,
    provider: str = "deepseek",
    reasoning_effort: str = "none",
    output_language: str = "source",
    max_output_tokens: int = 8192,
    api_key: str | None = None,
    client: Any | None = None,
) -> ChapterArtifact:
    """Extract validated, contiguous chapters from a corrected transcript."""
    input_path = Path(transcript_file).resolve()
    if not input_path.is_file():
        raise ValueError(f"transcript file does not exist: {input_path}")
    if provider not in {"deepseek", "local"}:
        raise ValueError("provider must be 'deepseek' or 'local'")
    if reasoning_effort not in {"none", "low", "high", "max"}:
        raise ValueError("DeepSeek reasoning_effort must be none, low, high, or max")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")
    source, segments = _load_source(input_path)
    normalized_output_language = resolve_output_language(
        output_language, str(source.get("language") or "unknown")
    )
    if provider == "local":
        from .local_text import load_local_text_client, local_text_model
        selected_model = local_text_model(model)
        text_client = client or load_local_text_client()
    else:
        selected_model = model or "deepseek-flash"
        text_client = client or _load_deepseek_client(api_key)
    usage = empty_usage()
    chapters: list[Chapter] | None = None
    for attempt in range(2):
        try:
            raw_chapters, attempt_usage = _request_chapters(
                text_client,
                model=selected_model,
                reasoning_effort=reasoning_effort,
                title=str(source.get("title") or ""),
                source_language=str(source.get("language") or "unknown"),
                output_language=normalized_output_language,
                segments=segments,
                max_output_tokens=max_output_tokens,
            )
            add_usage(usage, attempt_usage)
            chapters = _validate_chapters(raw_chapters, segments)
            break
        except ChapteringError:
            if attempt == 1:
                raise
        except Exception as exc:
            raise ChapteringError(f"chapter extraction failed: {exc}") from exc
    if chapters is None:
        raise ChapteringError("chapter extraction did not produce valid chapters")
    json_path, markdown_path, sections_dir, section_paths = _write_outputs(
        output_dir=Path(output_dir).resolve(),
        input_path=input_path,
        source=source,
        provider=provider,
        model=selected_model,
        reasoning_effort=reasoning_effort,
        output_language=normalized_output_language,
        chapters=chapters,
        segment_count=len(segments),
        usage=usage,
    )
    return ChapterArtifact(
        input_path=input_path,
        json_path=json_path,
        markdown_path=markdown_path,
        sections_dir=sections_dir,
        section_paths=section_paths,
        provider=provider,
        model=selected_model,
        chapter_count=len(chapters),
        segment_count=len(segments),
        **usage,
    )
