"""Evidence-grounded structured knowledge extraction by video chapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .costing import estimate_deepseek_flash_cost, local_text_cost, response_usage
from .languages import resolve_output_language
from .models import format_timestamp, timestamp_url
from .proofreading import _load_deepseek_client
from .structured_output import loads_json_object


class KnowledgeExtractionError(RuntimeError):
    """Raised when structured knowledge cannot be extracted or validated."""


@dataclass(frozen=True, slots=True)
class KnowledgeArtifact:
    input_path: Path
    json_path: Path
    markdown_path: Path
    items_dir: Path
    model: str
    chapter_count: int
    entity_count: int
    knowledge_count: int
    executed_chapters: int
    reused_chapters: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("input_path", "json_path", "markdown_path", "items_dir"):
            result[key] = str(result[key])
        return result


KNOWLEDGE_TYPES = (
    "fact",
    "number",
    "event",
    "causal_relation",
    "conclusion",
    "procedure",
    "definition",
    "recommendation",
    "uncertainty",
)

KNOWLEDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapter_summary": {"type": "string"},
        "entities": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "knowledge_items": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(KNOWLEDGE_TYPES)},
                    "statement": {"type": "string"},
                    "evidence_segment_ids": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {"type": "integer"},
                    },
                    "keywords": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "type",
                    "statement",
                    "evidence_segment_ids",
                    "keywords",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chapter_summary", "entities", "knowledge_items"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是多语言视频知识抽取器。请从一个章节的校对后转写中提取可复用的结构化知识，并输出 JSON。

必须遵守：
1. 只能提取输入文本明确表达的信息，不得根据常识补充或推测。
2. 每条 knowledge_item 必须引用一个或多个真实 segment ID 作为直接证据。
3. statement 应脱离口语上下文后仍能独立理解，但不得改变原意。
4. 数字、金额、时间、数量、比例和专有名词应准确保留。
5. 对存疑、传闻或估计内容使用 uncertainty，或降低 confidence，不要改写成确定事实。
6. 合并语义重复的条目；不要把每句话机械地变成知识条目。
7. entities 只保留理解本章有价值的人物、组织、地点、物品或概念。
8. 每章最多返回 12 条高价值 knowledge_items，优先保留可复用的事实、数字、因果和结论。
9. chapter_summary、实体描述、statement 和 keywords 必须使用 output_language 指定的语言；专有名词可保留原文，不得改变证据含义。
"""


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _load_inputs(chapters_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        chapters = json.loads(chapters_path.read_text(encoding="utf-8"))
        transcript_path = Path(chapters["input_transcript"]).resolve()
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = list(transcript["segments"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise KnowledgeExtractionError(f"cannot read chapter inputs: {exc}") from exc
    if not chapters.get("chapters") or not segments:
        raise KnowledgeExtractionError("chapter input is empty")
    return chapters, transcript, segments


def _chapter_segments(chapter: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_id = int(chapter["start_segment_id"])
    end_id = int(chapter["end_segment_id"])
    selected = [item for item in segments if start_id <= int(item["id"]) <= end_id]
    if not selected or int(selected[0]["id"]) != start_id or int(selected[-1]["id"]) != end_id:
        raise KnowledgeExtractionError(f"chapter {chapter.get('id')} segment range is incomplete")
    return selected


def _source_hash(
    chapter: dict[str, Any], segments: list[dict[str, Any]], output_language: str
) -> str:
    payload = {
        "chapter": chapter,
        "output_language": output_language,
        "segments": [
            {"id": item["id"], "text": item.get("corrected_text") or item.get("text") or ""}
            for item in segments
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _request(client: Any, *, model: str, reasoning_effort: str, output_language: str, chapter: dict[str, Any], segments: list[dict[str, Any]], max_output_tokens: int) -> tuple[dict[str, Any], dict[str, int]]:
    payload = {
        "output_language": output_language,
        "chapter": {
            "id": chapter["id"],
            "title": chapter["title"],
            "summary": chapter["summary"],
        },
        "segments": [
            {
                "id": int(item["id"]),
                "text": str(item.get("corrected_text") or item.get("text") or ""),
            }
            for item in segments
        ],
    }
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "chapter_knowledge",
                "strict": True,
                "schema": KNOWLEDGE_SCHEMA,
            }
        },
    )
    output_text = str(_value(response, "output_text", "") or "")
    if not output_text:
        raise KnowledgeExtractionError("text model returned no knowledge output")
    try:
        result = loads_json_object(output_text)
    except json.JSONDecodeError as exc:
        raise KnowledgeExtractionError(f"invalid knowledge output: {exc}") from exc
    return result, response_usage(response)


def _validate(
    raw: dict[str, Any],
    *,
    chapter: dict[str, Any],
    segments: list[dict[str, Any]],
    source_url: str,
) -> dict[str, Any]:
    segment_by_id = {int(item["id"]): item for item in segments}
    summary = str(raw.get("chapter_summary") or "").strip()
    if not summary:
        raise KnowledgeExtractionError(f"chapter {chapter['id']} has an empty summary")
    entities = []
    for entity in raw.get("entities", []):
        name = str(entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "").strip()
        description = str(entity.get("description") or "").strip()
        if name and entity_type and description:
            entities.append({"name": name, "type": entity_type, "description": description})

    items = []
    for index, item in enumerate(raw.get("knowledge_items", []), start=1):
        item_type = str(item.get("type") or "")
        statement = str(item.get("statement") or "").strip()
        confidence = str(item.get("confidence") or "")
        evidence_ids = list(dict.fromkeys(int(value) for value in item.get("evidence_segment_ids", [])))
        if item_type not in KNOWLEDGE_TYPES or not statement or confidence not in {"high", "medium", "low"}:
            raise KnowledgeExtractionError(f"chapter {chapter['id']} knowledge item {index} is invalid")
        if not evidence_ids or any(segment_id not in segment_by_id for segment_id in evidence_ids):
            raise KnowledgeExtractionError(f"chapter {chapter['id']} item {index} has invalid evidence IDs")
        evidence = [segment_by_id[segment_id] for segment_id in evidence_ids]
        start_seconds = min(float(value["start_seconds"]) for value in evidence)
        end_seconds = max(float(value["end_seconds"]) for value in evidence)
        items.append(
            {
                "id": f"c{int(chapter['id']):02d}-k{index:03d}",
                "type": item_type,
                "statement": statement,
                "keywords": list(dict.fromkeys(str(value).strip() for value in item.get("keywords", []) if str(value).strip())),
                "confidence": confidence,
                "evidence_segment_ids": evidence_ids,
                "evidence_text": "".join(str(value.get("corrected_text") or value.get("text") or "") for value in evidence),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "source_url": timestamp_url(source_url, start_seconds) if source_url else "",
            }
        )
    if not items:
        raise KnowledgeExtractionError(f"chapter {chapter['id']} contains no knowledge items")
    return {
        "chapter_id": int(chapter["id"]),
        "chapter_title": str(chapter["title"]),
        "chapter_summary": summary,
        "entities": entities,
        "knowledge_items": items,
    }


def _write_combined(output_dir: Path, chapters_payload: dict[str, Any], results: list[dict[str, Any]], provider: str, model: str, reasoning_effort: str, output_language: str, usage: dict[str, int]) -> tuple[Path, Path]:
    json_path = output_dir / "knowledge.json"
    markdown_path = output_dir / "knowledge.md"
    combined = {
        "source_url": chapters_payload.get("source_url", ""),
        "title": chapters_payload.get("title", ""),
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "source_language": chapters_payload.get("source_language", "unknown"),
        "output_language": output_language,
        "usage": usage,
        "cost_estimate": estimate_deepseek_flash_cost(usage) if provider == "deepseek" else local_text_cost(model),
        "chapters": results,
    }
    json_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {chapters_payload.get('title') or 'Video'} — Structured Knowledge", ""]
    for chapter in results:
        lines.extend([f"## {chapter['chapter_id']}. {chapter['chapter_title']}", "", chapter["chapter_summary"], ""])
        for item in chapter["knowledge_items"]:
            label = f"{format_timestamp(item['start_seconds'])}–{format_timestamp(item['end_seconds'])}"
            if item["source_url"]:
                label = f"[{label}]({item['source_url']})"
            lines.append(f"- **{item['type']}**：{item['statement']}（{label}）")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def extract_structured_knowledge(
    chapters_file: str | Path,
    output_dir: str | Path,
    *,
    model: str | None = None,
    provider: str = "deepseek",
    reasoning_effort: str = "none",
    output_language: str | None = None,
    max_output_tokens: int = 8192,
    api_key: str | None = None,
    client: Any | None = None,
) -> KnowledgeArtifact:
    """Extract and locally validate reusable knowledge from every chapter."""
    input_path = Path(chapters_file).resolve()
    if not input_path.is_file():
        raise ValueError(f"chapters file does not exist: {input_path}")
    if provider not in {"deepseek", "local"}:
        raise ValueError("provider must be 'deepseek' or 'local'")
    if reasoning_effort not in {"none", "low", "high", "max"}:
        raise ValueError("DeepSeek reasoning_effort must be none, low, high, or max")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")
    chapters_payload, transcript_payload, segments = _load_inputs(input_path)
    source_language = str(
        chapters_payload.get("source_language")
        or transcript_payload.get("language")
        or "unknown"
    )
    normalized_output_language = resolve_output_language(
        output_language or str(chapters_payload.get("output_language") or "source"),
        source_language,
    )
    destination = Path(output_dir).resolve()
    items_dir = destination / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    if provider == "local":
        from .local_text import load_local_text_client, local_text_model
        selected_model = local_text_model(model)
        text_client = client or load_local_text_client()
    else:
        selected_model = model or "deepseek-flash"
        text_client = client or _load_deepseek_client(api_key)
    results: list[dict[str, Any]] = []
    total_usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    executed = 0
    reused = 0
    source_url = str(chapters_payload.get("source_url") or "")

    for chapter in chapters_payload["chapters"]:
        chapter_id = int(chapter["id"])
        selected = _chapter_segments(chapter, segments)
        fingerprint = _source_hash(chapter, selected, normalized_output_language)
        item_path = items_dir / f"{chapter_id:02d}.json"
        cached = None
        if item_path.is_file():
            try:
                candidate = json.loads(item_path.read_text(encoding="utf-8"))
                if candidate.get("source_hash") == fingerprint and candidate.get("provider", "deepseek") == provider and candidate.get("model") == selected_model and candidate.get("reasoning_effort") == reasoning_effort:
                    cached = candidate
            except (OSError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            results.append(cached["result"])
            cached_usage = cached.get("usage", {})
            for key in total_usage:
                total_usage[key] += int(cached_usage.get(key, 0) or 0)
            reused += 1
            continue
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        result = None
        for attempt in range(2):
            try:
                raw, attempt_usage = _request(
                    text_client,
                    model=selected_model,
                    reasoning_effort=reasoning_effort,
                    output_language=normalized_output_language,
                    chapter=chapter,
                    segments=selected,
                    max_output_tokens=max_output_tokens,
                )
                for key in usage:
                    usage[key] += int(attempt_usage.get(key, 0) or 0)
                result = _validate(
                    raw,
                    chapter=chapter,
                    segments=selected,
                    source_url=source_url,
                )
                break
            except KnowledgeExtractionError:
                if attempt == 1:
                    raise
            except Exception as exc:
                raise KnowledgeExtractionError(
                    f"chapter {chapter_id} extraction failed: {exc}"
                ) from exc
        if result is None:
            raise KnowledgeExtractionError(
                f"chapter {chapter_id} extraction did not produce valid knowledge"
            )
        item_path.write_text(
            json.dumps(
                {"source_hash": fingerprint, "provider": provider, "model": selected_model, "reasoning_effort": reasoning_effort, "output_language": normalized_output_language, "usage": usage, "result": result},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        for key in total_usage:
            total_usage[key] += usage[key]
        results.append(result)
        executed += 1

    json_path, markdown_path = _write_combined(destination, chapters_payload, results, provider, selected_model, reasoning_effort, normalized_output_language, total_usage)
    return KnowledgeArtifact(
        input_path=input_path,
        json_path=json_path,
        markdown_path=markdown_path,
        items_dir=items_dir,
        model=selected_model,
        chapter_count=len(results),
        entity_count=sum(len(item["entities"]) for item in results),
        knowledge_count=sum(len(item["knowledge_items"]) for item in results),
        executed_chapters=executed,
        reused_chapters=reused,
        **total_usage,
    )
