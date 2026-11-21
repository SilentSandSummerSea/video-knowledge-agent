"""Timestamp-preserving transcript proofreading with the OpenAI API."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from .costing import add_usage, empty_usage, estimate_deepseek_flash_cost, local_text_cost, response_usage
from .models import format_timestamp, timestamp_url
from .project_env import project_secret
from .structured_output import loads_json_object
from .transcription import TranscriptSegment


class ProofreadingError(RuntimeError):
    """Raised when a transcript cannot be proofread safely."""


@dataclass(frozen=True, slots=True)
class ProofreadSegment:
    id: int
    start_seconds: float
    end_seconds: float
    asr_text: str
    corrected_text: str
    changed: bool
    confidence: str
    needs_review: bool
    note: str


@dataclass(frozen=True, slots=True)
class ProofreadingArtifact:
    input_path: Path
    json_path: Path
    markdown_path: Path
    review_queue_path: Path
    provider: str
    model: str
    segment_count: int
    changed_count: int
    review_count: int
    batch_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("input_path", "json_path", "markdown_path", "review_queue_path"):
            result[key] = str(result[key])
        return result


PROOFREAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "corrected_text": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "needs_review": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": [
                    "id",
                    "corrected_text",
                    "confidence",
                    "needs_review",
                    "note",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是多语言 ASR 文本校对员。根据 source_language 校对输入，只修正有充分上下文依据的识别错误、拼写、专有名词、标点和明显断句问题。
必须遵守：
1. 不增加、删除或总结原文信息，不改变说话者含义。
2. corrected_text 必须保留原文语言和文字系统，不得翻译成另一种语言；混合语言内容也应保持混合语言。
3. 不得猜测或补写 ASR 可能遗漏的主语、连接词、修饰语或事实。怀疑缺词时保留原文并标记 needs_review=true。
4. 每个 editable=true 的片段必须返回且只返回一次，ID 必须保持不变。
5. editable=false 的片段仅供上下文参考，不得返回。
6. 不确定时保留原文，needs_review=true，并用 source_language 对应的语言在 note 中简短说明疑点。
7. confidence 表示本次校对判断的置信度；note 没有内容时返回空字符串。
8. preferred_terms 是从用户输入或视频元数据取得的已验证拼写。仅在发音和上下文明确对应时使用；不得为了套用术语而改写无关文本。
"""


def _glossary_candidate(term: str) -> bool:
    return bool(
        " " in term.strip()
        or (len(term) > 1 and term.isupper())
        or re.search(r"[0-9]", term)
        or re.search(r"[^\x00-\x7f]", term)
    )


def _apply_verified_glossary(
    segments: list[ProofreadSegment], terms: list[str]
) -> list[ProofreadSegment]:
    """Normalize close variants only against spellings backed by the glossary.

    This deliberately avoids free-form entity guessing. A replacement requires the
    same number of whitespace-delimited words, a high character similarity score,
    and matching first characters for every word.
    """
    verified = [term for term in terms if _glossary_candidate(term)]
    result: list[ProofreadSegment] = []
    for segment in segments:
        text = segment.corrected_text
        applied: list[str] = []
        for term in sorted(verified, key=len, reverse=True):
            target_words = term.split()
            if not target_words:
                continue
            word_patterns = []
            for word in target_words:
                first = re.escape(word[0])
                word_patterns.append(first + r"[\w'’+.-]*")
            pattern = re.compile(
                r"\b" + r"\s+".join(word_patterns) + r"\b",
                re.IGNORECASE | re.UNICODE,
            )

            def replace(match: re.Match[str]) -> str:
                candidate = match.group(0)
                if candidate.casefold() == term.casefold():
                    return term if candidate != term else candidate
                candidate_words = candidate.split()
                if len(candidate_words) != len(target_words):
                    return candidate
                if any(
                    not left or not right or left[0].casefold() != right[0].casefold()
                    for left, right in zip(candidate_words, target_words)
                ):
                    return candidate
                ratio = SequenceMatcher(None, candidate.casefold(), term.casefold()).ratio()
                if ratio < 0.82:
                    return candidate
                applied.append(f"{candidate} → {term}")
                return term

            text = pattern.sub(replace, text)
        if applied:
            prior_note = segment.note.strip()
            glossary_note = "verified glossary: " + "; ".join(applied)
            result.append(
                ProofreadSegment(
                    id=segment.id,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    asr_text=segment.asr_text,
                    corrected_text=text,
                    changed=text != segment.asr_text,
                    confidence=segment.confidence,
                    needs_review=segment.needs_review,
                    note="; ".join(value for value in (prior_note, glossary_note) if value),
                )
            )
        else:
            result.append(segment)
    return result


def _enforce_note_language(
    segments: list[ProofreadSegment], source_language: str
) -> list[ProofreadSegment]:
    """Prevent review metadata from silently switching away from English."""
    normalized = source_language.strip().casefold()
    if normalized not in {"en", "eng", "english"}:
        return segments
    cjk = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
    result: list[ProofreadSegment] = []
    for segment in segments:
        if segment.note and cjk.search(segment.note):
            result.append(
                ProofreadSegment(
                    id=segment.id,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    asr_text=segment.asr_text,
                    corrected_text=segment.corrected_text,
                    changed=segment.changed,
                    confidence=segment.confidence,
                    needs_review=True,
                    note="Uncertain transcription; original wording preserved for review.",
                )
            )
        else:
            result.append(segment)
    return result


def _load_client(api_key: str | None = None) -> Any:
    key = project_secret("OPENAI_API_KEY", api_key)
    if not key:
        raise ProofreadingError(
            "OPENAI_API_KEY is not set; store the OpenAI API key in the environment"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProofreadingError(
            "the OpenAI SDK is not installed; install it with "
            "'python -m pip install -e .[openai]'"
        ) from exc
    return OpenAI(api_key=key)


def _load_deepseek_client(api_key: str | None = None) -> Any:
    key = project_secret("DEEPSEEK_API_KEY", api_key)
    if not key:
        raise ProofreadingError(
            "DEEPSEEK_API_KEY is not set; store the key in the project .env file"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProofreadingError(
            "the OpenAI-compatible SDK is not installed; install it with "
            "'python -m pip install -e .[openai]'"
        ) from exc
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _load_transcript(path: Path) -> tuple[dict[str, Any], list[TranscriptSegment]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofreadingError(f"cannot read transcript JSON: {exc}") from exc
    try:
        segments = [TranscriptSegment(**item) for item in payload["segments"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofreadingError(f"invalid transcript schema: {exc}") from exc
    if not segments:
        raise ProofreadingError("transcript contains no segments")
    if len({segment.id for segment in segments}) != len(segments):
        raise ProofreadingError("transcript contains duplicate segment IDs")
    return payload, segments


def _make_batches(
    segments: list[TranscriptSegment], batch_seconds: float, max_segments: int = 8
) -> list[list[TranscriptSegment]]:
    if batch_seconds <= 0:
        raise ValueError("batch_seconds must be greater than zero")
    if max_segments <= 0:
        raise ValueError("max_segments must be greater than zero")
    batches: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    batch_start = segments[0].start_seconds
    for segment in segments:
        if current and (
            segment.end_seconds - batch_start > batch_seconds
            or len(current) >= max_segments
        ):
            batches.append(current)
            current = []
            batch_start = segment.start_seconds
        current.append(segment)
    if current:
        batches.append(current)
    return batches


def _batch_payload(
    *,
    title: str,
    source_language: str,
    terms: list[str],
    context_before: list[TranscriptSegment],
    targets: list[TranscriptSegment],
    context_after: list[TranscriptSegment],
) -> dict[str, Any]:
    target_ids = {segment.id for segment in targets}
    records = [
        {
            "id": segment.id,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "text": segment.text,
            "editable": segment.id in target_ids,
        }
        for segment in [*context_before, *targets, *context_after]
    ]
    return {
        "video_title": title,
        "source_language": source_language,
        "preferred_terms": terms,
        "segments": records,
    }


def _request_batch_responses(
    client: Any,
    *,
    provider_name: str,
    model: str,
    reasoning_effort: str,
    title: str,
    source_language: str,
    terms: list[str],
    context_before: list[TranscriptSegment],
    targets: list[TranscriptSegment],
    context_after: list[TranscriptSegment],
    max_output_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    user_payload = _batch_payload(
        title=title,
        source_language=source_language,
        terms=terms,
        context_before=context_before,
        targets=targets,
        context_after=context_after,
    )
    request: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "proofread_transcript",
                "strict": True,
                "schema": PROOFREAD_SCHEMA,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    if provider_name == "deepseek":
        request["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise ProofreadingError(
            f"{provider_name} returned no structured proofreading output"
        )
    try:
        result = loads_json_object(output_text)
        return list(result["segments"]), response_usage(response)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ProofreadingError(
            f"invalid {provider_name} proofreading output: {exc}"
        ) from exc


def _find_git_root(start: Path) -> Path | None:
    candidate = start if start.is_dir() else start.parent
    for path in (candidate, *candidate.parents):
        if (path / ".git").exists():
            return path
    return None


def _run_codex(
    prompt: str,
    *,
    schema: dict[str, Any],
    model: str,
    reasoning_effort: str,
    workdir: Path,
    timeout_seconds: float,
) -> str:
    executable = shutil.which("codex")
    if not executable:
        raise ProofreadingError("Codex CLI is not installed or not available on PATH")
    git_root = _find_git_root(workdir)
    run_root = git_root or workdir
    with tempfile.TemporaryDirectory(prefix="video-proofread-") as directory:
        temp_dir = Path(directory)
        schema_path = temp_dir / "schema.json"
        output_path = temp_dir / "result.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = [
            executable,
            "exec",
            "-",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--color",
            "never",
            "--cd",
            str(run_root),
        ]
        if git_root is None:
            command.append("--skip-git-repo-check")
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=run_root,
                timeout=timeout_seconds,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise ProofreadingError(
                f"Codex proofreading timed out after {timeout_seconds:g} seconds"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2000:] or "unknown Codex CLI error"
            raise ProofreadingError(f"Codex proofreading failed: {detail}")
        if output_path.exists():
            return output_path.read_text(encoding="utf-8")
        return completed.stdout.strip()


def _request_batch_codex(
    *,
    runner: Callable[..., str],
    model: str,
    reasoning_effort: str,
    title: str,
    source_language: str,
    terms: list[str],
    context_before: list[TranscriptSegment],
    targets: list[TranscriptSegment],
    context_after: list[TranscriptSegment],
    workdir: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    user_payload = _batch_payload(
        title=title,
        source_language=source_language,
        terms=terms,
        context_before=context_before,
        targets=targets,
        context_after=context_after,
    )
    prompt = (
        SYSTEM_PROMPT
        + "\n不要调用工具、不要读取文件，只处理下面提供的 JSON，并只输出符合 schema 的 JSON。\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )
    output_text = runner(
        prompt,
        schema=PROOFREAD_SCHEMA,
        model=model,
        reasoning_effort=reasoning_effort,
        workdir=workdir,
        timeout_seconds=timeout_seconds,
    )
    if not output_text:
        raise ProofreadingError("Codex returned no structured proofreading output")
    try:
        result = json.loads(output_text)
        return list(result["segments"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ProofreadingError(f"invalid Codex proofreading output: {exc}") from exc


def _validate_batch(
    source: list[TranscriptSegment], result: Iterable[dict[str, Any]]
) -> list[ProofreadSegment]:
    source_by_id = {segment.id: segment for segment in source}
    returned: dict[int, dict[str, Any]] = {}
    for item in result:
        try:
            segment_id = int(item["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofreadingError(f"proofreading result has an invalid ID: {exc}") from exc
        # Some providers occasionally echo editable=false context records even
        # when instructed not to. They are untrusted and must never modify the
        # transcript, but are harmless to discard when every target is present.
        if segment_id not in source_by_id:
            continue
        if segment_id in returned:
            raise ProofreadingError(f"proofreading returned duplicate ID {segment_id}")
        returned[segment_id] = item
    expected_ids = set(source_by_id)
    if set(returned) != expected_ids:
        missing = sorted(expected_ids - set(returned))
        raise ProofreadingError(
            f"proofreading ID mismatch; missing={missing}"
        )

    validated: list[ProofreadSegment] = []
    for segment in source:
        item = returned[segment.id]
        corrected = str(item.get("corrected_text", "")).strip()
        confidence = str(item.get("confidence", ""))
        if not corrected:
            raise ProofreadingError(f"proofreading returned empty text for ID {segment.id}")
        if confidence not in {"high", "medium", "low"}:
            raise ProofreadingError(
                f"proofreading returned invalid confidence for ID {segment.id}"
            )
        validated.append(
            ProofreadSegment(
                id=segment.id,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                asr_text=segment.text,
                corrected_text=corrected,
                changed=corrected != segment.text,
                confidence=confidence,
                needs_review=bool(item.get("needs_review", False)),
                note=str(item.get("note", "")).strip(),
            )
        )
    return validated


def _write_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    source: dict[str, Any],
    provider: str,
    model: str,
    reasoning_effort: str,
    terms: list[str],
    segments: list[ProofreadSegment],
    batch_count: int,
    usage: dict[str, int],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "proofread.json"
    markdown_path = output_dir / "proofread.md"
    review_queue_path = output_dir / "review_queue.json"
    payload = {
        "source_url": source.get("source_url", ""),
        "title": source.get("title", ""),
        "input_transcript": str(input_path),
        "asr_model": source.get("model", ""),
        "language": source.get("language", "unknown"),
        "proofreading_provider": provider,
        "proofreading_model": model,
        "reasoning_effort": reasoning_effort,
        "duration_seconds": source.get("duration_seconds", 0.0),
        "batch_count": batch_count,
        "preferred_terms": list(terms),
        "usage": usage,
        "cost_estimate": (
            estimate_deepseek_flash_cost(usage) if provider == "deepseek"
            else local_text_cost(model) if provider == "local" else None
        ),
        "segments": [asdict(segment) for segment in segments],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    title = str(source.get("title") or input_path.stem)
    source_url = str(source.get("source_url") or "")
    lines = [f"# {title} — Proofread Transcript", ""]
    lines.append(f"- ASR model: `{source.get('model', '')}`")
    lines.append(f"- Language: `{source.get('language', 'unknown')}`")
    lines.append(f"- Proofreading provider: `{provider}`")
    lines.append(f"- Proofreading model: `{model}`")
    lines.append(f"- Reasoning: `{reasoning_effort}`")
    lines.append(f"- Batches: `{batch_count}`")
    lines.append("")
    for segment in segments:
        label = format_timestamp(segment.start_seconds)
        if source_url:
            label = f"[{label}]({timestamp_url(source_url, segment.start_seconds)})"
        review = " ⚠️ needs review" if segment.needs_review else ""
        lines.append(f"- {label} {segment.corrected_text}{review}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review_items = [
        {
            "segment_id": segment.id,
            "start_seconds": max(0.0, segment.start_seconds - 3.0),
            "end_seconds": segment.end_seconds + 3.0,
            "asr_text": segment.asr_text,
            "corrected_text": segment.corrected_text,
            "confidence": segment.confidence,
            "note": segment.note,
        }
        for segment in segments
        if segment.needs_review or segment.confidence == "low"
    ]
    review_queue_path.write_text(
        json.dumps(
            {
                "source_url": source.get("source_url", ""),
                "input_transcript": str(input_path),
                "preferred_terms": list(terms),
                "segments": review_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path, review_queue_path


def proofread_transcript(
    transcript_file: str | Path,
    output_dir: str | Path,
    *,
    provider: str = "deepseek",
    model: str | None = None,
    reasoning_effort: str = "none",
    batch_seconds: float = 300.0,
    context_segments: int = 3,
    max_output_tokens: int = 16384,
    terms: Iterable[str] = (),
    api_key: str | None = None,
    client: Any | None = None,
    codex_runner: Callable[..., str] | None = None,
    timeout_seconds: float = 600.0,
) -> ProofreadingArtifact:
    """Proofread a transcript while preserving every segment ID and timestamp."""
    input_path = Path(transcript_file).resolve()
    if not input_path.is_file():
        raise ValueError(f"transcript file does not exist: {input_path}")
    if context_segments < 0:
        raise ValueError("context_segments must not be negative")
    if provider not in {"deepseek", "codex", "openai", "local"}:
        raise ValueError("provider must be 'deepseek', 'codex', 'openai', or 'local'")
    if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max", "ultra"}:
        raise ValueError("unsupported reasoning_effort")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")
    source, segments = _load_transcript(input_path)
    source_language = str(source.get("language") or "unknown")
    batches = _make_batches(segments, batch_seconds)
    default_models = {
        "deepseek": "deepseek-flash",
        "codex": "gpt-6-luna",
        "openai": "gpt-4o-mini",
        "local": "",
    }
    if provider == "local":
        from .local_text import local_text_model
        selected_model = local_text_model(model)
    else:
        selected_model = model or default_models[provider]
    effective_reasoning_effort = (
        "low" if provider == "codex" and reasoning_effort == "none" else reasoning_effort
    )
    responses_client = None
    if provider == "deepseek":
        responses_client = client or _load_deepseek_client(api_key)
    elif provider == "openai":
        responses_client = client or _load_client(api_key)
    elif provider == "local":
        from .local_text import load_local_text_client
        responses_client = client or load_local_text_client()
    runner = codex_runner or _run_codex
    preferred_terms = [term.strip() for term in terms if term.strip()]
    corrected: list[ProofreadSegment] = []
    total_usage = empty_usage()

    offset = 0
    for batch_number, batch in enumerate(batches, start=1):
        start_index = offset
        end_index = start_index + len(batch)
        before = segments[max(0, start_index - context_segments) : start_index]
        after = segments[end_index : end_index + context_segments]
        try:
            if provider == "codex":
                raw_result = _request_batch_codex(
                    runner=runner,
                    model=selected_model,
                    reasoning_effort=effective_reasoning_effort,
                    title=str(source.get("title") or ""),
                    source_language=source_language,
                    terms=preferred_terms,
                    context_before=before,
                    targets=batch,
                    context_after=after,
                    workdir=input_path.parent,
                    timeout_seconds=timeout_seconds,
                )
            else:
                raw_result, usage = _request_batch_responses(
                    responses_client,
                    provider_name=provider,
                    model=selected_model,
                    reasoning_effort=effective_reasoning_effort,
                    title=str(source.get("title") or ""),
                    source_language=source_language,
                    terms=preferred_terms,
                    context_before=before,
                    targets=batch,
                    context_after=after,
                    max_output_tokens=max_output_tokens,
                )
                add_usage(total_usage, usage)
            corrected.extend(_validate_batch(batch, raw_result))
        except ProofreadingError:
            raise
        except Exception as exc:
            raise ProofreadingError(
                f"OpenAI proofreading failed for batch {batch_number}: {exc}"
            ) from exc
        offset = end_index

    corrected = _apply_verified_glossary(corrected, preferred_terms)
    corrected = _enforce_note_language(corrected, source_language)
    destination = Path(output_dir).resolve()
    json_path, markdown_path, review_queue_path = _write_outputs(
        output_dir=destination,
        input_path=input_path,
        source=source,
        provider=provider,
        model=selected_model,
        reasoning_effort=effective_reasoning_effort,
        terms=preferred_terms,
        segments=corrected,
        batch_count=len(batches),
        usage=total_usage,
    )
    return ProofreadingArtifact(
        input_path=input_path,
        json_path=json_path,
        markdown_path=markdown_path,
        review_queue_path=review_queue_path,
        provider=provider,
        model=selected_model,
        segment_count=len(corrected),
        changed_count=sum(segment.changed for segment in corrected),
        review_count=sum(segment.needs_review for segment in corrected),
        batch_count=len(batches),
        **total_usage,
    )
