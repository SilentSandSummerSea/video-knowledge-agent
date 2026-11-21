"""Resumable end-to-end video knowledge processing pipeline."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio import prepare_local_audio
from .asr_review import review_flagged_segments_groq
from .chaptering import export_chapter_sections, extract_chapters
from .costing import add_usage, empty_usage, estimate_deepseek_flash_cost
from .groq_transcription import transcribe_audio_groq_chunked
from .knowledge import extract_structured_knowledge
from .indexing import build_index
from .languages import normalize_language, normalize_output_language, resolve_output_language
from .moments import build_moments
from .project_env import project_secret
from .proofreading import proofread_transcript
from .publishing import publish_knowledge
from .terminology import build_asr_prompt, extract_metadata_terms, merge_terms
from .transcription import transcribe_audio


class PipelineError(RuntimeError):
    """Raised when an end-to-end processing stage fails."""


ProgressCallback = Callable[[str, str, int, str], None]


class ConsoleProgress:
    """Render pipeline progress without mixing it into the JSON result stream."""

    def __init__(self, *, width: int = 30) -> None:
        self.width = max(10, width)

    def __call__(self, stage: str, status: str, percent: int, detail: str = "") -> None:
        bounded = max(0, min(100, int(percent)))
        filled = round(self.width * bounded / 100)
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" - {detail}" if detail else ""
        print(
            f"[video] [{bar}] {bounded:3d}% {stage}: {status}{suffix}",
            file=sys.stderr,
            flush=True,
        )


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    status: str,
    percent: int,
    detail: str = "",
) -> None:
    if callback is None:
        return
    try:
        callback(stage, status, percent, detail)
    except Exception:
        # Display failures must never interrupt an otherwise resumable pipeline.
        return


@dataclass(frozen=True, slots=True)
class PipelineArtifact:
    video_id: str
    title: str
    source_url: str
    source_path: Path
    video_path: Path | None
    source_language: str
    output_language: str
    run_dir: Path
    manifest_path: Path
    audio_path: Path
    transcript_path: Path
    proofread_path: Path
    asr_review_path: Path | None
    asr_review_audit_path: Path | None
    chapters_path: Path
    chapter_sections_dir: Path
    moments_path: Path
    knowledge_dir: Path
    knowledge_metadata_path: Path
    structured_knowledge_path: Path
    index_database_path: Path
    executed_stages: tuple[str, ...]
    reused_stages: tuple[str, ...]
    deepseek_usage: dict[str, int]
    estimated_deepseek_cost_cny: float
    estimated_run_cost_cny: float
    cost_estimate_complete: bool
    pricing_periods: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in (
            "source_path",
            "video_path",
            "run_dir",
            "manifest_path",
            "audio_path",
            "transcript_path",
            "proofread_path",
            "chapters_path",
            "chapter_sections_dir",
            "moments_path",
            "knowledge_dir",
            "knowledge_metadata_path",
            "structured_knowledge_path",
            "index_database_path",
        ):
            result[key] = str(result[key]) if result[key] is not None else None
        for key in ("asr_review_path", "asr_review_audit_path"):
            result[key] = str(result[key]) if result[key] is not None else None
        result["executed_stages"] = list(self.executed_stages)
        result["reused_stages"] = list(self.reused_stages)
        result["pricing_periods"] = list(self.pricing_periods)
        return result


def _safe_video_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise PipelineError("media loader returned an empty or invalid video ID")
    return safe


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _completed_file(state: dict[str, Any], stage: str, key: str) -> Path | None:
    item = state.get("stages", {}).get(stage, {})
    if item.get("status") != "completed" or not item.get(key):
        return None
    path = Path(str(item[key])).resolve()
    return path if path.is_file() else None


def _billing_from_output(path: Path) -> tuple[dict[str, int] | None, dict[str, Any] | None]:
    payload = _read_json(path) or {}
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return None, None
    usage = empty_usage()
    for key in usage:
        usage[key] = int(raw_usage.get(key, 0) or 0)
    raw_estimate = payload.get("cost_estimate")
    estimate = raw_estimate if isinstance(raw_estimate, dict) else estimate_deepseek_flash_cost(usage)
    return usage, estimate


def process_video(
    video_file: str | Path,
    *,
    raw_root: str | Path = "data/raw",
    work_root: str | Path = "data/work",
    publish_root: str | Path = "data/outputs/knowledge",
    index_database: str | Path = "data/index/knowledge.db",
    language: str = "auto",
    output_language: str = "source",
    terms: Iterable[str] = (),
    force_stages: Iterable[str] = (),
    chunk_seconds: float = 120.0,
    context_seconds: float = 5.0,
    asr_provider: str | None = None,
    text_provider: str | None = None,
    asr_model: str | None = None,
    proofreading_model: str | None = None,
    chapter_model: str | None = None,
    knowledge_model: str | None = None,
    local_asr_device: str | None = None,
    local_asr_compute_type: str | None = None,
    reasoning_effort: str = "none",
    media_loader: Callable[..., Any] = prepare_local_audio,
    transcriber: Callable[..., Any] = transcribe_audio_groq_chunked,
    local_transcriber: Callable[..., Any] = transcribe_audio,
    proofreader: Callable[..., Any] = proofread_transcript,
    asr_reviewer: Callable[..., Any] = review_flagged_segments_groq,
    chapterer: Callable[..., Any] = extract_chapters,
    knowledge_extractor: Callable[..., Any] = extract_structured_knowledge,
    local_text_client: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
) -> PipelineArtifact:
    """Run or resume local audio extraction and the knowledge pipeline."""
    progress = progress_callback
    if progress is None and show_progress:
        progress = ConsoleProgress()
    _emit_progress(progress, "pipeline", "starting", 0)
    selected_asr_provider = (asr_provider or project_secret("ASR_PROVIDER") or "groq").strip().lower()
    selected_text_provider = (text_provider or project_secret("TEXT_PROVIDER") or "deepseek").strip().lower()
    if selected_asr_provider not in {"groq", "local"}:
        raise ValueError("ASR_PROVIDER must be 'groq' or 'local'")
    if selected_text_provider not in {"deepseek", "local"}:
        raise ValueError("TEXT_PROVIDER must be 'deepseek' or 'local'")
    selected_asr_model = asr_model or project_secret(
        "GROQ_MODEL" if selected_asr_provider == "groq" else "LOCAL_ASR_MODEL"
    ) or ("whisper-large-v3-turbo" if selected_asr_provider == "groq" else "small")
    if selected_text_provider == "local":
        from .local_text import local_text_base_url, local_text_model
        selected_text_model = local_text_model()
        selected_text_endpoint = local_text_base_url()
    else:
        selected_text_model = project_secret("DEEPSEEK_MODEL") or "deepseek-flash"
        selected_text_endpoint = "https://api.deepseek.com"
    selected_proofreading_model = proofreading_model or selected_text_model
    selected_chapter_model = chapter_model or selected_text_model
    selected_knowledge_model = knowledge_model or selected_text_model
    selected_local_device = (local_asr_device or project_secret("LOCAL_ASR_DEVICE") or "cpu").strip().lower()
    selected_local_compute = (local_asr_compute_type or project_secret("LOCAL_ASR_COMPUTE_TYPE") or (
        "float16" if selected_local_device == "cuda" else "int8"
    )).strip()
    if selected_local_device not in {"cpu", "cuda"}:
        raise ValueError("LOCAL_ASR_DEVICE must be 'cpu' or 'cuda'")
    local_client = local_text_client

    def get_local_client() -> Any:
        nonlocal local_client
        if local_client is None:
            from .local_text import load_local_text_client
            local_client = load_local_text_client()
        return local_client

    raw_dir = Path(raw_root).resolve()
    work_dir = Path(work_root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    requested_language = normalize_language(language)
    requested_output_language = normalize_output_language(output_language)
    user_terms = tuple(term.strip() for term in terms if term.strip())
    forced = {str(stage).strip().lower() for stage in force_stages if str(stage).strip()}
    unknown_forced = forced - {"transcribe", "proofread", "asr_review", "chapters", "knowledge"}
    if unknown_forced:
        raise ValueError(f"unsupported forced stages: {sorted(unknown_forced)}")
    executed: list[str] = []
    reused: list[str] = []
    previous_settings: dict[str, Any] = {}
    _emit_progress(progress, "extract_audio", "running", 0)
    try:
        media = media_loader(video_file, raw_dir)
    except Exception as exc:
        _emit_progress(progress, "extract_audio", "failed", 0, str(exc))
        raise PipelineError(f"local audio extraction failed: {exc}") from exc
    video_id = _safe_video_id(str(media.video_id))
    source_path = Path(getattr(media, "source_path", None) or media.video_path).resolve()
    video_path = Path(media.video_path).resolve() if media.video_path is not None else None
    audio_path = Path(media.file_path).resolve()
    run_dir = work_dir / video_id
    manifest_path = run_dir / "pipeline.json"
    existing = _read_json(manifest_path)
    state = existing or {
        "version": 2,
        "requested_file": str(source_path),
        "source_has_video": video_path is not None,
        "source_url": str(media.source_url),
        "video_id": video_id,
        "title": str(media.title),
        "stages": {},
    }
    if state.get("requested_file") != str(source_path):
        raise PipelineError("media ID is associated with a different local file")
    state["source_has_video"] = video_path is not None
    raw_settings = state.get("settings")
    if isinstance(raw_settings, dict):
        previous_settings = raw_settings
    stage = state.setdefault("stages", {}).get("extract_audio", {})
    if existing and stage.get("status") == "completed" and audio_path.is_file():
        reused.append("extract_audio")
        _emit_progress(progress, "extract_audio", "reused", 10, video_id)
    else:
        executed.append("extract_audio")
        _emit_progress(progress, "extract_audio", "completed", 10, video_id)
    state["stages"]["extract_audio"] = {
        "status": "completed", "audio_path": str(audio_path),
        "source_path": str(source_path),
        "video_path": str(video_path) if video_path is not None else None,
    }
    _write_manifest(manifest_path, state)

    metadata_terms = extract_metadata_terms(audio_path)
    preferred_terms = tuple(merge_terms(user_terms, metadata_terms))

    transcript_path = _completed_file(state, "transcribe", "json_path")
    conventional_transcript = run_dir / "groq" / "transcript.json"
    if transcript_path is None and selected_asr_provider == "groq" and conventional_transcript.is_file():
        transcript_path = conventional_transcript.resolve()
    force_transcribe = (
        "transcribe" in forced
        or (existing is not None and (
            str(previous_settings.get("asr_provider") or "groq") != selected_asr_provider
            or str(previous_settings.get("asr_model") or "whisper-large-v3-turbo") != selected_asr_model
            or (selected_asr_provider == "local" and (
                str(previous_settings.get("local_asr_device") or "cpu") != selected_local_device
                or str(previous_settings.get("local_asr_compute_type") or "int8") != selected_local_compute
            ))
        ))
    )
    if force_transcribe:
        transcript_path = None
    if transcript_path is not None and requested_language != "auto":
        existing_language = str((_read_json(transcript_path) or {}).get("language") or "")
        try:
            normalized_existing_language = normalize_language(existing_language)
        except ValueError:
            normalized_existing_language = "unknown"
        if normalized_existing_language != requested_language:
            transcript_path = None
            force_transcribe = True
    _emit_progress(progress, "transcribe", "running", 10)
    if transcript_path is None:
        try:
            if selected_asr_provider == "groq":
                transcript = transcriber(
                    audio_path,
                    run_dir / "groq",
                    model_name=selected_asr_model,
                    language=requested_language,
                    prompt=build_asr_prompt(preferred_terms),
                    core_seconds=chunk_seconds,
                    context_seconds=context_seconds,
                )
            else:
                transcript = local_transcriber(
                    audio_path,
                    run_dir / "local_asr",
                    model_name=selected_asr_model,
                    language=None if requested_language == "auto" else requested_language,
                    device=selected_local_device,
                    compute_type=selected_local_compute,
                )
        except Exception as exc:
            _emit_progress(progress, "transcribe", "failed", 10, str(exc))
            raise PipelineError(f"transcribe stage failed: {exc}") from exc
        transcript_path = Path(transcript.json_path).resolve()
        executed.append("transcribe")
        _emit_progress(progress, "transcribe", "completed", 35)
    else:
        reused.append("transcribe")
        _emit_progress(progress, "transcribe", "reused", 35)
    state["stages"]["transcribe"] = {
        "status": "completed",
        "json_path": str(transcript_path),
        "requested_language": requested_language,
        "provider": selected_asr_provider,
        "model": selected_asr_model,
    }
    transcript_payload = _read_json(transcript_path) or {}
    source_language = str(
        transcript_payload.get("language")
        or (requested_language if requested_language != "auto" else "unknown")
    )
    normalized_output_language = resolve_output_language(
        requested_output_language, source_language
    )
    state["stages"]["transcribe"]["detected_language"] = source_language
    state["settings"] = {
        "requested_language": requested_language,
        "source_language": source_language,
        "requested_output_language": requested_output_language,
        "output_language": normalized_output_language,
        "user_terms": list(user_terms),
        "metadata_terms": metadata_terms,
        "preferred_terms": list(preferred_terms),
        "asr_provider": selected_asr_provider,
        "asr_model": selected_asr_model,
        "local_asr_device": selected_local_device,
        "local_asr_compute_type": selected_local_compute,
        "text_provider": selected_text_provider,
        "text_endpoint": selected_text_endpoint,
        "proofreading_model": selected_proofreading_model,
        "chapter_model": selected_chapter_model,
        "knowledge_model": selected_knowledge_model,
    }
    _write_manifest(manifest_path, state)

    force_proofread = (
        force_transcribe or "proofread" in forced
        or (existing is not None and (
            str(previous_settings.get("text_provider") or "deepseek") != selected_text_provider
            or str(previous_settings.get("proofreading_model") or "deepseek-flash") != selected_proofreading_model
            or str(previous_settings.get("text_endpoint") or "https://api.deepseek.com") != selected_text_endpoint
        ))
    )
    proofread_path = None if force_proofread else _completed_file(state, "proofread", "json_path")
    conventional_proofread = run_dir / "proofread" / "proofread.json"
    if proofread_path is None and not force_proofread and conventional_proofread.is_file():
        proofread_path = conventional_proofread.resolve()
    _emit_progress(progress, "proofread", "running", 35)
    if proofread_path is None:
        try:
            kwargs = {
                "provider": selected_text_provider,
                "model": selected_proofreading_model,
                "reasoning_effort": reasoning_effort,
                "terms": preferred_terms,
            }
            if selected_text_provider == "local":
                kwargs["client"] = get_local_client()
            proofread = proofreader(transcript_path, run_dir / "proofread", **kwargs)
        except Exception as exc:
            _emit_progress(progress, "proofread", "failed", 35, str(exc))
            raise PipelineError(f"proofread stage failed: {exc}") from exc
        proofread_path = Path(proofread.json_path).resolve()
        executed.append("proofread")
        _emit_progress(progress, "proofread", "completed", 55)
    else:
        reused.append("proofread")
        _emit_progress(progress, "proofread", "reused", 55)
    proofread_usage, proofread_cost = _billing_from_output(proofread_path)
    state["stages"]["proofread"] = {
        "status": "completed",
        "json_path": str(proofread_path),
        "reasoning_effort": (_read_json(proofread_path) or {}).get("reasoning_effort"),
        "usage": proofread_usage,
        "cost_estimate": proofread_cost,
        "preferred_terms": list(preferred_terms),
        "review_queue_path": str(Path(proofread_path).with_name("review_queue.json")),
    }
    _write_manifest(manifest_path, state)

    try:
        normalized_source_language = normalize_language(source_language)
    except ValueError:
        normalized_source_language = "unknown"
    asr_review_path: Path | None = None
    asr_review_audit_path: Path | None = None
    review_executed = False
    if normalized_source_language == "ja" and selected_asr_provider == "groq":
        _emit_progress(progress, "asr_review", "running", 55)
        force_asr_review = force_proofread or "asr_review" in forced
        asr_review_path = None if force_asr_review else _completed_file(
            state, "asr_review", "json_path"
        )
        conventional_review = run_dir / "asr_review" / "proofread_reviewed.json"
        if asr_review_path is None and not force_asr_review and conventional_review.is_file():
            asr_review_path = conventional_review.resolve()
        if asr_review_path is None:
            try:
                review = asr_reviewer(
                    audio_path,
                    proofread_path,
                    run_dir / "asr_review",
                    model_name="whisper-large-v3",
                    language="ja",
                    terms=preferred_terms,
                )
            except Exception as exc:
                _emit_progress(progress, "asr_review", "failed", 55, str(exc))
                raise PipelineError(f"selective ASR review stage failed: {exc}") from exc
            asr_review_path = Path(review.json_path).resolve()
            asr_review_audit_path = Path(review.audit_path).resolve()
            state["stages"]["asr_review"] = {
                "status": "completed",
                "json_path": str(asr_review_path),
                "audit_path": str(asr_review_audit_path),
                "model": review.model,
                "candidate_count": review.candidate_count,
                "replacement_count": review.replacement_count,
                "audio_seconds": review.audio_seconds,
                "billed_audio_seconds_estimate": review.billed_audio_seconds_estimate,
            }
            executed.append("asr_review")
            review_executed = True
            _write_manifest(manifest_path, state)
            _emit_progress(progress, "asr_review", "completed", 62)
        else:
            raw_review_stage = state.get("stages", {}).get("asr_review", {})
            audit_value = raw_review_stage.get("audit_path")
            if audit_value:
                candidate = Path(str(audit_value)).resolve()
                asr_review_audit_path = candidate if candidate.is_file() else None
            reused.append("asr_review")
            _emit_progress(progress, "asr_review", "reused", 62)
    else:
        _emit_progress(
            progress,
            "asr_review",
            "skipped",
            62,
            ("local ASR mode" if selected_asr_provider == "local"
             else f"source language: {normalized_source_language}"),
        )

    text_input_path = asr_review_path or proofread_path

    previous_output_language = str(previous_settings.get("output_language") or "")
    conventional_chapters = run_dir / "chapters" / "chapters.json"
    if not previous_output_language and conventional_chapters.is_file():
        previous_output_language = str(
            (_read_json(conventional_chapters) or {}).get("output_language") or "zh"
        )
    force_output_language = bool(
        previous_output_language
        and previous_output_language != normalized_output_language
    )
    force_chapters = (
        force_proofread or review_executed or force_output_language or "chapters" in forced
        or (existing is not None and str(previous_settings.get("chapter_model") or "deepseek-flash")
            != selected_chapter_model)
    )
    chapters_path = None if force_chapters else _completed_file(state, "chapters", "json_path")
    if chapters_path is None and not force_chapters and conventional_chapters.is_file():
        chapters_path = conventional_chapters.resolve()
    _emit_progress(progress, "chapters", "running", 62)
    if chapters_path is None:
        try:
            kwargs = {
                "model": selected_chapter_model,
                "reasoning_effort": reasoning_effort,
                "output_language": normalized_output_language,
            }
            if selected_text_provider == "local":
                kwargs.update(provider="local", client=get_local_client())
            chapters = chapterer(text_input_path, run_dir / "chapters", **kwargs)
        except Exception as exc:
            _emit_progress(progress, "chapters", "failed", 62, str(exc))
            raise PipelineError(f"chapters stage failed: {exc}") from exc
        chapters_path = Path(chapters.json_path).resolve()
        executed.append("chapters")
        _emit_progress(progress, "chapters", "completed", 74)
    else:
        reused.append("chapters")
        _emit_progress(progress, "chapters", "reused", 74)
    chapter_usage, chapter_cost = _billing_from_output(chapters_path)
    state["stages"]["chapters"] = {
        "status": "completed",
        "json_path": str(chapters_path),
        "reasoning_effort": (_read_json(chapters_path) or {}).get("reasoning_effort"),
        "usage": chapter_usage,
        "cost_estimate": chapter_cost,
    }
    _write_manifest(manifest_path, state)
    _emit_progress(progress, "chapter_export", "running", 74)
    try:
        chapter_sections_dir, section_paths = export_chapter_sections(chapters_path)
    except Exception as exc:
        _emit_progress(progress, "chapter_export", "failed", 74, str(exc))
        raise PipelineError(f"chapter export stage failed: {exc}") from exc
    state["stages"]["chapter_export"] = {
        "status": "completed",
        "sections_dir": str(chapter_sections_dir),
        "file_count": len(section_paths),
    }
    _write_manifest(manifest_path, state)
    _emit_progress(
        progress,
        "chapter_export",
        "completed",
        78,
        f"{len(section_paths)} files",
    )

    _emit_progress(progress, "moments", "running", 78)
    try:
        moment_artifact = build_moments(chapters_path, run_dir / "moments", video_id)
    except Exception as exc:
        _emit_progress(progress, "moments", "failed", 78, str(exc))
        raise PipelineError(f"moment stage failed: {exc}") from exc
    moments_path = moment_artifact.json_path
    state["stages"]["moments"] = {
        "status": "completed",
        "json_path": str(moments_path),
        "moment_count": moment_artifact.moment_count,
        "segment_count": moment_artifact.segment_count,
    }
    (reused if moment_artifact.reused else executed).append("moments")
    _write_manifest(manifest_path, state)
    _emit_progress(progress, "moments", "reused" if moment_artifact.reused else "completed", 78,
                   f"{moment_artifact.moment_count} moments")

    force_knowledge = (
        force_chapters or "knowledge" in forced
        or (existing is not None and str(previous_settings.get("knowledge_model") or "deepseek-flash")
            != selected_knowledge_model)
    )
    structured_knowledge_path = None if force_knowledge else _completed_file(state, "knowledge", "json_path")
    conventional_knowledge = run_dir / "knowledge" / "knowledge.json"
    if structured_knowledge_path is None and not force_knowledge and conventional_knowledge.is_file():
        structured_knowledge_path = conventional_knowledge.resolve()
    _emit_progress(progress, "knowledge", "running", 78)
    if structured_knowledge_path is None:
        try:
            kwargs = {
                "model": selected_knowledge_model,
                "reasoning_effort": reasoning_effort,
                "output_language": normalized_output_language,
            }
            if selected_text_provider == "local":
                kwargs.update(provider="local", client=get_local_client())
            knowledge = knowledge_extractor(chapters_path, run_dir / "knowledge", **kwargs)
        except Exception as exc:
            _emit_progress(progress, "knowledge", "failed", 78, str(exc))
            raise PipelineError(f"knowledge stage failed: {exc}") from exc
        structured_knowledge_path = Path(knowledge.json_path).resolve()
        executed.append("knowledge")
        _emit_progress(progress, "knowledge", "completed", 90)
    else:
        reused.append("knowledge")
        _emit_progress(progress, "knowledge", "reused", 90)
    knowledge_usage, knowledge_cost = _billing_from_output(structured_knowledge_path)
    state["stages"]["knowledge"] = {
        "status": "completed",
        "json_path": str(structured_knowledge_path),
        "reasoning_effort": (_read_json(structured_knowledge_path) or {}).get("reasoning_effort"),
        "usage": knowledge_usage,
        "cost_estimate": knowledge_cost,
    }
    _write_manifest(manifest_path, state)

    _emit_progress(progress, "publish", "running", 90)
    try:
        published = publish_knowledge(
            chapters_path,
            chapter_sections_dir,
            publish_root,
            video_id,
            structured_knowledge_path,
            moments_path,
        )
    except Exception as exc:
        _emit_progress(progress, "publish", "failed", 90, str(exc))
        raise PipelineError(f"knowledge publish stage failed: {exc}") from exc
    state["stages"]["publish"] = {
        "status": "completed",
        "knowledge_dir": str(published.knowledge_dir),
        "metadata_path": str(published.metadata_path),
        "chapter_count": published.chapter_count,
        "moment_count": published.moment_count,
    }
    _write_manifest(manifest_path, state)
    _emit_progress(progress, "publish", "completed", 96)
    _emit_progress(progress, "index", "running", 96)
    try:
        index_artifact = build_index(publish_root, index_database)
    except Exception as exc:
        _emit_progress(progress, "index", "failed", 96, str(exc))
        raise PipelineError(f"index stage failed: {exc}") from exc
    state["stages"]["index"] = {
        "status": "completed",
        "database_path": str(index_artifact.database_path),
        "chapter_count": index_artifact.chapter_count,
        "knowledge_count": index_artifact.knowledge_count,
        "moment_count": index_artifact.moment_count,
    }
    _emit_progress(
        progress,
        "index",
        "completed",
        100,
        f"{index_artifact.chapter_count} chapters",
    )
    total_usage = empty_usage()
    total_cost = 0.0
    run_cost = 0.0
    complete = True
    periods: set[str] = set()
    for stage_name in ("proofread", "chapters", "knowledge"):
        stage = state["stages"].get(stage_name, {})
        usage = stage.get("usage")
        estimate = stage.get("cost_estimate")
        if not isinstance(usage, dict) or not isinstance(estimate, dict):
            complete = False
            continue
        add_usage(total_usage, usage)
        stage_cost = float(estimate.get("estimated_cost_cny", 0.0) or 0.0)
        total_cost += stage_cost
        if stage_name in executed:
            run_cost += stage_cost
        period = str(estimate.get("pricing_period") or "")
        if period:
            periods.add(period)
    state["cost_summary"] = {
        "usage": total_usage,
        "estimated_artifact_cost_cny": round(total_cost, 6),
        "estimated_run_cost_cny": round(run_cost, 6),
        "complete": complete,
        "pricing_periods": sorted(periods),
        "note": "output_tokens already include reasoning_tokens",
    }
    _write_manifest(manifest_path, state)
    _emit_progress(progress, "pipeline", "completed", 100, video_id)

    return PipelineArtifact(
        video_id=video_id,
        title=str(state.get("title") or ""),
        source_url=str(state.get("source_url") or media.source_url),
        source_path=source_path,
        video_path=video_path,
        source_language=source_language,
        output_language=normalized_output_language,
        run_dir=run_dir,
        manifest_path=manifest_path,
        audio_path=audio_path,
        transcript_path=transcript_path,
        proofread_path=proofread_path,
        asr_review_path=asr_review_path,
        asr_review_audit_path=asr_review_audit_path,
        chapters_path=chapters_path,
        chapter_sections_dir=chapter_sections_dir,
        moments_path=moments_path,
        knowledge_dir=published.knowledge_dir,
        knowledge_metadata_path=published.metadata_path,
        structured_knowledge_path=structured_knowledge_path,
        index_database_path=index_artifact.database_path,
        executed_stages=tuple(executed),
        reused_stages=tuple(reused),
        deepseek_usage=total_usage,
        estimated_deepseek_cost_cny=round(total_cost, 6),
        estimated_run_cost_cny=round(run_cost, 6),
        cost_estimate_complete=complete,
        pricing_periods=tuple(sorted(periods)),
    )
