"""Small utility entry point for the project foundation."""

from __future__ import annotations

import argparse
import json
import sys

from .answering import AnsweringError, answer_question
from .audio import LocalMediaError, prepare_local_audio
from .chaptering import ChapteringError, extract_chapters
from .groq_transcription import transcribe_audio_groq_chunked
from .models import timestamp_url
from .knowledge import KnowledgeExtractionError, extract_structured_knowledge
from .indexing import IndexingError, build_index, search_index
from .moments import MomentError, backfill_moments
from .pipeline import PipelineError, process_video
from .proofreading import ProofreadingError, proofread_transcript
from .project_env import project_secret
from .transcription import TranscriptionError, transcribe_audio
from .video_clips import VideoClipError, extract_moment_video
from .visual_ocr import VisualOcrError, extract_moment_ocr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-knowledge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    link = subparsers.add_parser("timestamp-link", help="Create a timestamped source URL")
    link.add_argument("url")
    link.add_argument("seconds", type=float)

    audio = subparsers.add_parser(
        "prepare-audio", help="Normalize audio from a local video or audio file"
    )
    audio.add_argument("media_file")
    audio.add_argument("--output-dir", default="data/raw")
    audio.add_argument("--ffmpeg-path")

    transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe an audio file with Groq or local Whisper"
    )
    transcribe.add_argument("audio_file")
    transcribe.add_argument("--output-dir", default="data/work/transcript")
    transcribe.add_argument("--provider", choices=("groq", "local"), default=None)
    transcribe.add_argument("--model", default=None)
    transcribe.add_argument(
        "--language",
        default="auto",
        help="Input language code such as zh, en, or ja; default: auto-detect",
    )
    transcribe.add_argument("--prompt", default=None)
    transcribe.add_argument("--chunk-seconds", type=float, default=120.0)
    transcribe.add_argument("--context-seconds", type=float, default=5.0)
    transcribe.add_argument("--device", choices=("cuda", "cpu"), default=None)
    transcribe.add_argument("--compute-type", default=None)

    proofread = subparsers.add_parser(
        "proofread", help="Proofread transcript JSON with DeepSeek, Codex, or OpenAI"
    )
    proofread.add_argument("transcript_file")
    proofread.add_argument("--output-dir", default="data/work/proofread")
    proofread.add_argument(
        "--provider",
        choices=("deepseek", "codex", "openai", "local"),
        default=None,
    )
    proofread.add_argument("--model", default=None)
    proofread.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max", "ultra"),
        default="none",
    )
    proofread.add_argument("--batch-seconds", type=float, default=300.0)
    proofread.add_argument("--context-segments", type=int, default=3)
    proofread.add_argument(
        "--term",
        action="append",
        default=[],
        help="Preferred spelling for a domain term; may be repeated",
    )

    chapters = subparsers.add_parser(
        "chapters", help="Split a timestamped transcript into validated topic chapters"
    )
    chapters.add_argument("transcript_file")
    chapters.add_argument("--output-dir", default="data/work/chapters")
    chapters.add_argument("--model", default=None)
    chapters.add_argument("--provider", choices=("deepseek", "local"), default=None)
    chapters.add_argument("--output-language", default="source")
    chapters.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "high", "max"),
        default="none",
    )

    process = subparsers.add_parser(
        "process", help="Run or resume the local-media knowledge pipeline"
    )
    process.add_argument("media_file")
    process.add_argument("--raw-dir", default="data/raw")
    process.add_argument("--work-dir", default="data/work")
    process.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    process.add_argument("--index-database", default="data/index/knowledge.db")
    process.add_argument(
        "--language",
        default="auto",
        help="Input language code such as zh, en, or ja; default: auto-detect",
    )
    process.add_argument("--output-language", default="source")
    process.add_argument("--asr-provider", choices=("groq", "local"), default=None)
    process.add_argument("--text-provider", choices=("deepseek", "local"), default=None)
    process.add_argument("--chunk-seconds", type=float, default=120.0)
    process.add_argument("--context-seconds", type=float, default=5.0)
    process.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "high", "max"),
        default="none",
        help="DeepSeek reasoning mode for text stages; none is the low-cost default",
    )
    process.add_argument(
        "--term",
        action="append",
        default=[],
        help="Preferred spelling for a domain term; may be repeated",
    )
    process.add_argument(
        "--force-stage",
        action="append",
        choices=("transcribe", "proofread", "asr_review", "chapters", "knowledge"),
        default=[],
        help="Re-run this stage and its downstream dependents; may be repeated",
    )

    knowledge = subparsers.add_parser(
        "knowledge", help="Extract evidence-grounded structured knowledge by chapter"
    )
    knowledge.add_argument("chapters_file")
    knowledge.add_argument("--output-dir", default="data/work/knowledge")
    knowledge.add_argument("--model", default=None)
    knowledge.add_argument("--provider", choices=("deepseek", "local"), default=None)
    knowledge.add_argument("--output-language", default=None)
    knowledge.add_argument(
        "--reasoning-effort", choices=("none", "low", "high", "max"), default="none"
    )

    index = subparsers.add_parser(
        "index", help="Build a local SQLite FTS5 index from published knowledge"
    )
    index.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    index.add_argument("--database", default="data/index/knowledge.db")

    backfill = subparsers.add_parser(
        "moments-backfill", help="Create and index local time windows for published videos"
    )
    backfill.add_argument("--work-dir", default="data/work")
    backfill.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    backfill.add_argument("--database", default="data/index/knowledge.db")

    clip = subparsers.add_parser(
        "clip-moment-video", help="Extract a local video range around a published Moment"
    )
    clip.add_argument("video_id")
    target = clip.add_mutually_exclusive_group(required=True)
    target.add_argument("--moment-id")
    target.add_argument("--at", type=float, dest="at_seconds")
    clip.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    clip.add_argument("--work-dir", default="data/work")
    clip.add_argument("--output-dir", default="data/outputs/clips")
    clip.add_argument("--padding-seconds", type=float, default=5.0)
    clip.add_argument("--max-height", type=int, default=720)
    clip.add_argument("--ffmpeg-path")

    visual = subparsers.add_parser(
        "ocr-moment", help="Extract keyframes and local OCR from a cached Moment clip"
    )
    visual.add_argument("clip_file")
    visual.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    visual.add_argument("--sample-every-seconds", type=float, default=3.0)
    visual.add_argument("--max-frames", type=int, default=12)
    visual.add_argument("--min-confidence", type=float, default=0.35)

    search = subparsers.add_parser("search", help="Search the local knowledge index")
    search.add_argument("query")
    search.add_argument("--database", default="data/index/knowledge.db")
    search.add_argument("--limit", type=int, default=8)

    answer = subparsers.add_parser(
        "answer", help="Answer from indexed speech and on-demand visual evidence"
    )
    answer.add_argument("question")
    answer.add_argument("--database", default="data/index/knowledge.db")
    answer.add_argument("--work-dir", default="data/work")
    answer.add_argument("--knowledge-dir", default="data/outputs/knowledge")
    answer.add_argument("--clips-dir", default="data/outputs/clips")
    answer.add_argument("--text-provider", choices=("deepseek", "local"), default=None)
    answer.add_argument("--model", default=None)
    answer.add_argument("--visual-mode", choices=("auto", "off", "ocr", "frames"), default="auto")
    answer.add_argument("--answer-language", choices=("zh", "en", "ja"), default="zh")
    answer.add_argument("--limit", type=int, default=4)
    answer.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.command == "timestamp-link":
        print(timestamp_url(args.url, args.seconds))
        return 0
    if args.command == "prepare-audio":
        try:
            artifact = prepare_local_audio(args.media_file, args.output_dir, ffmpeg_path=args.ffmpeg_path)
        except (LocalMediaError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "transcribe":
        try:
            selected_provider = args.provider or project_secret("ASR_PROVIDER") or "groq"
            if selected_provider == "groq":
                artifact = transcribe_audio_groq_chunked(
                    args.audio_file,
                    args.output_dir,
                    model_name=args.model or project_secret("GROQ_MODEL") or "whisper-large-v3-turbo",
                    language=args.language,
                    prompt=args.prompt,
                    core_seconds=args.chunk_seconds,
                    context_seconds=args.context_seconds,
                )
            elif selected_provider == "local":
                device = args.device or project_secret("LOCAL_ASR_DEVICE") or "cpu"
                compute_type = args.compute_type or (
                    project_secret("LOCAL_ASR_COMPUTE_TYPE") or
                    ("float16" if device == "cuda" else "int8")
                )
                artifact = transcribe_audio(
                    args.audio_file,
                    args.output_dir,
                    model_name=args.model or project_secret("LOCAL_ASR_MODEL") or "small",
                    language=None if args.language == "auto" else args.language,
                    device=device,
                    compute_type=compute_type,
                )
            else:
                raise ValueError("ASR_PROVIDER must be 'groq' or 'local'")
        except (TranscriptionError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "proofread":
        try:
            selected_provider = args.provider or project_secret("TEXT_PROVIDER") or "deepseek"
            artifact = proofread_transcript(
                args.transcript_file,
                args.output_dir,
                provider=selected_provider,
                model=args.model or (project_secret("DEEPSEEK_MODEL") if selected_provider == "deepseek" else None),
                reasoning_effort=args.reasoning_effort,
                batch_seconds=args.batch_seconds,
                context_segments=args.context_segments,
                terms=args.term,
            )
        except (ProofreadingError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "chapters":
        try:
            selected_provider = args.provider or project_secret("TEXT_PROVIDER") or "deepseek"
            artifact = extract_chapters(
                args.transcript_file,
                args.output_dir,
                provider=selected_provider,
                model=args.model or (project_secret("DEEPSEEK_MODEL") if selected_provider == "deepseek" else None),
                reasoning_effort=args.reasoning_effort,
                output_language=args.output_language,
            )
        except (ChapteringError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "process":
        try:
            artifact = process_video(
                args.media_file,
                raw_root=args.raw_dir,
                work_root=args.work_dir,
                publish_root=args.knowledge_dir,
                index_database=args.index_database,
                language=args.language,
                output_language=args.output_language,
                asr_provider=args.asr_provider,
                text_provider=args.text_provider,
                terms=args.term,
                chunk_seconds=args.chunk_seconds,
                context_seconds=args.context_seconds,
                reasoning_effort=args.reasoning_effort,
                force_stages=args.force_stage,
            )
        except (PipelineError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge":
        try:
            selected_provider = args.provider or project_secret("TEXT_PROVIDER") or "deepseek"
            artifact = extract_structured_knowledge(
                args.chapters_file,
                args.output_dir,
                provider=selected_provider,
                model=args.model or (project_secret("DEEPSEEK_MODEL") if selected_provider == "deepseek" else None),
                reasoning_effort=args.reasoning_effort,
                output_language=args.output_language,
            )
        except (KnowledgeExtractionError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "index":
        try:
            artifact = build_index(args.knowledge_dir, args.database)
        except (IndexingError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "moments-backfill":
        try:
            report = backfill_moments(args.work_dir, args.knowledge_dir, args.database)
        except (MomentError, IndexingError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "clip-moment-video":
        try:
            artifact = extract_moment_video(
                args.video_id,
                moment_id=args.moment_id,
                at_seconds=args.at_seconds,
                knowledge_root=args.knowledge_dir,
                work_root=args.work_dir,
                output_root=args.output_dir,
                padding_seconds=args.padding_seconds,
                max_height=args.max_height,
                ffmpeg_path=args.ffmpeg_path,
            )
        except (VideoClipError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ocr-moment":
        try:
            artifact = extract_moment_ocr(
                args.clip_file,
                knowledge_root=args.knowledge_dir,
                sample_every_seconds=args.sample_every_seconds,
                max_frames=args.max_frames,
                min_confidence=args.min_confidence,
            )
        except (VisualOcrError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "search":
        try:
            results = search_index(args.query, args.database, limit=args.limit)
        except (IndexingError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(
            json.dumps(
                [result.to_dict() for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "answer":
        try:
            result = answer_question(
                args.question, database_path=args.database,
                work_root=args.work_dir, knowledge_root=args.knowledge_dir,
                clips_root=args.clips_dir, visual_mode=args.visual_mode,
                answer_language=args.answer_language, limit=args.limit,
                text_provider=args.text_provider, model=args.model,
            )
        except (AnsweringError, IndexingError, VideoClipError, VisualOcrError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.format == "json"
              else result["answer_markdown"])
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
