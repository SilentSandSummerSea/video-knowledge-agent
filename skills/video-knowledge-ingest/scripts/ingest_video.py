"""Run the local-video knowledge pipeline from this repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_knowledge_agent.pipeline import process_video  # noqa: E402
from video_knowledge_agent.project_env import project_secret  # noqa: E402


def _load_windows_user_variable(name: str) -> None:
    if os.environ.get(name) or sys.platform != "win32":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
        if value:
            os.environ[name] = str(value)
    except OSError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", type=Path)
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output-language", default="source")
    parser.add_argument(
        "--force-stage", action="append",
        choices=("transcribe", "proofread", "asr_review", "chapters", "knowledge"),
        default=[],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    video = args.video_file.expanduser().resolve()
    if not video.is_file():
        parser.error(f"local video file does not exist: {video}")
    _load_windows_user_variable("GROQ_API_KEY")
    _load_windows_user_variable("DEEPSEEK_API_KEY")
    if args.dry_run:
        asr_provider = project_secret("ASR_PROVIDER") or "groq"
        text_provider = project_secret("TEXT_PROVIDER") or "deepseek"
        print(json.dumps({
            "video_file": str(video), "language": args.language,
            "output_language": args.output_language,
            "asr_provider": asr_provider,
            "text_provider": text_provider,
            "groq_configured": bool(project_secret("GROQ_API_KEY")),
            "deepseek_configured": bool(project_secret("DEEPSEEK_API_KEY")),
            "local_text_model_configured": bool(project_secret("LOCAL_TEXT_MODEL")),
            "stages": ["extract_audio", "transcribe", "proofread", "asr_review",
                       "chapters", "moments", "knowledge", "publish", "index"],
        }, ensure_ascii=False, indent=2))
        return 0
    artifact = process_video(
        video, raw_root=PROJECT_ROOT / "data/raw", work_root=PROJECT_ROOT / "data/work",
        publish_root=PROJECT_ROOT / "data/outputs/knowledge",
        index_database=PROJECT_ROOT / "data/index/knowledge.db",
        terms=args.term, language=args.language,
        output_language=args.output_language, force_stages=args.force_stage,
    )
    print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
