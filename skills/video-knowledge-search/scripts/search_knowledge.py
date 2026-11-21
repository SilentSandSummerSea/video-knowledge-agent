"""Query the local video knowledge index for the Codex skill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
KNOWLEDGE_ROOT = PROJECT_ROOT / "data" / "outputs" / "knowledge"
DATABASE_PATH = PROJECT_ROOT / "data" / "index" / "knowledge.db"

sys.path.insert(0, str(SOURCE_ROOT))

from video_knowledge_agent.indexing import build_index  # noqa: E402
from video_knowledge_agent.retrieval import search_with_context  # noqa: E402


def _index_is_stale() -> bool:
    if not DATABASE_PATH.is_file():
        return True
    database_mtime = DATABASE_PATH.stat().st_mtime
    for pattern in ("*/metadata.json", "*/structured_knowledge.json", "*/moments.json", "*/chapters/*.md"):
        if any(path.stat().st_mtime > database_mtime for path in KNOWLEDGE_ROOT.glob(pattern)):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--context-seconds", type=float, default=15.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rebuilt = args.rebuild or _index_is_stale()
    if rebuilt:
        build_index(KNOWLEDGE_ROOT, DATABASE_PATH)
    results = search_with_context(
        args.query,
        DATABASE_PATH,
        PROJECT_ROOT / "data" / "work",
        KNOWLEDGE_ROOT,
        limit=args.limit,
        context_seconds=args.context_seconds,
    )
    compact = []
    for result in results:
        item = result
        content = str(item["content"])
        item["content"] = content if len(content) <= 1800 else content[:1800] + "…"
        compact.append(item)
    print(
        json.dumps(
            {
                "query": args.query,
                "index_rebuilt": rebuilt,
                "result_count": len(compact),
                "results": compact,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
