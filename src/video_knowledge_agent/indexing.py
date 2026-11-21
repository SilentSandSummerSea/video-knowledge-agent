"""Local SQLite FTS5 indexing and retrieval for published video knowledge."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class IndexingError(RuntimeError):
    """Raised when the knowledge index cannot be built or queried."""


@dataclass(frozen=True, slots=True)
class IndexArtifact:
    database_path: Path
    video_count: int
    chapter_count: int
    knowledge_count: int
    moment_count: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["database_path"] = str(self.database_path)
        return result


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    video_id: str
    kind: str
    chapter_id: int
    title: str
    content: str
    source_url: str
    start_seconds: float
    end_seconds: float
    score: float
    file_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_CJK_RUN = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\uff66-\uff9f]+"
)
_WORD = re.compile(r"[A-Za-z0-9_]+")
_QUERY_EXPANSIONS = {
    "净赚": ("净利润", "利润", "收益"),
    "多少钱": ("金额", "金币", "费用", "收入", "利润"),
    "成本": ("费用", "支出", "成本"),
    "家属": ("家庭", "遗属", "家属"),
    "分配": ("分成", "份额", "分配"),
    "赚钱": ("利润", "收益", "收入"),
}


def _search_tokens(text: str) -> str:
    """Generate n-gram tokens that make CJK text searchable in stock FTS5."""
    tokens: list[str] = []
    for match in _CJK_RUN.finditer(text):
        value = match.group(0)
        tokens.extend(value)
        tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        tokens.extend(value[index : index + 3] for index in range(len(value) - 2))
    tokens.extend(word.lower() for word in _WORD.findall(text))
    return " ".join(dict.fromkeys(tokens))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexingError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IndexingError(f"expected a JSON object in {path}")
    return value


def _document_id(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def _chapter_documents(video_dir: Path, metadata: dict[str, Any]) -> Iterable[dict[str, Any]]:
    video_id = str(metadata.get("video_id") or video_dir.name)
    source_url = str(metadata.get("source_url") or "")
    for chapter in metadata.get("chapters", []):
        relative = str(chapter["file"])
        path = (video_dir / relative).resolve()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IndexingError(f"cannot read published chapter {path}: {exc}") from exc
        chapter_id = int(chapter["id"])
        title = str(chapter["title"])
        keywords = " ".join(map(str, chapter.get("topics", [])))
        yield {
            "id": _document_id(video_id, "chapter", chapter_id),
            "video_id": video_id,
            "kind": "chapter",
            "chapter_id": chapter_id,
            "title": title,
            "content": content,
            "keywords": keywords,
            "source_url": source_url,
            "start_seconds": float(chapter.get("start_seconds", 0)),
            "end_seconds": float(chapter.get("end_seconds", 0)),
            "file_path": str(path),
        }


def _knowledge_documents(video_dir: Path, metadata: dict[str, Any]) -> Iterable[dict[str, Any]]:
    relative = metadata.get("structured_knowledge_file")
    if not relative:
        return
    path = (video_dir / str(relative)).resolve()
    payload = _read_json(path)
    language = str(payload.get("output_language") or metadata.get("output_language") or "zh").casefold()
    evidence_label = "根拠：" if language in {"ja", "jpn", "japanese"} else "Evidence: " if language in {"en", "eng", "english"} else "证据："
    video_id = str(metadata.get("video_id") or video_dir.name)
    chapter_meta = {int(item["id"]): item for item in metadata.get("chapters", [])}
    for chapter in payload.get("chapters", []):
        chapter_id = int(chapter["chapter_id"])
        title = str(chapter.get("chapter_title") or chapter_meta.get(chapter_id, {}).get("title") or "")
        file_path = str((video_dir / chapter_meta.get(chapter_id, {}).get("file", "")).resolve())
        for item in chapter.get("knowledge_items", []):
            statement = str(item.get("statement") or "").strip()
            evidence = str(item.get("evidence_text") or "").strip()
            content = statement if not evidence else f"{statement}\n{evidence_label}{evidence}"
            item_id = str(item.get("id") or _document_id(statement))
            yield {
                "id": _document_id(video_id, "knowledge", item_id),
                "video_id": video_id,
                "kind": str(item.get("type") or "knowledge"),
                "chapter_id": chapter_id,
                "title": title,
                "content": content,
                "keywords": " ".join(map(str, item.get("keywords", []))),
                "source_url": str(item.get("source_url") or metadata.get("source_url") or ""),
                "start_seconds": float(item.get("start_seconds", 0)),
                "end_seconds": float(item.get("end_seconds", 0)),
                "file_path": file_path,
            }


def _moment_documents(video_dir: Path, metadata: dict[str, Any]) -> Iterable[dict[str, Any]]:
    relative = metadata.get("moments_file")
    if not relative:
        return
    path = (video_dir / str(relative)).resolve()
    payload = _read_json(path)
    video_id = str(metadata.get("video_id") or video_dir.name)
    if payload.get("video_id") != video_id or not isinstance(payload.get("moments"), list):
        raise IndexingError(f"invalid published moments in {path}")
    chapters = {int(item["id"]): item for item in metadata.get("chapters", [])}
    for item in payload["moments"]:
        chapter_id = int(item["chapter_id"])
        chapter = chapters.get(chapter_id)
        if chapter is None:
            raise IndexingError(f"moment references unknown chapter {chapter_id} in {path}")
        text = str(item.get("text") or "").strip()
        if not text:
            raise IndexingError(f"empty moment in {path}")
        yield {
            "id": _document_id(video_id, "moment", item["id"]),
            "video_id": video_id,
            "kind": "moment",
            "chapter_id": chapter_id,
            "title": str(chapter["title"]),
            "content": text,
            "keywords": " ".join(map(str, chapter.get("topics", []))),
            "source_url": str(item["source_url"]),
            "start_seconds": float(item["start_seconds"]),
            "end_seconds": float(item["end_seconds"]),
            "file_path": str((video_dir / str(chapter["file"])).resolve()),
        }


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def build_index(
    knowledge_root: str | Path = "data/outputs/knowledge",
    database_path: str | Path = "data/index/knowledge.db",
) -> IndexArtifact:
    """Rebuild the local index from the clean published knowledge tree."""
    root = Path(knowledge_root).resolve()
    if not root.is_dir():
        raise ValueError(f"knowledge directory does not exist: {root}")
    database = Path(database_path).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    video_count = 0
    chapter_count = 0
    knowledge_count = 0
    moment_count = 0
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = _read_json(metadata_path)
        video_dir = metadata_path.parent
        chapters = list(_chapter_documents(video_dir, metadata))
        knowledge = list(_knowledge_documents(video_dir, metadata))
        moments = list(_moment_documents(video_dir, metadata))
        documents.extend(chapters)
        documents.extend(knowledge)
        documents.extend(moments)
        video_count += 1
        chapter_count += len(chapters)
        knowledge_count += len(knowledge)
        moment_count += len(moments)
    if not documents:
        raise IndexingError(f"no indexable documents found under {root}")

    try:
        with closing(_connect(database)) as connection:
            with connection:
                connection.executescript(
                """
                DROP TABLE IF EXISTS search_index;
                DROP TABLE IF EXISTS documents;
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    chapter_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    file_path TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE search_index USING fts5(
                    id UNINDEXED,
                    title,
                    content,
                    keywords,
                    tokens,
                    tokenize='unicode61'
                );
                """
            )
                for document in documents:
                    connection.execute(
                    """INSERT INTO documents
                    (id, video_id, kind, chapter_id, title, content, keywords,
                     source_url, start_seconds, end_seconds, file_path)
                    VALUES (:id, :video_id, :kind, :chapter_id, :title, :content,
                            :keywords, :source_url, :start_seconds, :end_seconds, :file_path)""",
                    document,
                )
                    searchable = " ".join(
                        [document["title"], document["content"], document["keywords"]]
                    )
                    connection.execute(
                    "INSERT INTO search_index (id, title, content, keywords, tokens) VALUES (?, ?, ?, ?, ?)",
                    (
                        document["id"],
                        document["title"],
                        document["content"],
                        document["keywords"],
                        _search_tokens(searchable),
                    ),
                )
    except sqlite3.Error as exc:
        raise IndexingError(f"cannot build SQLite FTS5 index: {exc}") from exc
    return IndexArtifact(database, video_count, chapter_count, knowledge_count, moment_count)


def _match_query(query: str) -> str:
    tokens = _search_tokens(query).split()
    useful = [token for token in tokens if len(token) >= 2 or len(tokens) == 1]
    if not useful:
        useful = tokens
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in useful)


def _expanded_query(query: str) -> tuple[str, list[str]]:
    additions: list[str] = []
    for phrase, synonyms in _QUERY_EXPANSIONS.items():
        if phrase in query:
            additions.extend(synonyms)
    additions = list(dict.fromkeys(additions))
    return " ".join([query, *additions]), additions


def _rerank_score(row: sqlite3.Row, query: str, expansions: list[str]) -> float:
    searchable = " ".join(
        [str(row["title"]), str(row["content"]), str(row["keywords"])]
    ).lower()
    body_searchable = " ".join(
        [str(row["content"]), str(row["keywords"])]
    ).lower()
    units: list[str] = []
    for match in _CJK_RUN.finditer(query):
        value = match.group(0)
        units.extend(value[index : index + 2] for index in range(len(value) - 1))
        units.extend(value[index : index + 3] for index in range(len(value) - 2))
    units.extend(word.lower() for word in _WORD.findall(query))
    score = sum((2.0 if len(unit) >= 3 else 1.0) for unit in set(units) if unit in searchable)
    score += sum(3.0 for value in expansions if value.lower() in body_searchable)
    if query.lower() in searchable:
        score += 12.0
    if str(row["kind"]) != "chapter":
        score += 5.0
    if "净赚" in query and "净利润" in body_searchable:
        score += 25.0 if str(row["kind"]) != "chapter" else 8.0
    if "净赚" in query and "净利润是" in body_searchable:
        score += 20.0
    if "家属" in query and any(value in body_searchable for value in ("家属", "家庭", "遗属")):
        score += 12.0
    if "阵亡" in query and any(value in body_searchable for value in ("阵亡", "死者")):
        score += 8.0
    if "多少钱" in query and str(row["kind"]) == "number":
        score += 4.0
    if any(cue in query for cue in ("哪些", "如何", "为什么")) and str(row["kind"]) != "chapter":
        score += 2.0
    score += max(0.0, -float(row["rank"])) * 0.01
    return score


def search_index(
    query: str,
    database_path: str | Path = "data/index/knowledge.db",
    *,
    limit: int = 8,
) -> list[SearchResult]:
    """Search chapters and structured items, returning timestamped evidence."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    database = Path(database_path).resolve()
    if not database.is_file():
        raise ValueError(f"index database does not exist: {database}")
    expanded, expansions = _expanded_query(query)
    match = _match_query(expanded)
    try:
        with closing(_connect(database)) as connection:
            rows = connection.execute(
                """
                SELECT d.*, bm25(search_index, 1.5, 1.0, 2.0, 0.0) AS rank
                FROM search_index
                JOIN documents d ON d.id = search_index.id
                WHERE search_index MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, max(limit * 50, 500)),
            ).fetchall()
    except sqlite3.Error as exc:
        raise IndexingError(f"cannot search SQLite FTS5 index: {exc}") from exc
    ranked = sorted(
        rows,
        key=lambda row: _rerank_score(row, query, expansions),
        reverse=True,
    )[:limit]
    return [
        SearchResult(
            id=str(row["id"]),
            video_id=str(row["video_id"]),
            kind=str(row["kind"]),
            chapter_id=int(row["chapter_id"]),
            title=str(row["title"]),
            content=str(row["content"]),
            source_url=str(row["source_url"]),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            score=_rerank_score(row, query, expansions),
            file_path=str(row["file_path"]),
        )
        for row in ranked
    ]
