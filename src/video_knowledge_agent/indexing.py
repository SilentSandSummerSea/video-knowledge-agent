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
_QUESTION_PHRASES = re.compile(
    r"在哪里|在哪儿|在哪|哪里|什么|为什么|为何|怎么|如何|多少|"
    r"どこ|どの|なぜ|どう|いつ|何ですか|ですか|ますか|"
    r"\b(?:where|what|which|how|why|when|is|are|the|a|an|do|does)\b",
    re.IGNORECASE,
)


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
    video_title = str(metadata.get("title") or "")
    for chapter in metadata.get("chapters", []):
        relative = str(chapter["file"])
        path = (video_dir / relative).resolve()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IndexingError(f"cannot read published chapter {path}: {exc}") from exc
        chapter_id = int(chapter["id"])
        title = str(chapter["title"])
        keywords = " ".join([video_title, *map(str, chapter.get("topics", []))])
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
    video_title = str(metadata.get("title") or "")
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
                "keywords": " ".join([video_title, *map(str, item.get("keywords", []))]),
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
    video_title = str(metadata.get("title") or "")
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
            "keywords": " ".join([video_title, *map(str, chapter.get("topics", []))]),
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


def _query_terms(query: str) -> list[str]:
    meaningful = _QUESTION_PHRASES.sub(" ", query.casefold())
    # A Japanese topic particle left after removing a question phrase is not
    # part of the searched noun (e.g. 店はどこ -> 店).
    meaningful = re.sub(r"(?<=[\u3400-\u9fff])[はがをにの](?=\s|$)", " ", meaningful)
    return list(dict.fromkeys(_search_tokens(meaningful).split()))


def _match_query(query: str) -> str:
    terms = _query_terms(query)
    useful = [term for term in terms if len(term) >= 2 or len(terms) == 1]
    if not useful:
        useful = terms
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in useful
    )


def _rerank_score(row: sqlite3.Row, query: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    title = str(row["title"]).casefold()
    content = str(row["content"]).casefold()
    keywords = str(row["keywords"]).casefold()
    weights = {term: 0.5 if len(term) == 1 else min(len(term), 3) for term in terms}
    total = sum(weights.values())

    def coverage(value: str) -> float:
        return sum(weight for term, weight in weights.items() if term in value) / total

    body_coverage = coverage(content)
    # Covering distinct query terms across the video title and the passage is
    # more useful than repeating one term in several fields.
    score = (
        12.0 * coverage(" ".join([title, content, keywords]))
        + 2.0 * coverage(title)
        + 2.0 * body_coverage
        + coverage(keywords)
    )
    # A short evidence item is more precise than an entire chapter when both
    # cover the same query; BM25 breaks ties using corpus frequency and length.
    if body_coverage and str(row["kind"]) != "chapter":
        score += 1.0
    score += min(2.0, max(0.0, -float(row["rank"])) * 0.05)
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
    match = _match_query(query)
    if not match:
        return []
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
        key=lambda row: _rerank_score(row, query),
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
            score=_rerank_score(row, query),
            file_path=str(row["file_path"]),
        )
        for row in ranked
    ]
