import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_knowledge_agent.indexing import SearchResult
from video_knowledge_agent.retrieval import search_with_context


class ContextRetrievalTests(unittest.TestCase):
    def test_knowledge_hit_uses_reviewed_transcript_and_nearby_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "work" / "video123" / "asr_review" / "proofread_reviewed.json"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(json.dumps({"segments": [
                {"id": 1, "start_seconds": 0, "end_seconds": 5, "corrected_text": "Context before."},
                {"id": 2, "start_seconds": 5, "end_seconds": 10, "corrected_text": "Rift runs at 90fps.", "needs_review": True},
                {"id": 3, "start_seconds": 10, "end_seconds": 15, "corrected_text": "Context after."},
            ]}), encoding="utf-8")
            metadata = root / "published" / "video123" / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({
                "title": "VR design talk",
                "source_url": "https://example.com/watch?v=video123",
            }), encoding="utf-8")
            hit = SearchResult(
                id="k1", video_id="video123", kind="number", chapter_id=1,
                title="Frame rate", content="Rift runs at 90fps.",
                source_url="https://example.com/watch?v=video123&t=5",
                start_seconds=5, end_seconds=10, score=10, file_path="chapter.md",
            )
            with patch("video_knowledge_agent.retrieval.search_index", return_value=[hit]):
                results = search_with_context(
                    "Rift 90fps", root / "index.db", root / "work", root / "published",
                    context_seconds=5,
                )

            self.assertEqual(results[0]["video_title"], "VR design talk")
            context = results[0]["context"]
            self.assertEqual(context["precision"], "knowledge_evidence")
            self.assertEqual([item["id"] for item in context["segments"]], [1, 2, 3])
            self.assertTrue(context["segments"][1]["needs_review"])
            self.assertEqual(context["transcript_file"], "asr_review/proofread_reviewed.json")
            self.assertEqual(context["evidence_url"], "https://example.com/watch?v=video123&t=5")

    def test_chapter_hit_anchors_to_matching_sentence_and_missing_transcript_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "work" / "video123" / "proofread" / "proofread.json"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(json.dumps({"segments": [
                {"id": 1, "start_seconds": 0, "end_seconds": 8, "corrected_text": "Opening remarks."},
                {"id": 2, "start_seconds": 20, "end_seconds": 25, "corrected_text": "Rift needs 90fps."},
                {"id": 3, "start_seconds": 50, "end_seconds": 55, "corrected_text": "Closing remarks."},
            ]}), encoding="utf-8")
            hit = SearchResult(
                id="c1", video_id="video123", kind="chapter", chapter_id=1,
                title="Frame rate", content="Rift needs 90fps.",
                source_url="https://example.com/watch?v=video123",
                start_seconds=0, end_seconds=60, score=5, file_path="chapter.md",
            )
            with patch("video_knowledge_agent.retrieval.search_index", return_value=[hit]):
                found = search_with_context(
                    "Rift 90fps", root / "index.db", root / "work", root / "published",
                )
                absent = search_with_context(
                    "Rift 90fps", root / "index.db", root / "absent", root / "published",
                )

            self.assertEqual(found[0]["context"]["precision"], "segment")
            self.assertEqual(found[0]["context"]["evidence_start_seconds"], 20)
            self.assertEqual(found[0]["context"]["evidence_url"], "https://example.com/watch?v=video123&t=20")
            self.assertEqual(absent[0]["context"]["precision"], "index_only")
            self.assertEqual(absent[0]["context"]["segments"], [])

    def test_moment_hit_keeps_its_exact_time_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "work" / "video123" / "proofread" / "proofread.json"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(json.dumps({"segments": [
                {"id": 1, "start_seconds": 20, "end_seconds": 25, "corrected_text": "Rift needs 90fps."},
            ]}), encoding="utf-8")
            hit = SearchResult(
                id="m1", video_id="video123", kind="moment", chapter_id=1,
                title="Frame rate", content="Rift needs 90fps.",
                source_url="https://example.com/watch?v=video123&t=20",
                start_seconds=20, end_seconds=25, score=5, file_path="chapter.md",
            )
            with patch("video_knowledge_agent.retrieval.search_index", return_value=[hit]):
                results = search_with_context(
                    "Rift 90fps", root / "index.db", root / "work", root / "published",
                )
            self.assertEqual(results[0]["context"]["precision"], "moment")
            self.assertEqual(results[0]["context"]["evidence_start_seconds"], 20)
            self.assertEqual(results[0]["context"]["evidence_end_seconds"], 25)


if __name__ == "__main__":
    unittest.main()
