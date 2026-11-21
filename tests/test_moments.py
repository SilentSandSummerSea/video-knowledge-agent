import json
import tempfile
import unittest
from pathlib import Path

from video_knowledge_agent.moments import MomentError, build_moments


class MomentTests(unittest.TestCase):
    def test_groups_all_segments_once_without_crossing_chapters_and_reuses_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "proofread.json"
            transcript.write_text(json.dumps({
                "source_url": "https://example.com/watch?v=one",
                "segments": [
                    {"id": 1, "start_seconds": 0, "end_seconds": 10, "corrected_text": "Opening."},
                    {"id": 2, "start_seconds": 11, "end_seconds": 20, "corrected_text": "First point.", "needs_review": True},
                    {"id": 3, "start_seconds": 22, "end_seconds": 30, "corrected_text": "Second point."},
                    {"id": 4, "start_seconds": 31, "end_seconds": 42, "corrected_text": "New chapter."},
                    {"id": 5, "start_seconds": 44, "end_seconds": 50, "corrected_text": "Conclusion."},
                ],
            }), encoding="utf-8")
            chapters = root / "chapters.json"
            chapters.write_text(json.dumps({
                "source_url": "https://example.com/watch?v=one",
                "input_transcript": str(transcript),
                "chapters": [
                    {"id": 1, "title": "Part one", "start_segment_id": 1, "end_segment_id": 3},
                    {"id": 2, "title": "Part two", "start_segment_id": 4, "end_segment_id": 5},
                ],
            }), encoding="utf-8")

            first = build_moments(chapters, root / "moments", "one", max_seconds=25)
            second = build_moments(chapters, root / "moments", "one", max_seconds=25)
            payload = json.loads(first.json_path.read_text(encoding="utf-8"))
            moments = payload["moments"]

            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertEqual(first.moment_count, 3)
            self.assertEqual([item["segment_ids"] for item in moments], [[1, 2], [3], [4, 5]])
            self.assertEqual([item["chapter_id"] for item in moments], [1, 1, 2])
            self.assertEqual(moments[0]["start_seconds"], 0)
            self.assertEqual(moments[0]["end_seconds"], 20)
            self.assertTrue(moments[0]["needs_review"])
            self.assertEqual(moments[1]["source_url"], "https://example.com/watch?v=one&t=22")
            self.assertEqual(payload["segment_count"], 5)

    def test_rejects_missing_segment_in_chapter_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "proofread.json"
            transcript.write_text(json.dumps({"segments": [
                {"id": 1, "start_seconds": 0, "end_seconds": 2, "corrected_text": "One"},
                {"id": 3, "start_seconds": 3, "end_seconds": 5, "corrected_text": "Three"},
            ]}), encoding="utf-8")
            chapters = root / "chapters.json"
            chapters.write_text(json.dumps({
                "source_url": "https://example.com/video",
                "input_transcript": str(transcript),
                "chapters": [{"id": 1, "title": "All", "start_segment_id": 1, "end_segment_id": 3}],
            }), encoding="utf-8")

            with self.assertRaisesRegex(MomentError, "missing or repeated segment 2"):
                build_moments(chapters, root / "moments", "one")
