import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_knowledge_agent.visual_ocr import VisualOcrError, extract_moment_ocr


class VisualOcrTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("VIDEO_KNOWLEDGE_TEST_TMP"))
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.clip = root / "clips" / "video-1" / "clip.mp4"
        self.clip.parent.mkdir(parents=True)
        self.clip.write_bytes(b"sample clip")
        (self.clip.parent / "clip.json").write_text(json.dumps({
            "video_id": "video-1", "moment_id": "video-1:c01:m001",
            "source_hash": "hash-1", "start_seconds": 10.0,
            "end_seconds": 20.0, "file": "clip.mp4",
        }), encoding="utf-8")
        self.knowledge = root / "knowledge"
        published = self.knowledge / "video-1"
        published.mkdir(parents=True)
        (published / "metadata.json").write_text(json.dumps({
            "source_url": "https://www.youtube.com/watch?v=video-1",
        }), encoding="utf-8")
        self.transcript = root / "transcript.json"
        self.transcript.write_text(json.dumps({"segments": [
            {"id": 1, "start_seconds": 11, "end_seconds": 14, "corrected_text": "first"},
            {"id": 2, "start_seconds": 16, "end_seconds": 19, "corrected_text": "second"},
        ]}), encoding="utf-8")
        (published / "moments.json").write_text(json.dumps({
            "video_id": "video-1", "source_hash": "hash-1",
            "source_language": "en", "input_transcript": str(self.transcript),
            "moments": [{"id": "video-1:c01:m001", "chapter_id": 1,
                         "segment_ids": [1, 2]}],
        }), encoding="utf-8")

    def test_ocr_frames_align_to_absolute_video_time_and_reuse_cache(self):
        from PIL import Image

        calls = []

        def engine(_image):
            calls.append(True)
            return SimpleNamespace(txts=["Settings", "noise"], scores=[0.9, 0.1],
                                   boxes=[[[0, 0], [10, 0], [10, 5], [0, 5]], []])

        frames = [(2.0, Image.new("RGB", (40, 20))),
                  (7.0, Image.new("RGB", (40, 20)))]
        with patch("video_knowledge_agent.visual_ocr._sample_frames", return_value=frames):
            first = extract_moment_ocr(self.clip, knowledge_root=self.knowledge,
                                       ocr_engine=engine)
            second = extract_moment_ocr(self.clip, knowledge_root=self.knowledge,
                                        ocr_engine=lambda _: self.fail("OCR reran"))
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(len(calls), 2)
        payload = json.loads(first.json_path.read_text(encoding="utf-8"))
        self.assertEqual([frame["video_seconds"] for frame in payload["frames"]], [12.0, 17.0])
        self.assertEqual(payload["frames"][0]["aligned_segment_ids"], [1])
        self.assertEqual(payload["frames"][1]["aligned_segment_ids"], [2])
        self.assertEqual(payload["frames"][0]["lines"][0]["text"], "Settings")
        self.assertEqual(len(payload["frames"][0]["lines"]), 1)
        self.assertIn("t=12", payload["frames"][0]["source_url"])

    def test_rejects_stale_clip_manifest(self):
        manifest_path = self.clip.parent / "clip.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_hash"] = "other"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(VisualOcrError, "stale or mismatched"):
            extract_moment_ocr(self.clip, knowledge_root=self.knowledge,
                               ocr_engine=lambda _: None)


if __name__ == "__main__":
    unittest.main()
