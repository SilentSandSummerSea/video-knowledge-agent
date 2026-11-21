import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.answering import AnsweringError, answer_question


def _hit() -> dict:
    return {
        "id": "item-1", "video_id": "video-1", "video_title": "面包教程",
        "kind": "moment", "title": "发酵", "content": "面团发酵到两倍大",
        "start_seconds": 60.0, "end_seconds": 70.0,
        "source_url": "https://example.com/video", "file_path": "chapter.md",
        "context": {
            "precision": "moment", "evidence_start_seconds": 60.0,
            "evidence_end_seconds": 70.0,
            "evidence_url": "https://example.com/video?t=60",
            "segments": [{"id": 1, "start_seconds": 60.0, "end_seconds": 65.0,
                          "text": "发酵到两倍大", "needs_review": False}],
        },
    }


class _Client:
    def __init__(self, claims, limitations=""):
        self.request = None
        self.answer = {"claims": claims, "limitations": limitations}
        self.responses = self

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            output_text=json.dumps(self.answer, ensure_ascii=False),
            usage={"input_tokens": 200, "output_tokens": 80},
        )


class AnsweringTests(unittest.TestCase):
    def test_text_answer_uses_evidence_link_without_visual_work(self):
        client = _Client([{"text": "视频说面团应发酵到两倍大。", "evidence_ids": ["E1"]}])
        with tempfile.TemporaryDirectory() as directory:
            result = answer_question(
                "面团发酵到什么程度", database_path=Path(directory) / "index.db",
                work_root=directory, knowledge_root=directory, clips_root=directory,
                visual_mode="off", text_provider="deepseek", model="deepseek-flash",
                client=client, searcher=lambda *args, **kwargs: [_hit()],
                index_builder=lambda *args: None,
                clipper=lambda *args, **kwargs: self.fail("clip should not be fetched"),
            )
        self.assertEqual(result["visual_status"], "not_requested")
        self.assertIn("[00:01:00](https://example.com/video?t=60)", result["answer_markdown"])
        self.assertIn("发酵到两倍大", client.request["input"][1]["content"])
        self.assertEqual(client.request["reasoning"], {"effort": "none"})

    def test_visual_answer_uses_ocr_and_selected_frame(self):
        client = _Client([{"text": "画面中的面团明显膨胀。", "evidence_ids": ["F1", "E1"]}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame_001.jpg"
            frame.write_bytes(b"\xff\xd8\xff\xd9")
            ocr = root / "ocr.json"
            ocr.write_text(json.dumps({"frames": [{
                "image_file": frame.name, "video_seconds": 63.0,
                "source_url": "https://example.com/video?t=63",
                "lines": [{"text": "发酵后", "confidence": 0.95}],
                "aligned_speech": ["发酵到两倍大"],
            }]}, ensure_ascii=False), encoding="utf-8")
            result = answer_question(
                "画面里发酵到什么程度？", database_path=root / "index.db",
                work_root=root, knowledge_root=root, clips_root=root,
                text_provider="deepseek", model="deepseek-flash", client=client,
                searcher=lambda *args, **kwargs: [_hit()],
                index_builder=lambda *args: None,
                clipper=lambda *args, **kwargs: SimpleNamespace(file_path=root / "clip.mp4"),
                ocr_extractor=lambda *args, **kwargs: SimpleNamespace(json_path=ocr),
            )
        self.assertEqual(result["visual_status"], "frames_sent")
        self.assertIn("发酵后", client.request["input"][1]["content"][0]["text"])
        self.assertTrue(any(part["type"] == "input_image" for part in client.request["input"][1]["content"]))
        self.assertIn("https://example.com/video?t=63", result["answer_markdown"])
        self.assertIn("[截图]", result["answer_markdown"])

    def test_visual_failure_is_reported_without_inventing_frame_evidence(self):
        client = _Client([{"text": "语音提到发酵到两倍大。", "evidence_ids": ["E1"]}])
        with tempfile.TemporaryDirectory() as directory:
            result = answer_question(
                "画面里发酵到什么程度？", database_path=Path(directory) / "index.db",
                work_root=directory, knowledge_root=directory, clips_root=directory,
                text_provider="deepseek", model="deepseek-flash", client=client,
                searcher=lambda *args, **kwargs: [_hit()],
                index_builder=lambda *args: None,
                clipper=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("video unavailable")),
            )
        self.assertEqual(result["visual_status"], "unavailable")
        self.assertIn("无法核对画面", result["answer_markdown"])
        self.assertFalse(any(item["id"].startswith("F") for item in result["evidence"]))

    def test_model_cannot_create_its_own_evidence_id(self):
        client = _Client([{"text": "无法核实的结论", "evidence_ids": ["F99"]}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AnsweringError, "valid evidence IDs"):
                answer_question(
                    "面团发酵", database_path=Path(directory) / "index.db",
                    work_root=directory, knowledge_root=directory, clips_root=directory,
                    visual_mode="off", text_provider="deepseek", model="deepseek-flash",
                    client=client, searcher=lambda *args, **kwargs: [_hit()],
                    index_builder=lambda *args: None,
                )


if __name__ == "__main__":
    unittest.main()
