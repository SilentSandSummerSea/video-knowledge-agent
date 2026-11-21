import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.chaptering import ChapteringError, extract_chapters


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(self.output, ensure_ascii=False),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=40,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
        )


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


class SequencedFakeResponses(FakeResponses):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.output = next(self.outputs)
        return super().create(**kwargs)


class SequencedFakeClient:
    def __init__(self, outputs):
        self.responses = SequencedFakeResponses(outputs)


def write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_url": "https://example.com/video",
                "title": "测试视频",
                "language": "en",
                "segments": [
                    {"id": 1, "start_seconds": 0, "end_seconds": 5, "corrected_text": "先介绍任务。"},
                    {"id": 2, "start_seconds": 5, "end_seconds": 12, "corrected_text": "然后准备工具。"},
                    {"id": 3, "start_seconds": 12, "end_seconds": 19, "corrected_text": "开始执行任务。"},
                    {"id": 4, "start_seconds": 19, "end_seconds": 25, "corrected_text": "最后总结结果。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def valid_output():
    return {
        "chapters": [
            {
                "title": "准备阶段",
                "start_segment_id": 1,
                "end_segment_id": 2,
                "summary": "介绍任务并准备工具。",
                "topics": ["任务", "工具"],
                "confidence": "high",
            },
            {
                "title": "执行与总结",
                "start_segment_id": 3,
                "end_segment_id": 4,
                "summary": "执行任务并总结结果。",
                "topics": ["执行", "结果"],
                "confidence": "high",
            },
        ]
    }


class ChapteringTests(unittest.TestCase):
    def test_extracts_contiguous_chapters_and_derives_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "proofread.json"
            write_transcript(transcript)
            client = FakeClient(valid_output())

            artifact = extract_chapters(transcript, root / "out", client=client)

            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact.chapter_count, 2)
            self.assertEqual(result["chapters"][0]["start_seconds"], 0.0)
            self.assertEqual(result["chapters"][0]["end_seconds"], 12.0)
            self.assertEqual(result["chapters"][1]["start_seconds"], 12.0)
            self.assertEqual(result["chapters"][1]["end_seconds"], 25.0)
            self.assertEqual(artifact.input_tokens, 100)
            self.assertEqual(artifact.cached_input_tokens, 20)
            self.assertEqual(artifact.reasoning_tokens, 10)
            self.assertEqual(len(artifact.section_paths), 2)
            first_section = artifact.section_paths[0].read_text(encoding="utf-8")
            self.assertIn("## 转写正文", first_section)
            self.assertIn("先介绍任务。然后准备工具。", first_section)
            self.assertEqual(first_section.count("https://example.com/video"), 1)
            request = client.responses.requests[0]
            self.assertEqual(request["model"], "deepseek-flash")
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertEqual(request["max_output_tokens"], 8192)
            self.assertEqual(request["text"]["format"]["type"], "json_schema")
            user_payload = json.loads(request["input"][1]["content"])
            self.assertEqual(user_payload["source_language"], "en")
            self.assertEqual(user_payload["output_language"], "en")
            self.assertEqual(result["source_language"], "en")
            self.assertEqual(result["output_language"], "en")

    def test_rejects_gap_between_chapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "proofread.json"
            write_transcript(transcript)
            output = valid_output()
            output["chapters"][0]["end_segment_id"] = 1
            with self.assertRaises(ChapteringError):
                extract_chapters(
                    transcript,
                    root / "out",
                    client=FakeClient(output),
                )

    def test_retries_once_after_invalid_structured_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "proofread.json"
            write_transcript(transcript)
            malformed = valid_output()
            del malformed["chapters"][1]["start_segment_id"]
            client = SequencedFakeClient([malformed, valid_output()])

            artifact = extract_chapters(transcript, root / "out", client=client)

            self.assertEqual(artifact.chapter_count, 2)
            self.assertEqual(len(client.responses.requests), 2)
            self.assertEqual(artifact.input_tokens, 200)


if __name__ == "__main__":
    unittest.main()
