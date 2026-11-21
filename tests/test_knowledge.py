import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.knowledge import (
    KnowledgeExtractionError,
    extract_structured_knowledge,
)


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(self.output, ensure_ascii=False),
            usage=SimpleNamespace(
                input_tokens=50,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=5),
                output_tokens_details=SimpleNamespace(reasoning_tokens=4),
            ),
        )


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


class SequenceResponses(FakeResponses):
    def __init__(self, outputs):
        super().__init__(None)
        self.outputs = list(outputs)

    def create(self, **kwargs):
        self.output = self.outputs[len(self.requests)]
        return super().create(**kwargs)


class SequenceClient:
    def __init__(self, outputs):
        self.responses = SequenceResponses(outputs)


def write_inputs(root: Path) -> Path:
    transcript = root / "proofread.json"
    transcript.write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [
                    {"id": 1, "start_seconds": 0, "end_seconds": 4, "corrected_text": "准备费用为十金币。"},
                    {"id": 2, "start_seconds": 4, "end_seconds": 9, "corrected_text": "任务完成后获得二十金币。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chapters = root / "chapters.json"
    chapters.write_text(
        json.dumps(
            {
                "source_url": "https://example.com/video",
                "title": "测试视频",
                "source_language": "en",
                "output_language": "zh",
                "input_transcript": str(transcript),
                "chapters": [
                    {"id": 1, "title": "收支", "summary": "任务收支。", "start_segment_id": 1, "end_segment_id": 2}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return chapters


def valid_output(evidence_id=1):
    return {
        "chapter_summary": "任务投入十金币并获得二十金币。",
        "entities": [{"name": "金币", "type": "货币", "description": "任务使用的计价单位"}],
        "knowledge_items": [
            {
                "type": "number",
                "statement": "任务准备费用为十金币。",
                "evidence_segment_ids": [evidence_id],
                "keywords": ["费用", "金币"],
                "confidence": "high",
            }
        ],
    }


class KnowledgeTests(unittest.TestCase):
    def test_extracts_evidence_and_resumes_chapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chapters = write_inputs(root)
            client = FakeClient(valid_output())
            first = extract_structured_knowledge(chapters, root / "out", client=client)
            second = extract_structured_knowledge(chapters, root / "out", client=client)

            self.assertEqual(first.executed_chapters, 1)
            self.assertEqual(second.reused_chapters, 1)
            self.assertEqual(len(client.responses.requests), 1)
            data = json.loads(first.json_path.read_text(encoding="utf-8"))
            item = data["chapters"][0]["knowledge_items"][0]
            self.assertEqual(item["evidence_text"], "准备费用为十金币。")
            self.assertEqual(item["start_seconds"], 0.0)
            self.assertIn("t=0", item["source_url"])
            self.assertEqual(client.responses.requests[0]["reasoning"], {"effort": "none"})
            self.assertEqual(client.responses.requests[0]["max_output_tokens"], 8192)
            self.assertEqual(
                client.responses.requests[0]["text"]["format"]["schema"]
                ["properties"]["knowledge_items"]["maxItems"],
                12,
            )
            request_payload = json.loads(
                client.responses.requests[0]["input"][1]["content"]
            )
            self.assertEqual(request_payload["output_language"], "zh")
            self.assertEqual(data["source_language"], "en")
            self.assertEqual(data["output_language"], "zh")

    def test_rejects_evidence_outside_chapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chapters = write_inputs(root)
            with self.assertRaises(KnowledgeExtractionError):
                extract_structured_knowledge(
                    chapters,
                    root / "out",
                    client=FakeClient(valid_output(evidence_id=99)),
                )

    def test_retries_once_after_invalid_structured_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chapters = write_inputs(root)
            invalid = valid_output()
            invalid["knowledge_items"][0].pop("statement")
            client = SequenceClient([invalid, valid_output()])

            artifact = extract_structured_knowledge(
                chapters,
                root / "out",
                client=client,
            )

            self.assertEqual(len(client.responses.requests), 2)
            self.assertEqual(artifact.input_tokens, 100)
            self.assertEqual(artifact.output_tokens, 40)


if __name__ == "__main__":
    unittest.main()
