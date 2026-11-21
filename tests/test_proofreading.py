import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.proofreading import ProofreadingError, _make_batches, proofread_transcript
from video_knowledge_agent.transcription import TranscriptSegment


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(next(self.outputs), ensure_ascii=False),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=40,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


class FencedFakeResponses(FakeResponses):
    def create(self, **kwargs):
        response = super().create(**kwargs)
        response.output_text = f"```json\n{response.output_text}\n```"
        return response


class FakeClient:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


class FencedFakeClient:
    def __init__(self, outputs):
        self.responses = FencedFakeResponses(outputs)


def write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_url": "https://example.com/video",
                "title": "屠龙之后",
                "model": "groq/whisper-large-v3-turbo",
                "language": "zh",
                "duration_seconds": 12.0,
                "segments": [
                    {
                        "id": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "text": "成功涂掉一头龙",
                    },
                    {
                        "id": 2,
                        "start_seconds": 5.0,
                        "end_seconds": 12.0,
                        "text": "取下龙龄",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class ProofreadingTests(unittest.TestCase):
    def test_batches_cap_segment_count_for_complete_model_responses(self):
        segments = [
            TranscriptSegment(id=index, start_seconds=index, end_seconds=index + 1, text=f"line {index}")
            for index in range(10)
        ]
        batches = _make_batches(segments, batch_seconds=300)
        self.assertEqual([len(batch) for batch in batches], [8, 2])

    def test_accepts_fenced_structured_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            output = {
                "segments": [
                    {"id": 1, "corrected_text": "成功屠掉一头龙", "confidence": "high", "needs_review": False, "note": ""},
                    {"id": 2, "corrected_text": "取下龙鳞", "confidence": "high", "needs_review": False, "note": ""},
                ]
            }
            artifact = proofread_transcript(
                transcript, root / "out", client=FencedFakeClient([output])
            )
            self.assertEqual(artifact.changed_count, 2)

    def test_proofreads_and_preserves_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            client = FakeClient(
                [
                    {
                        "segments": [
                            {
                                "id": 1,
                                "corrected_text": "成功屠掉一头龙",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "同音字修正",
                            },
                            {
                                "id": 2,
                                "corrected_text": "取下龙鳞",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "术语修正",
                            },
                        ]
                    }
                ]
            )

            artifact = proofread_transcript(
                transcript,
                root / "out",
                provider="openai",
                terms=["屠龙", "龙鳞"],
                client=client,
            )

            self.assertEqual(artifact.changed_count, 2)
            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertEqual(result["segments"][0]["start_seconds"], 0.0)
            self.assertEqual(result["segments"][1]["end_seconds"], 12.0)
            self.assertEqual(result["segments"][1]["asr_text"], "取下龙龄")
            self.assertEqual(result["segments"][1]["corrected_text"], "取下龙鳞")
            self.assertEqual(result["usage"]["input_tokens"], 100)
            self.assertEqual(artifact.output_tokens, 40)
            request = client.responses.requests[0]
            self.assertEqual(request["text"]["format"]["type"], "json_schema")
            user_payload = json.loads(request["input"][1]["content"])
            self.assertEqual(user_payload["source_language"], "zh")
            self.assertEqual(result["language"], "zh")
            self.assertIn("不得翻译", request["input"][0]["content"])
            self.assertIn("不得猜测或补写", request["input"][0]["content"])
            self.assertIn("source_language", request["input"][0]["content"])

    def test_verified_glossary_normalizes_close_entity_and_writes_review_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            transcript.write_text(
                json.dumps(
                    {
                        "source_url": "https://example.com/video",
                        "title": "Game design",
                        "model": "groq/whisper-large-v3-turbo",
                        "language": "en",
                        "duration_seconds": 5.0,
                        "segments": [
                            {
                                "id": 1,
                                "start_seconds": 0.0,
                                "end_seconds": 5.0,
                                "text": "This is a screenshot from Sirius Sam.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    {
                        "segments": [
                            {
                                "id": 1,
                                "corrected_text": "This is a screenshot from Sirius Sam.",
                                "confidence": "low",
                                "needs_review": True,
                                "note": "Uncertain game title.",
                            }
                        ]
                    }
                ]
            )

            artifact = proofread_transcript(
                transcript,
                root / "out",
                provider="openai",
                terms=["Serious Sam"],
                client=client,
            )

            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertIn("Serious Sam", result["segments"][0]["corrected_text"])
            self.assertIn("Sirius Sam → Serious Sam", result["segments"][0]["note"])
            review = json.loads(artifact.review_queue_path.read_text(encoding="utf-8"))
            self.assertEqual(review["segments"][0]["segment_id"], 1)
            self.assertEqual(review["segments"][0]["start_seconds"], 0.0)

    def test_rejects_missing_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            client = FakeClient(
                [
                    {
                        "segments": [
                            {
                                "id": 1,
                                "corrected_text": "成功屠掉一头龙",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "",
                            }
                        ]
                    }
                ]
            )

            with self.assertRaises(ProofreadingError):
                proofread_transcript(
                    transcript, root / "out", provider="openai", client=client
                )

    def test_ignores_echoed_read_only_context_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            client = FakeClient(
                [
                    {
                        "segments": [
                            {"id": 1, "corrected_text": "成功屠掉一头龙", "confidence": "high", "needs_review": False, "note": ""},
                            {"id": 2, "corrected_text": "上下文不得采用", "confidence": "high", "needs_review": False, "note": ""},
                        ]
                    },
                    {
                        "segments": [
                            {"id": 1, "corrected_text": "上下文不得采用", "confidence": "high", "needs_review": False, "note": ""},
                            {"id": 2, "corrected_text": "取下龙鳞", "confidence": "high", "needs_review": False, "note": ""},
                        ]
                    },
                ]
            )
            artifact = proofread_transcript(
                transcript,
                root / "out",
                provider="openai",
                batch_seconds=6,
                context_segments=1,
                client=client,
            )
            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertEqual(result["segments"][0]["corrected_text"], "成功屠掉一头龙")
            self.assertEqual(result["segments"][1]["corrected_text"], "取下龙鳞")

    def test_replaces_wrong_language_review_note_for_english(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            transcript.write_text(
                json.dumps(
                    {
                        "source_url": "https://example.com/video",
                        "title": "English talk",
                        "model": "groq/whisper-large-v3-turbo",
                        "language": "English",
                        "duration_seconds": 5.0,
                        "segments": [{"id": 1, "start_seconds": 0, "end_seconds": 5, "text": "unclear words"}],
                    }
                ),
                encoding="utf-8",
            )
            client = FakeClient(
                [{"segments": [{"id": 1, "corrected_text": "unclear words", "confidence": "low", "needs_review": True, "note": "无法确定原文"}]}]
            )
            artifact = proofread_transcript(transcript, root / "out", provider="openai", client=client)
            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result["segments"][0]["note"],
                "Uncertain transcription; original wording preserved for review.",
            )

    def test_codex_uses_low_cost_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            calls = []

            def fake_runner(prompt, **kwargs):
                calls.append((prompt, kwargs))
                return json.dumps(
                    {
                        "segments": [
                            {
                                "id": 1,
                                "corrected_text": "成功屠掉一头龙",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "同音字修正",
                            },
                            {
                                "id": 2,
                                "corrected_text": "取下龙鳞",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "术语修正",
                            },
                        ]
                    },
                    ensure_ascii=False,
                )

            artifact = proofread_transcript(
                transcript,
                root / "out",
                provider="codex",
                codex_runner=fake_runner,
            )

            self.assertEqual(artifact.provider, "codex")
            self.assertEqual(artifact.model, "gpt-6-luna")
            self.assertEqual(calls[0][1]["reasoning_effort"], "low")
            self.assertIn("不要调用工具", calls[0][0])

    def test_deepseek_is_default_with_flash_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.json"
            write_transcript(transcript)
            client = FakeClient(
                [
                    {
                        "segments": [
                            {
                                "id": 1,
                                "corrected_text": "成功屠掉一头龙",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "同音字修正",
                            },
                            {
                                "id": 2,
                                "corrected_text": "取下龙鳞",
                                "confidence": "high",
                                "needs_review": False,
                                "note": "术语修正",
                            },
                        ]
                    }
                ]
            )

            artifact = proofread_transcript(
                transcript,
                root / "out",
                client=client,
            )

            self.assertEqual(artifact.provider, "deepseek")
            self.assertEqual(artifact.model, "deepseek-flash")
            self.assertEqual(
                client.responses.requests[0]["text"]["format"]["type"],
                "json_schema",
            )
            self.assertEqual(
                client.responses.requests[0]["reasoning"],
                {"effort": "none"},
            )
            self.assertEqual(client.responses.requests[0]["max_output_tokens"], 16384)


if __name__ == "__main__":
    unittest.main()
