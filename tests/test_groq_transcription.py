import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.groq_transcription import (
    _remove_overlapping_exact_duplicates,
    transcribe_audio_groq,
)
from video_knowledge_agent.transcription import TranscriptSegment


class FakeTranscriptions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            language="zh",
            duration=5.0,
            segments=[
                SimpleNamespace(start=0.0, end=2.5, text="屠掉了一头龙"),
                SimpleNamespace(start=2.5, end=5.0, text="开始分配战利品"),
            ],
        )


class FakeClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())


class GroqTranscriptionTests(unittest.TestCase):
    def test_removes_only_overlapping_exact_duplicates(self):
        segments = [
            TranscriptSegment(1, 118.0, 121.2, "边界上的同一句话"),
            TranscriptSegment(2, 119.0, 121.5, "边界上的同一句话"),
            TranscriptSegment(3, 130.0, 132.0, "边界上的同一句话"),
        ]

        result = _remove_overlapping_exact_duplicates(segments)

        self.assertEqual([segment.id for segment in result], [1, 3])
        self.assertEqual(
            [segment.start_seconds for segment in result],
            sorted(segment.start_seconds for segment in result),
        )

    def test_writes_timestamped_groq_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.m4a"
            audio.write_bytes(b"audio")
            client = FakeClient()

            artifact = transcribe_audio_groq(
                audio,
                root / "output",
                prompt="屠龙、战利品",
                client=client,
            )

            self.assertEqual(artifact.segment_count, 2)
            self.assertEqual(
                client.audio.transcriptions.request["model"],
                "whisper-large-v3-turbo",
            )
            self.assertEqual(
                client.audio.transcriptions.request["timestamp_granularities"],
                ["segment", "word"],
            )
            self.assertNotIn("language", client.audio.transcriptions.request)
            self.assertEqual(artifact.language, "zh")
            transcript = artifact.markdown_path.read_text(encoding="utf-8")
            self.assertIn("屠掉了一头龙", transcript)
            self.assertIn("groq/whisper-large-v3-turbo", transcript)

    def test_explicit_language_is_sent_to_groq(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.m4a"
            audio.write_bytes(b"audio")
            client = FakeClient()

            transcribe_audio_groq(
                audio,
                root / "output",
                language="ja",
                client=client,
            )

            self.assertEqual(client.audio.transcriptions.request["language"], "ja")


if __name__ == "__main__":
    unittest.main()
