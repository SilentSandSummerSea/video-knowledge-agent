import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.transcription import transcribe_audio


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.options = kwargs

    def transcribe(self, audio_path, **kwargs):
        segments = [
            SimpleNamespace(start=0.0, end=2.5, text=" 第一段 "),
            SimpleNamespace(start=2.5, end=4.0, text="第二段"),
        ]
        info = SimpleNamespace(language="zh", language_probability=0.99, duration=4.0)
        return iter(segments), info


class TranscriptionTests(unittest.TestCase):
    def test_writes_timestamped_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.m4a"
            audio.write_bytes(b"audio")
            output = root / "output"

            artifact = transcribe_audio(
                audio,
                output,
                model_factory=FakeModel,
            )

            self.assertEqual(artifact.segment_count, 2)
            self.assertTrue(artifact.json_path.exists())
            markdown = artifact.markdown_path.read_text(encoding="utf-8")
            self.assertIn("00:00:00", markdown)
            self.assertIn("第一段", markdown)


if __name__ == "__main__":
    unittest.main()
