import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.audio import LocalMediaError, prepare_local_audio


class AudioTests(unittest.TestCase):
    def test_local_video_extracts_audio_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"local fixture")
            ffmpeg = root / "ffmpeg.exe"
            ffmpeg.write_bytes(b"fixture")
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                Path(command[-1]).write_bytes(b"audio fixture")
                return SimpleNamespace(returncode=0, stderr="")

            options = {"ffmpeg_path": ffmpeg, "runner": run, "validator": lambda path: 12.5, "stream_probe": lambda path: True}
            first = prepare_local_audio(video, root / "raw", **options)
            second = prepare_local_audio(video, root / "raw", **options)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.video_id, second.video_id)
            self.assertEqual(first.video_path, video.resolve())
            self.assertEqual(first.source_path, video.resolve())
            self.assertEqual(first.source_url, f"local-video://{first.video_id}")
            self.assertEqual(first.duration_seconds, 12.5)
            self.assertEqual(calls[0][calls[0].index("-i") + 1], str(video.resolve()))
            metadata = json.loads((first.file_path.parent / "source.info.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["local_video_path"], str(video.resolve()))

    def test_rejects_url_and_missing_video(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "local media file"):
                prepare_local_audio("https://example.com/video.mp4", directory)
            with self.assertRaisesRegex(ValueError, "local media file"):
                prepare_local_audio(Path(directory) / "missing.mp4", directory)

    def test_ffmpeg_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            ffmpeg = root / "ffmpeg.exe"
            ffmpeg.write_bytes(b"fixture")
            with self.assertRaisesRegex(LocalMediaError, "FFmpeg audio extraction failed"):
                prepare_local_audio(
                    video, root / "raw", ffmpeg_path=ffmpeg,
                    runner=lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="broken"),
                    validator=lambda path: 10,
                    stream_probe=lambda path: True,
                )

    def test_local_audio_has_no_video_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech.m4a"
            source.write_bytes(b"fixture")
            ffmpeg = root / "ffmpeg.exe"
            ffmpeg.write_bytes(b"fixture")

            def run(command, **kwargs):
                Path(command[-1]).write_bytes(b"normalized audio")
                return SimpleNamespace(returncode=0, stderr="")

            artifact = prepare_local_audio(
                source, root / "raw", ffmpeg_path=ffmpeg, runner=run,
                validator=lambda path: 15.0, stream_probe=lambda path: False,
            )
            self.assertEqual(artifact.source_path, source.resolve())
            self.assertIsNone(artifact.video_path)
            self.assertTrue(artifact.source_url.startswith("local-audio://"))
            metadata = json.loads((artifact.file_path.parent / "source.info.json").read_text(encoding="utf-8"))
            self.assertIs(metadata["has_video"], False)

    def test_audio_sidecar_preserves_source_reference_and_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.webm"
            source.write_bytes(b"fixture")
            (root / "example.info.json").write_text(json.dumps({
                "webpage_url": "https://example.org/watch?v=123",
                "title": "Example Video", "uploader": "Example Channel",
                "tags": ["Example Term"],
            }), encoding="utf-8")
            ffmpeg = root / "ffmpeg.exe"
            ffmpeg.write_bytes(b"fixture")

            def run(command, **kwargs):
                Path(command[-1]).write_bytes(b"normalized audio")
                return SimpleNamespace(returncode=0, stderr="")

            options = {"ffmpeg_path": ffmpeg, "runner": run, "validator": lambda path: 15.0, "stream_probe": lambda path: False}
            first = prepare_local_audio(source, root / "raw", **options)
            second = prepare_local_audio(source, root / "raw", **options)
            self.assertEqual(first.source_url, "https://example.org/watch?v=123")
            self.assertEqual(second.source_url, first.source_url)
            self.assertEqual(second.title, "Example Video")
            self.assertEqual(second.author, "Example Channel")
            metadata = json.loads((first.file_path.parent / "source.info.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["tags"], ["Example Term"])


if __name__ == "__main__":
    unittest.main()
