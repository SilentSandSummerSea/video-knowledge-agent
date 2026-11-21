import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.video_clips import VideoClipError, extract_moment_video


class VideoClipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.knowledge = self.root / "knowledge"
        self.work = self.root / "work"
        self.clips = self.root / "clips"
        self.video = self.root / "source.mp4"
        self.video.write_bytes(b"local video fixture")
        published = self.knowledge / "video-1"
        published.mkdir(parents=True)
        (published / "metadata.json").write_text(json.dumps({
            "video_id": "video-1", "source_url": "local-video://video-1",
            "moments_file": "moments.json",
        }), encoding="utf-8")
        (published / "moments.json").write_text(json.dumps({
            "video_id": "video-1", "source_hash": "a" * 64,
            "moments": [
                {"id": "video-1:c01:m001", "chapter_id": 1,
                 "start_seconds": 0, "end_seconds": 12},
                {"id": "video-1:c01:m002", "chapter_id": 1,
                 "start_seconds": 12, "end_seconds": 36},
            ],
        }), encoding="utf-8")
        state = self.work / "video-1"
        state.mkdir(parents=True)
        (state / "pipeline.json").write_text(json.dumps({
            "video_id": "video-1", "requested_file": str(self.video),
        }), encoding="utf-8")
        self.ffmpeg = self.root / "ffmpeg.exe"
        self.ffmpeg.write_bytes(b"fixture")
        self.calls = []

    def run_ffmpeg(self, command, **kwargs):
        self.calls.append(command)
        Path(command[-1]).write_bytes(b"short video test fixture")
        return SimpleNamespace(returncode=0, stderr="")

    def extract(self, **kwargs):
        return extract_moment_video(
            "video-1", knowledge_root=self.knowledge, work_root=self.work,
            output_root=self.clips, ffmpeg_path=self.ffmpeg,
            runner=self.run_ffmpeg,
            validator=lambda path, expected: self.assertTrue(path.is_file()),
            **kwargs,
        )

    def test_extracts_only_padded_local_moment_and_reuses_clip(self):
        artifact = self.extract(moment_id="video-1:c01:m002")
        self.assertEqual((artifact.start_seconds, artifact.end_seconds), (7, 41))
        self.assertFalse(artifact.reused)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][self.calls[0].index("-i") + 1], str(self.video.resolve()))
        self.assertEqual(self.calls[0][self.calls[0].index("-ss") + 1], "7.0")
        self.assertEqual(self.calls[0][self.calls[0].index("-t") + 1], "34.0")
        self.assertEqual(artifact.file_path.read_bytes(), b"short video test fixture")
        self.assertTrue(self.extract(at_seconds=13).reused)
        self.assertEqual(len(self.calls), 1)

    def test_first_moment_padding_clamps_to_zero(self):
        artifact = self.extract(at_seconds=0)
        self.assertEqual((artifact.start_seconds, artifact.end_seconds), (0, 17))

    def test_missing_source_and_invalid_id_fail_closed(self):
        self.video.unlink()
        with self.assertRaisesRegex(VideoClipError, "local source video is unavailable"):
            self.extract(at_seconds=13)
        with self.assertRaisesRegex(ValueError, "invalid video_id"):
            extract_moment_video("../video-1", at_seconds=15, knowledge_root=self.knowledge)

    def test_audio_only_source_cannot_extract_video(self):
        state_path = self.work / "video-1" / "pipeline.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["source_has_video"] = False
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(VideoClipError, "audio-only"):
            self.extract(at_seconds=15)


if __name__ == "__main__":
    unittest.main()
