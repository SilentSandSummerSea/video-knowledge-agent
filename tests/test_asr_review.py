import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_agent.asr_review import review_flagged_segments_groq


class FakeTranscriptions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text="Oculus Riftは90fpsです。")


class FakeClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())


class ASRReviewTests(unittest.TestCase):
    def test_reviews_only_flagged_segments_and_preserves_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.webm"
            audio.write_bytes(b"fake")
            proofread = root / "proofread.json"
            proofread.write_text(
                json.dumps(
                    {
                        "source_url": "https://example.com/video",
                        "title": "VR",
                        "segments": [
                            {"id": 1, "start_seconds": 0, "end_seconds": 5, "asr_text": "Liftは90fps", "corrected_text": "Liftは90fps", "changed": False, "confidence": "low", "needs_review": True, "note": ""},
                            {"id": 2, "start_seconds": 5, "end_seconds": 10, "asr_text": "正常です", "corrected_text": "正常です", "changed": False, "confidence": "high", "needs_review": False, "note": ""},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            client = FakeClient()

            from unittest.mock import patch
            import numpy as np

            with patch("video_knowledge_agent.asr_review._decode_mono_16k", return_value=(np.zeros((1, 160000), dtype="int16"), 16000)), patch("video_knowledge_agent.asr_review._write_flac", side_effect=lambda path, samples, rate: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(b"flac"))):
                artifact = review_flagged_segments_groq(
                    audio,
                    proofread,
                    root / "reviewed",
                    language="ja",
                    terms=["Oculus Rift"],
                    client=client,
                    request_interval_seconds=0,
                )

            result = json.loads(artifact.json_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact.candidate_count, 1)
            self.assertEqual(artifact.replacement_count, 1)
            self.assertEqual(result["segments"][0]["corrected_text"], "Oculus Riftは90fps")
            self.assertEqual(result["segments"][1]["corrected_text"], "正常です")
            self.assertEqual(client.audio.transcriptions.requests[0]["model"], "whisper-large-v3")

    def test_keeps_proofread_text_when_full_model_adds_no_verified_term(self):
        from video_knowledge_agent.asr_review import (
            _canonicalize_candidate_terms,
            _safe_candidate,
            _verified_term_hits,
        )

        prior = "またこれはLiftonに関しての話ですが"
        candidate = "またこれはLiptonに関しての話ですが"
        terms = ["Oculus Rift", "Gear VR"]
        self.assertTrue(_safe_candidate(candidate, prior))
        self.assertEqual(_verified_term_hits(candidate, terms), 0)
        self.assertEqual(_verified_term_hits(prior, terms), 0)
        self.assertEqual(
            _canonicalize_candidate_terms("このSanJoseで", ["San Jose"]),
            "このSan Joseで",
        )

    def test_verified_term_patch_consumes_complete_wrong_latin_token(self):
        from video_knowledge_agent.asr_review import _patch_verified_term_differences

        selected, applied = _patch_verified_term_differences(
            "またこれはLiftonに関しての話ですが",
            "またこれはRiftに関しての話ですが",
            ["Rift"],
        )

        self.assertEqual(selected, "またこれはRiftに関しての話ですが")
        self.assertEqual(applied, ["Rift"])

    def test_verified_term_patch_merges_one_character_diff_island(self):
        from video_knowledge_agent.asr_review import _patch_verified_term_differences

        selected, applied = _patch_verified_term_differences(
            "3の瀬で開発者会議",
            "このSan Joseで開発者会議",
            ["San Jose"],
        )

        self.assertEqual(selected, "このSan Joseで開発者会議")
        self.assertEqual(applied, ["San Jose"])

    def test_verified_term_patch_preserves_text_between_multiple_terms(self):
        from video_knowledge_agent.asr_review import _patch_verified_term_differences

        selected, _ = _patch_verified_term_differences(
            "リフトは90fps、EAVRは60fpsです。",
            "Riftは90fps, Gear VRは60fpsです。",
            ["Rift", "Gear VR"],
        )

        self.assertEqual(selected, "Riftは90fps、Gear VRは60fpsです。")

    def test_verified_multiword_term_does_not_duplicate_equal_suffix(self):
        from video_knowledge_agent.asr_review import _patch_verified_term_differences

        selected, _ = _patch_verified_term_differences(
            "HomeStar VRとOculus Connect4",
            "HOMESTAR VRとOculus Connect 4",
            ["HOMESTAR VR", "Oculus Connect 4"],
        )

        self.assertEqual(selected, "HOMESTAR VRとOculus Connect 4")


if __name__ == "__main__":
    unittest.main()
