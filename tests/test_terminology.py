import json
import tempfile
import unittest
from pathlib import Path

from video_knowledge_agent.terminology import build_asr_prompt, extract_metadata_terms, merge_terms


class TerminologyTests(unittest.TestCase):
    def test_extracts_verified_terms_from_media_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.webm"
            audio.write_bytes(b"audio")
            (root / "source.info.json").write_text(
                json.dumps(
                    {
                        "title": "Bridging UX and Game Design",
                        "uploader": "GDC Festival of Gaming",
                        "tags": ["epic games", "fortnite", "UX"],
                        "description": "Epic Games' Jim Brown discusses UX.",
                    }
                ),
                encoding="utf-8",
            )
            terms = extract_metadata_terms(audio)
            self.assertIn("GDC Festival of Gaming", terms)
            self.assertIn("epic games", terms)
            self.assertIn("Jim Brown", terms)

    def test_user_spelling_wins_when_terms_are_merged(self):
        self.assertEqual(
            merge_terms(["Serious Sam", "UX"], ["serious sam", "Epic Games"]),
            ["Serious Sam", "UX", "Epic Games"],
        )

    def test_asr_prompt_preserves_whole_prioritized_terms_within_limit(self):
        terms = ["user term", "x" * 900, "metadata term", "测" * 300]
        prompt = build_asr_prompt(terms)
        self.assertEqual(prompt, "user term, metadata term")
        self.assertLessEqual(len(prompt.encode("utf-8")), 896)
        self.assertIsNone(build_asr_prompt([]))


if __name__ == "__main__":
    unittest.main()
