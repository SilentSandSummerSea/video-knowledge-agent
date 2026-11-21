import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicationBoundaryTests(unittest.TestCase):
    def test_public_core_has_no_website_downloader_dependency(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("yt-dlp", project)
        for source in (ROOT / "src" / "video_knowledge_agent").glob("*.py"):
            text = source.read_text(encoding="utf-8")
            self.assertNotIn("yt_dlp", text, source.name)
            self.assertNotIn("download_ranges", text, source.name)

    def test_private_adapter_and_generated_data_are_ignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("local-tools/", "data/imports/", "data/index/", ".env", "*.mp4"):
            self.assertIn(pattern, ignored)


if __name__ == "__main__":
    unittest.main()
