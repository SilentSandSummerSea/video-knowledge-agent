import json
import tempfile
import unittest
from pathlib import Path

from video_knowledge_agent.publishing import publish_knowledge


class PublishingTests(unittest.TestCase):
    def test_publishes_only_chapters_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "sections"
            sections.mkdir()
            (sections / "01-开场.md").write_text("# 开场", encoding="utf-8")
            chapter_json = root / "chapters.json"
            chapter_json.write_text(
                json.dumps(
                    {
                        "title": "测试视频",
                        "source_url": "https://example.com/video",
                        "chapters": [
                            {
                                "id": 1,
                                "title": "开场",
                                "start_seconds": 0,
                                "end_seconds": 30,
                                "topics": ["介绍"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = publish_knowledge(chapter_json, sections, root / "out", "BV1")

            self.assertEqual(result.chapter_count, 1)
            self.assertTrue((result.chapters_dir / "01-开场.md").is_file())
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["chapter_count"], 1)
            self.assertEqual(
                sorted(path.name for path in result.knowledge_dir.iterdir()),
                ["chapters", "metadata.json"],
            )


if __name__ == "__main__":
    unittest.main()
