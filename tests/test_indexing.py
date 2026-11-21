import json
import tempfile
import unittest
from pathlib import Path

from video_knowledge_agent.indexing import build_index, search_index


class IndexingTests(unittest.TestCase):
    def test_builds_and_searches_chinese_knowledge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "published" / "BV1"
            chapters = video / "chapters"
            chapters.mkdir(parents=True)
            chapter_file = chapters / "01-收益.md"
            chapter_file.write_text("# 收益\n\n任务最后获得净利润三十金币。", encoding="utf-8")
            structured = {
                "chapters": [
                    {
                        "chapter_id": 1,
                        "chapter_title": "收益",
                        "knowledge_items": [
                            {
                                "id": "c01-k001",
                                "type": "number",
                                "statement": "任务净利润为三十金币。",
                                "evidence_text": "最终净利润三十金币。",
                                "keywords": ["净利润", "金币"],
                                "start_seconds": 10,
                                "end_seconds": 15,
                                "source_url": "https://example.com/video?t=10",
                            }
                        ],
                    }
                ]
            }
            (video / "structured_knowledge.json").write_text(
                json.dumps(structured, ensure_ascii=False), encoding="utf-8"
            )
            (video / "moments.json").write_text(
                json.dumps({
                    "video_id": "BV1",
                    "moments": [{
                        "id": "BV1:c01:m001",
                        "chapter_id": 1,
                        "text": "最终净利润三十金币。",
                        "start_seconds": 10,
                        "end_seconds": 15,
                        "source_url": "https://example.com/video?t=10",
                    }],
                }, ensure_ascii=False), encoding="utf-8",
            )
            metadata = {
                "video_id": "BV1",
                "source_url": "https://example.com/video",
                "structured_knowledge_file": "structured_knowledge.json",
                "moments_file": "moments.json",
                "chapters": [
                    {
                        "id": 1,
                        "title": "收益",
                        "file": "chapters/01-收益.md",
                        "start_seconds": 0,
                        "end_seconds": 30,
                        "topics": ["利润"],
                    }
                ],
            }
            (video / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )

            japanese_video = root / "published" / "YTJP"
            japanese_chapters = japanese_video / "chapters"
            japanese_chapters.mkdir(parents=True)
            (japanese_chapters / "01-バランス.md").write_text(
                "# ゲームバランス\n\nプレイヤーの選択肢を調整する。",
                encoding="utf-8",
            )
            (japanese_video / "structured_knowledge.json").write_text(
                json.dumps(
                    {
                        "output_language": "ja",
                        "chapters": [
                            {
                                "chapter_id": 1,
                                "chapter_title": "ゲームバランス",
                                "knowledge_items": [
                                    {
                                        "id": "c01-k001",
                                        "type": "definition",
                                        "statement": "ゲームバランスは選択肢を調整する設計である。",
                                        "evidence_text": "プレイヤーの選択肢を調整する。",
                                        "keywords": ["ゲームバランス", "選択肢"],
                                        "start_seconds": 20,
                                        "end_seconds": 25,
                                        "source_url": "https://example.com/jp?t=20",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (japanese_video / "metadata.json").write_text(
                json.dumps(
                    {
                        "video_id": "YTJP",
                        "source_url": "https://example.com/jp",
                        "structured_knowledge_file": "structured_knowledge.json",
                        "chapters": [
                            {
                                "id": 1,
                                "title": "ゲームバランス",
                                "file": "chapters/01-バランス.md",
                                "start_seconds": 0,
                                "end_seconds": 30,
                                "topics": ["ゲームバランス"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            artifact = build_index(root / "published", root / "knowledge.db")
            results = search_index("净赚多少金币", artifact.database_path)
            japanese_results = search_index("バランス", artifact.database_path)

            self.assertEqual(artifact.video_count, 2)
            self.assertEqual(artifact.chapter_count, 2)
            self.assertEqual(artifact.knowledge_count, 2)
            self.assertEqual(artifact.moment_count, 1)
            self.assertTrue(results)
            self.assertEqual(results[0].video_id, "BV1")
            self.assertIn("金币", results[0].content)
            self.assertTrue(any(result.source_url.endswith("t=10") for result in results))
            self.assertTrue(any(result.kind == "moment" and "净利润" in result.content for result in results))
            self.assertTrue(japanese_results)
            self.assertEqual(japanese_results[0].video_id, "YTJP")
            self.assertIn("根拠：", japanese_results[0].content)
            self.assertNotIn("证据：", japanese_results[0].content)


if __name__ == "__main__":
    unittest.main()
