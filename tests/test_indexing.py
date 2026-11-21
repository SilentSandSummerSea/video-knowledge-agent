import json
import tempfile
import unittest
from pathlib import Path

from video_knowledge_agent.indexing import build_index, search_index


class IndexingTests(unittest.TestCase):
    def test_ranking_uses_subject_terms_and_video_title_across_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = root / "published"

            def add_video(video_id: str, video_title: str, chapter_title: str, content: str):
                video = published / video_id
                chapters = video / "chapters"
                chapters.mkdir(parents=True)
                (chapters / "01.md").write_text(content, encoding="utf-8")
                (video / "metadata.json").write_text(
                    json.dumps({
                        "video_id": video_id,
                        "title": video_title,
                        "source_url": f"https://example.com/{video_id}",
                        "chapters": [{
                            "id": 1, "title": chapter_title, "file": "chapters/01.md",
                            "start_seconds": 0, "end_seconds": 30, "topics": [],
                        }],
                    }, ensure_ascii=False), encoding="utf-8",
                )

            add_video("guide", "鬼武者攻略", "商店兑换", "在商店兑换道具。")
            add_video("commentary", "鬼武者评论", "鬼武者评测", "鬼武者的动作系统。")
            add_video("path", "关卡路线设计", "位置选择", "玩家在哪里站立会影响视线。")
            add_video("baking", "Bread baking", "Dough fermentation", "The dough doubles in size.")
            add_video("question", "Question writing", "Where to begin", "Where should a question begin?")
            add_video("profit", "项目结算", "净利润", "净利润为三十金币。")
            add_video("expense", "游戏经济", "金币支出", "金币成本为二十。")
            database = build_index(published, root / "knowledge.db").database_path

            self.assertEqual(search_index("鬼武者商店在哪里", database, limit=1)[0].video_id, "guide")
            self.assertEqual(search_index("店はどこ", database, limit=1)[0].video_id, "guide")
            self.assertEqual(search_index("Where is the dough?", database, limit=1)[0].video_id, "baking")
            self.assertEqual(search_index("净赚多少金币", database, limit=1)[0].video_id, "profit")
            self.assertEqual(search_index("在哪里", database), [])

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
