import unittest

from video_knowledge_agent.models import KnowledgeChunk, format_timestamp, timestamp_url


class ModelTests(unittest.TestCase):
    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(3723.8), "01:02:03")

    def test_timestamp_url_preserves_query(self) -> None:
        url = timestamp_url("https://www.bilibili.com/video/BV123?p=2", 125)
        self.assertIn("p=2", url)
        self.assertIn("t=125", url)

    def test_chunk_citation(self) -> None:
        chunk = KnowledgeChunk(
            id="chunk-1",
            video_id="video-1",
            start_seconds=65,
            end_seconds=90,
            text="Evidence text",
        )
        citation = chunk.citation("https://www.youtube.com/watch?v=abc")
        self.assertIn("00:01:05", citation)
        self.assertIn("t=65", citation)

    def test_chunk_rejects_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeChunk("c", "v", 10, 10, "text")


if __name__ == "__main__":
    unittest.main()
