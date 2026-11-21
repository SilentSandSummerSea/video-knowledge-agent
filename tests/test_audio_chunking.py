import unittest

from video_knowledge_agent.audio_chunking import plan_audio_chunks


class AudioChunkingTests(unittest.TestCase):
    def test_two_minute_cores_have_five_second_context(self):
        chunks = plan_audio_chunks(305, core_seconds=120, context_seconds=5)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            (chunks[0].core_start, chunks[0].core_end),
            (0, 120),
        )
        self.assertEqual(
            (chunks[0].upload_start, chunks[0].upload_end),
            (0, 125),
        )
        self.assertEqual(
            (chunks[1].upload_start, chunks[1].upload_end),
            (115, 245),
        )
        self.assertEqual(
            (chunks[2].core_start, chunks[2].core_end),
            (240, 305),
        )
        self.assertEqual(
            (chunks[2].upload_start, chunks[2].upload_end),
            (235, 305),
        )


if __name__ == "__main__":
    unittest.main()
