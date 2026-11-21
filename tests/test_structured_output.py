import unittest

from video_knowledge_agent.structured_output import loads_json_object


class StructuredOutputTests(unittest.TestCase):
    def test_parses_fenced_and_wrapped_json(self):
        self.assertEqual(loads_json_object('```json\n{"ok": true}\n```'), {"ok": True})
        self.assertEqual(loads_json_object('result:\n{"ok": true}\nend'), {"ok": True})


if __name__ == "__main__":
    unittest.main()
