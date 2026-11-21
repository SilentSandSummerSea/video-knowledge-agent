import unittest
from types import SimpleNamespace

from video_knowledge_agent.local_text import LocalTextClient, LocalTextError, local_text_base_url


class LocalTextTests(unittest.TestCase):
    def test_responses_adapter_uses_local_chat_json_schema(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"chapters":[]}'))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
            )

        client = LocalTextClient(SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ))
        result = client.responses.create(
            model="local-model",
            input=[{"role": "system", "content": "Return JSON"},
                   {"role": "user", "content": "Split chapters"}],
            text={"format": {"type": "json_schema", "name": "chapters",
                             "strict": True, "schema": {"type": "object"}}},
            max_output_tokens=400,
            reasoning={"effort": "none"},
        )
        self.assertEqual(result.output_text, '{"chapters":[]}')
        self.assertEqual(result.usage["input_tokens"], 12)
        self.assertEqual(result.usage["output_tokens"], 5)
        self.assertEqual(calls[0]["response_format"]["json_schema"]["name"], "chapters")
        self.assertEqual(calls[0]["model"], "local-model")
        self.assertEqual(calls[0]["max_tokens"], 400)

    def test_local_mode_refuses_non_loopback_server(self):
        self.assertEqual(local_text_base_url("http://localhost:11434/v1"),
                         "http://localhost:11434/v1/")
        with self.assertRaises(LocalTextError):
            local_text_base_url("https://api.example.com/v1")


if __name__ == "__main__":
    unittest.main()
