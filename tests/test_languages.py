import unittest

from video_knowledge_agent.languages import (
    normalize_language,
    normalize_output_language,
    resolve_output_language,
)


class LanguageTests(unittest.TestCase):
    def test_normalizes_common_chinese_english_and_japanese_aliases(self):
        self.assertEqual(normalize_language("中文"), "zh")
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(normalize_language("日语"), "ja")
        self.assertEqual(normalize_language(None), "auto")

    def test_output_language_must_be_explicit(self):
        with self.assertRaises(ValueError):
            normalize_language("auto", allow_auto=False)

    def test_output_defaults_to_detected_source_language(self):
        self.assertEqual(normalize_output_language(None), "source")
        self.assertEqual(resolve_output_language("source", "Japanese"), "ja")
        self.assertEqual(resolve_output_language("zh", "Japanese"), "zh")

    def test_rejects_invalid_language_code(self):
        with self.assertRaises(ValueError):
            normalize_language("not a language")


if __name__ == "__main__":
    unittest.main()
