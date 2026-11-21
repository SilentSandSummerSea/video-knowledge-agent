import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_knowledge_agent.project_env import project_secret


class ProjectEnvTests(unittest.TestCase):
    def test_project_file_precedes_process_environment(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("VIDEO_KNOWLEDGE_TEST_TMP")) as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local credentials\nGROQ_API_KEY='file-groq'\nDEEPSEEK_API_KEY=\"file-deepseek\"\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GROQ_API_KEY": "env-groq"}):
                self.assertEqual(project_secret("GROQ_API_KEY", env_file=env_file), "file-groq")
                self.assertEqual(project_secret("DEEPSEEK_API_KEY", env_file=env_file), "file-deepseek")
                self.assertEqual(project_secret("GROQ_API_KEY", "explicit", env_file=env_file), "explicit")

    def test_missing_or_empty_file_value_falls_back_to_environment(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("VIDEO_KNOWLEDGE_TEST_TMP")) as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GROQ_API_KEY=\n", encoding="utf-8")
            with patch.dict(os.environ, {"GROQ_API_KEY": "env-groq"}):
                self.assertEqual(project_secret("GROQ_API_KEY", env_file=env_file), "env-groq")
                self.assertEqual(project_secret("GROQ_API_KEY", env_file=Path(directory) / "missing"), "env-groq")


if __name__ == "__main__":
    unittest.main()
