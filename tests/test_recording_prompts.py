"""Prompt selection from isolated recording metadata; no user files touched."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import voice_recording as recorder


class RecordingPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.metadata = self.base / "metadata"
        self.recordings = self.base / "recordings"
        self.patches = [patch.object(recorder, "RECORD_INFO", self.metadata),
                        patch.object(recorder, "DATASET", self.recordings)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def save_metadata(self, *positions):
        self.metadata.mkdir(exist_ok=True)
        for order, position in enumerate(positions):
            file = self.metadata / f"prompt-{position}.json"
            file.write_text(json.dumps({"prompt_index": position,
                                        "saved_filename": f"prompt-{position}.wav"}), encoding="utf-8")
            os.utime(file, (1_700_000_000 + order, 1_700_000_000 + order))

    def test_fresh_install_starts_at_first_basic_without_creating_folders(self):
        self.assertEqual(recorder.initial_index(), 0)
        self.assertIn("基础稿 第 1 段", recorder.prompt_label(recorder.initial_index()))
        self.assertFalse(self.metadata.exists())
        self.assertFalse(self.recordings.exists())

    def test_refresh_resumes_first_missing_basic(self):
        self.save_metadata(0, 1, 3)
        self.assertEqual(recorder.initial_index(), 2)
        self.assertEqual(recorder.initial_index(), 2)
        self.save_metadata(2)
        self.assertEqual(recorder.initial_index(), 4)

    def test_completed_basics_advance_to_first_supplement(self):
        first = recorder.first_new_prompt_index()
        self.save_metadata(*range(first))
        self.assertEqual(recorder.initial_index(), first)

    def test_supplement_users_do_not_return_to_missing_basics(self):
        first = recorder.first_new_prompt_index()
        self.save_metadata(0, first, first + 1)
        self.assertEqual(recorder.initial_index(), first + 2)

    def test_completed_supplements_keep_existing_batch_navigation(self):
        first = recorder.first_new_prompt_index()
        last = len(recorder.PROMPTS) - 1
        self.save_metadata(*range(first, len(recorder.PROMPTS)))
        self.assertEqual(recorder.initial_index(), recorder.next_prompt_index(last))

    def test_invalid_metadata_does_not_skip_basic_prompts(self):
        self.metadata.mkdir()
        (self.metadata / "corrupt.json").write_text("{", encoding="utf-8")
        self.save_metadata(-1, len(recorder.PROMPTS) + 5)
        self.assertEqual(recorder.initial_index(), 0)


if __name__ == "__main__":
    unittest.main()
