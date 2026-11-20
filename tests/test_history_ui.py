"""Exercise the desktop library via the same snapshots used by its Gradio events."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PLUGIN = Path(__file__).resolve().parents[1] / "Applio/tabs/plugins/installed/CoverStudio/plugin.py"
spec = importlib.util.spec_from_file_location("history_ui_plugin", PLUGIN)
ui = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ui
spec.loader.exec_module(ui)


class HistoryViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        redirect = patch.object(ui.pipeline, "OUTPUT", self.output)
        redirect.start()
        self.addCleanup(redirect.stop)

    def add_work(self, suffix, song):
        folder = self.output / f"run_20261120_190000_{suffix}"
        folder.mkdir()
        (folder / "cover.wav").write_bytes(b"finished fixture")
        (folder / "job.json").write_text(json.dumps({"original_song": song, "vocal_gain": 2.5}), encoding="utf-8")
        return folder.name

    def test_refresh_and_manage_do_not_autoselect_or_keep_an_invisible_work(self):
        absent = ui._history_refresh()
        self.assertIsNone(absent[1])
        self.assertFalse(absent[4]["visible"])
        self.assertFalse(absent[6]["visible"])
        selected = self.add_work("bbbb", "Alpha.wav")
        self.add_work("aaaa", "Beta.wav")
        listed = ui._history_refresh()
        self.assertIn("Alpha", listed[0])
        self.assertIsNone(listed[1], "new results must not open an editing form")
        event = ui.gr.EventData(None, {"run_id": selected})
        selection = ui._history_pick(event)
        self.assertTrue(selection[2]["visible"])
        self.assertEqual(selection[0], selected)
        self.assertEqual(selection[4:6], ("Alpha", 2.5))
        narrowed = ui._history_refresh("Beta", selected=selected)
        self.assertNotIn("Alpha", narrowed[0])
        self.assertIsNone(narrowed[1])
        self.assertTrue(narrowed[6]["visible"], "filters remain available to clear the query")
        self.assertFalse(narrowed[7]["visible"])

    def test_moving_final_item_out_and_back_synchronizes_all_panels(self):
        run = self.add_work("cccc", "One.wav")
        moved = ui._history_trash(run)
        self.assertIsNone(moved[1])
        self.assertTrue(moved[4]["visible"])
        self.assertTrue(moved[5]["interactive"])
        self.assertEqual(moved[2]["value"], run)
        self.assertFalse(moved[6]["visible"])
        restored = ui._history_restore(run)
        self.assertEqual(restored[1], run)
        self.assertTrue(restored[6]["visible"])
        self.assertFalse(restored[4]["visible"])
        self.assertFalse(restored[5]["interactive"])
        self.assertIn("One", restored[0])

    def test_untrusted_title_stays_text_in_both_card_and_inspector(self):
        run = self.add_work("dddd", "My Demo.wav")
        renamed = ui._history_rename(run, '<b>我的歌</b> "2"')
        for rendered in (renamed[0], renamed[8]):
            self.assertIn("&lt;b&gt;", rendered)
            self.assertNotIn("<b>我的歌</b>", rendered)

    def test_manage_control_is_a_button_with_expanded_state(self):
        run = self.add_work("eeee", "Controls.wav")
        rendered = ui._history_html(ui._history_rows(ui.library.list_works(self.output)), 1)
        self.assertIn('class="history-pick"', rendered)
        self.assertIn('aria-expanded="false"', rendered)
        self.assertNotIn('aria-pressed=', rendered)


if __name__ == "__main__":
    unittest.main()
