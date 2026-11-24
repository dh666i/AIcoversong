"""Exercise the desktop library via the same snapshots used by its Gradio events."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
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
        self.assertFalse(absent[3]["visible"])
        self.assertFalse(absent[4]["visible"])
        selected = self.add_work("bbbb", "Alpha.wav")
        self.add_work("aaaa", "Beta.wav")
        listed = ui._history_refresh()
        self.assertIn("Alpha", listed[0])
        self.assertIsNone(listed[1], "new results must not open an editing form")
        event = ui.gr.EventData(None, {"run_id": selected})
        selection = ui._history_pick(event)
        self.assertTrue(selection[2]["visible"])
        self.assertEqual(selection[0], selected)
        self.assertEqual((selection[4], selection[5]["value"]), ("Alpha", 2.5))
        self.assertTrue(selection[9]["visible"])
        narrowed = ui._history_refresh("Beta", selected=selected)
        self.assertNotIn("Alpha", narrowed[0])
        self.assertIsNone(narrowed[1])
        self.assertTrue(narrowed[3]["visible"], "filters remain available to clear the query")
        self.assertFalse(narrowed[4]["visible"])

    def test_permanent_delete_and_cancel_synchronize_panels(self):
        run = self.add_work("cccc", "One.wav")
        self.assertEqual(ui._history_delete([], selected=run), ui.gr.skip())
        self.assertTrue((self.output / run).is_dir())
        deleted = ui._history_delete([run], selected=run, bulk_ids=[run])
        self.assertIsNone(deleted[1])
        self.assertIn("已删除 1 首", deleted[2])
        self.assertFalse(deleted[3]["visible"])
        self.assertFalse(deleted[4]["visible"])
        self.assertEqual(deleted[-5]["value"], [])
        self.assertFalse(deleted[-4]["visible"])
        self.assertFalse((self.output / run).exists())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_untrusted_title_stays_text_in_both_card_and_inspector(self):
        run = self.add_work("dddd", "My Demo.wav")
        renamed = ui._history_rename(run, '<b>我的歌</b> "2"')
        for rendered in (renamed[0], renamed[5]):
            self.assertIn("&lt;b&gt;", rendered)
            self.assertNotIn("<b>我的歌</b>", rendered)

    def test_filtered_bulk_choices_include_items_beyond_the_card_limit(self):
        ids = [self.add_work(str(n), f"demo-{n}") for n in range(63)]
        other = self.add_work("other", "different.wav")
        updates = ui._history_refresh("demo-", "title", bulk_ids=[ids[20], other])
        choices = updates[-5]
        self.assertEqual(len(choices["choices"]), 63)
        self.assertEqual(choices["value"], [ids[20]])
        self.assertEqual(updates[0].count('class="history-entry"'), 60)
        removed = ui._history_delete(ids, "demo-", bulk_ids=ids)
        self.assertEqual(removed[-5]["value"], [])
        self.assertIn("已删除 63 首", removed[2])
        self.assertTrue((self.output / other).is_dir())
        self.assertEqual([p.name for p in self.output.iterdir()], [other])

    def test_manage_control_is_a_button_with_expanded_state(self):
        run = self.add_work("eeee", "Controls.wav")
        rendered = ui._history_html(ui._history_rows(ui.library.list_works(self.output)), 1)
        self.assertIn('class="history-pick"', rendered)
        self.assertIn('aria-expanded="false"', rendered)
        self.assertNotIn('aria-pressed=', rendered)

    def test_downloads_bind_to_the_selected_work_and_available_stem(self):
        run_id = self.add_work("download", "旧歌.wav")
        folder = self.output / run_id
        (folder / "instrumental.wav").write_bytes(b"fixture")
        updates = ui._history_pick(ui.gr.EventData(None, {"run_id": run_id}))
        note = updates[3]
        self.assertIn("下载分轨", note)
        self.assertIn(f'data-export-run="{run_id}"', note)
        self.assertIn('data-export-track="instrumental"', note)
        self.assertNotIn('data-export-track="converted"', note)
        result = ui._result_html({"mix": folder / "cover.wav", "instrumental": folder / "instrumental.wav"}, "旧歌")
        for track in ("mix", "instrumental"):
            self.assertIn(f'data-export-track="{track}"', result)

    def test_card_durations_read_wav_headers_without_mounting_media(self):
        run_id = self.add_work("abcd", "Preview.wav")
        path = self.output / run_id / "cover.wav"
        for seconds in (7, 9):
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * (8000 * seconds))
            markup = ui._history_html(ui.library.list_works(self.output), 1)
            self.assertIn(f'时长 0:0{seconds}', markup)
            self.assertIn('aria-label="试听：Preview"', markup)
            self.assertIn('aria-label="下载：Preview"', markup)
            self.assertNotIn("<audio", markup)
        path.write_bytes(b"unreadable header")
        markup = ui._history_html(ui.library.list_works(self.output), 1)
        self.assertNotIn("work-duration", markup)
        self.assertIn('aria-label="试听：Preview"', markup)


if __name__ == "__main__":
    unittest.main()
