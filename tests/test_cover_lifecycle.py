"""Task lifecycle regression checks; no audio inference, training or network."""
from __future__ import annotations

import importlib.util
import asyncio
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "Applio/tabs/plugins/installed/CoverStudio/plugin.py"
spec = importlib.util.spec_from_file_location("cover_lifecycle_plugin", PLUGIN)
ui = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ui
spec.loader.exec_module(ui)


class CoverLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.closed = []
        self.gate = None
        self.failure = False
        self.block_start = False
        self.started = threading.Event()
        self.patches = [
            patch.object(ui, "_index_for", return_value=""),
            patch.object(ui, "_result_html", return_value="<audio>finished</audio>"),
            patch.object(ui.pipeline, "run_cover", side_effect=self.fake_cover),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        if self.gate:
            self.gate.set()
        with ui._COVER_TASKS_LOCK:
            for task in ui._COVER_TASKS.values():
                if not task.finished:
                    task.cancel.set()
        deadline = time.monotonic() + 3
        while ui._GPU_WORK.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        for item in reversed(self.patches):
            item.stop()
        with ui._COVER_TASKS_LOCK:
            ui._COVER_TASKS.clear()
        self.assertFalse(ui._GPU_WORK.locked(), "worker did not release the GPU slot")

    def fake_cover(self, *arguments):
        self.calls.append(arguments)
        number = len(self.calls)
        self.started.set()
        try:
            yield {"run": f"run-{number}", "progress": {"state": "running", "steps": ["转换"], "step": 0}}
            if self.gate:
                self.gate.wait(2)
            if self.failure:
                raise RuntimeError("worker test error")
            yield {"run": f"run-{number}", "mix": "test.wav", "progress": {"state": "done", "steps": ["转换"], "elapsed": 1}}
        finally:
            self.closed.append(number)

    def start(self, token=""):
        return ui._start_cover("same-song.wav", "voice.pth", "auto", "original", .75, 1., 0., 0., True, .25, token)

    def wait(self, token):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = ui._task_snapshot(token)
            if snapshot and snapshot[2]:
                return snapshot
            time.sleep(.005)
        self.fail("task failed to reach a terminal state")

    def test_same_upload_can_run_twice_without_again_or_refresh(self):
        first = self.start()
        first_token = first[-2]
        self.wait(first_token)
        finished = ui._poll_cover(first_token)
        self.assertTrue(finished[5]["interactive"])
        self.assertFalse(finished[-1]["active"])
        second = self.start(first_token)
        self.assertNotEqual(first_token, second[-2])
        self.assertEqual(self.wait(second[-2])[1]["progress"]["state"], "done")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertEqual(self.closed, [1, 2])

    def test_failure_releases_slot_and_same_upload_can_retry(self):
        self.failure = True
        token = self.start()[-2]
        self.assertEqual(self.wait(token)[1]["progress"]["state"], "failed")
        self.failure = False
        retry = self.start(token)[-2]
        self.assertEqual(self.wait(retry)[1]["progress"]["state"], "done")

    def test_stop_waits_for_worker_cleanup_before_enabling_start(self):
        self.gate = threading.Event()
        token = self.start()[-2]
        self.assertTrue(self.started.wait(1))
        stopped = ui._stop_cover(token)
        self.assertFalse(stopped[5]["interactive"])
        self.gate.set()
        self.assertEqual(self.wait(token)[1]["progress"]["state"], "stopped")
        self.assertEqual(self.closed, [1])
        retry = self.start(token)[-2]
        self.assertEqual(self.wait(retry)[1]["progress"]["state"], "done")

    def test_double_click_returns_existing_job_instead_of_duplicate(self):
        self.gate = threading.Event()
        token = self.start()[-2]
        duplicate = self.start(token)
        self.assertEqual(token, duplicate[-2])
        self.gate.set()
        self.wait(token)
        self.assertEqual(len(self.calls), 1)

    def test_other_session_cannot_start_or_stop_an_active_job(self):
        self.gate = threading.Event()
        token = self.start()[-2]
        with self.assertRaises(ui.gr.Error):
            self.start("")
        unrelated = ui._stop_cover("not-this-browser-task")
        self.assertEqual(unrelated[0]["state"], "idle")
        self.assertFalse(ui._task_snapshot(token)[3])
        self.gate.set()
        self.wait(token)

    def test_training_cannot_run_alongside_cover(self):
        self.gate = threading.Event()
        token = self.start()[-2]
        with patch.object(ui.trainer, "train_voice") as training:
            snapshot = list(ui._train(True))
            training.assert_not_called()
        self.assertEqual(snapshot[0][0]["state"], "failed")
        self.gate.set()
        self.wait(token)

    def test_training_releases_slot_after_close(self):
        def fake_train(_):
            yield {"state": "running"}
            yield {"state": "done"}
        with patch.object(ui.trainer, "train_voice", side_effect=fake_train):
            stream = ui._train(True)
            next(stream)
            with self.assertRaises(ui.gr.Error):
                self.start()
            stream.close()
        token = self.start()[-2]
        self.wait(token)

    def test_refresh_token_restores_completed_result(self):
        token = self.start()[-2]
        self.wait(token)
        view = ui._poll_cover(token)
        self.assertEqual(view[0]["state"], "done")
        self.assertTrue(view[3]["visible"])
        reset = ui._again(token)
        self.assertEqual(reset[0]["state"], "idle")
        self.assertEqual(reset[-2], "")
        self.assertEqual(reset[1], ui.gr.skip())

    def test_thread_start_failure_does_not_leave_a_busy_job(self):
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start worker")):
            with self.assertRaisesRegex(RuntimeError, "cannot start worker"):
                self.start()
        self.assertFalse(ui._GPU_WORK.locked())
        self.assertEqual(ui._COVER_TASKS, {})
        self.wait(self.start()[-2])

    def test_visibility_survives_actual_gradio_postprocessing(self):
        asyncio.run(self._check_gradio_postprocessing())

    async def _check_gradio_postprocessing(self):
        from gradio.state_holder import SessionState

        with ui.gr.Blocks() as page:
            progress = ui._progress_view("test-progress")
            form, stage = ui.gr.Column(), ui.gr.Column()
            result = ui.gr.HTML()
            again, start, stop = ui.gr.Button(), ui.gr.Button(), ui.gr.Button()
            empty, timer = ui.gr.HTML(), ui.gr.Timer(active=False)
            event = start.click(ui._poll_cover,
                                outputs=[progress, form, stage, result, again, start, stop, empty, timer])
        state = SessionState(page)
        # Reuse one real Gradio session across running, completion, reset and
        # another run, so old component constructor state cannot mask a bug.
        for name, finished, stop_visible, start_enabled in [
            ("running", False, True, False), ("done", True, False, True),
            ("idle", True, False, True), ("running", False, True, False),
            ("failed", True, False, True), ("stopped", True, False, True),
        ]:
            snapshot = None if name == "idle" else (
                "test", {"progress": {"state": name}}, finished, False)
            prediction = ui._cover_view(snapshot)
            self.assertEqual(prediction[1], ui.gr.skip(), "do not remount the upload form")
            updates = [value for value in prediction if isinstance(value, dict) and value.get("__type__") == "update"]
            self.assertEqual(len(updates), len({id(value) for value in updates}))
            wire = await page.postprocess_data(page.fns[event["id"]], prediction, state)
            self.assertEqual(wire[6]["visible"], stop_visible, name)
            self.assertEqual(wire[5]["interactive"], start_enabled, name)
            self.assertEqual(wire[8]["active"], name == "running", name)

    def test_stop_is_not_nested_in_stage_that_changes_visibility(self):
        asyncio.run(self._check_stop_parent())

    async def _check_stop_parent(self):
        with patch.object(ui, "_voices", return_value=[]), patch.object(ui.pipeline, "refine_available", return_value=False):
            with ui.gr.Blocks() as page:
                ui._cover_tab()
        stage = next(block for block in page.blocks.values() if getattr(block, "elem_id", None) == "cover-stage")
        stop = next(block for block in page.blocks.values() if getattr(block, "elem_id", None) == "stop-cover")
        self.assertNotIn(stop, stage.children)
        self.assertFalse(stop.visible)

    def test_training_uses_fresh_updates_after_postprocessing(self):
        def fake_train(_):
            yield {"state": "running"}
            yield {"state": "done"}
        with patch.object(ui.trainer, "train_voice", side_effect=fake_train):
            stream = ui._train_worker(True)
            initial = next(stream)
            # A framework may consume a response in place. Later responses
            # must not reuse that same dictionary for a different component.
            initial[1].clear()
            initial[2].clear()
            complete = next(stream)
            self.assertTrue(complete[1]["visible"])
            self.assertFalse(complete[2]["visible"])
            stream.close()


if __name__ == "__main__":
    unittest.main()
