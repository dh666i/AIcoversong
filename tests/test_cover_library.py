"""Unit tests for the local works-library boundary and versioned remixes."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import cover_library as library


class CoverLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "output"
        self.root.mkdir()
        self.run = self._make_run("run_20261118_120000_aabbccdd", refined=True)

    def tearDown(self):
        self.temp.cleanup()

    def _make_run(self, name: str, *, refined: bool = False) -> Path:
        run = self.root / name
        run.mkdir()
        source = run / "source.mp3"
        source.write_bytes(b"source")
        (run / "cover.wav").write_bytes(b"cover")
        job = {
            "song": str(source), "original_song": "D:/songs/My Song [demo].mp3",
            "run_dir": str(run), "vocal_gain": 0.0, "instrumental_gain": 0.0,
            "clarity": 0.75, "reverb": 1.0, "backing": "original",
        }
        (run / "job.json").write_text(json.dumps(job), encoding="utf-8")
        for filename in ("instrumental.wav", "converted_vocals.wav"):
            (run / filename).write_bytes(b"audio")
        if refined:
            for filename in ("lead_dry.wav", "lead_reverb.wav", "backing_vocals.wav", "converted_backing.wav"):
                (run / filename).write_bytes(b"audio")
        else:
            (run / "vocals.wav").write_bytes(b"audio")
        return run

    def test_list_and_get_work_use_safe_metadata(self):
        rows = library.list_works(self.root)
        self.assertEqual([row["run_id"] for row in rows], [self.run.name])
        work = library.get_work(self.run.name, self.root)
        self.assertEqual(work["title"], "My Song")
        self.assertEqual(Path(work["cover"]).resolve(), (self.run / "cover.wav").resolve())

    def test_paths_outside_output_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "run_bad").mkdir()
        with self.assertRaises(ValueError):
            library.get_work(outside / "run_bad", self.root)
        with self.assertRaises(ValueError):
            library.trash_work(Path(self.temp.name) / "../run_bad", self.root)

    def test_rename_only_updates_work_metadata(self):
        renamed = library.rename_work(self.run.name, "我的成品", self.root)
        self.assertEqual(renamed["title"], "我的成品")
        self.assertTrue((self.run / "source.mp3").exists())
        data = json.loads((self.run / "work.json").read_text(encoding="utf-8"))
        self.assertEqual(data["title"], "我的成品")

    def test_trash_and_restore_are_reversible(self):
        removed = library.trash_work(self.run.name, self.root)
        self.assertFalse(self.run.exists())
        self.assertTrue(Path(removed["path"]).is_dir())
        restored = library.restore_work(removed["path"], self.root)
        self.assertEqual(restored["run_id"], self.run.name)
        self.assertTrue(self.run.is_dir())

    def test_restore_by_run_id_from_trash(self):
        library.trash_work(self.run.name, self.root)
        restored = library.restore_work(self.run.name, self.root)
        self.assertEqual(restored["run_id"], self.run.name)

    def test_remix_creates_new_version_and_keeps_parent(self):
        calls = []
        fake = types.ModuleType("vocal_mix")

        def mix_cover(**kwargs):
            calls.append(kwargs)
            Path(kwargs["output_mix"]).write_bytes(b"new cover")
            Path(kwargs["output_vocal"]).write_bytes(b"new vocal")
            return {"mix_gain_db": -1.0}

        fake.mix_cover = mix_cover
        old = sys.modules.get("vocal_mix")
        sys.modules["vocal_mix"] = fake
        try:
            result = library.remix_work(self.run.name, output_root=self.root, clarity=0.9, reverb=0.5)
        finally:
            if old is None:
                sys.modules.pop("vocal_mix", None)
            else:
                sys.modules["vocal_mix"] = old
        new_run = Path(result["path"])
        self.assertNotEqual(new_run, self.run)
        self.assertTrue(self.run.exists())
        self.assertTrue((new_run / "cover.wav").is_file())
        self.assertEqual(result["parent_run"], self.run.name)
        self.assertEqual(calls[0]["clarity"], 0.9)
        job = json.loads((new_run / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["parent_run"], self.run.name)

    def test_remix_requires_stems(self):
        basic = self._make_run("run_20261118_120001_eeff0011", refined=False)
        (basic / "converted_vocals.wav").unlink()
        with self.assertRaisesRegex(ValueError, "必要文件"):
            library.remix_work(basic.name, output_root=self.root)


if __name__ == "__main__":
    unittest.main()
