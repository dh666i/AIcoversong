"""Unit tests for the local works-library boundary and versioned remixes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import cover_library as library
import audio_export


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
            library.delete_work(Path(self.temp.name) / "../run_bad", self.root)

    def test_rename_only_updates_work_metadata(self):
        renamed = library.rename_work(self.run.name, "我的成品", self.root)
        self.assertEqual(renamed["title"], "我的成品")
        self.assertTrue((self.run / "source.mp3").exists())
        data = json.loads((self.run / "work.json").read_text(encoding="utf-8"))
        self.assertEqual(data["title"], "我的成品")

    def test_delete_only_removes_the_selected_directory(self):
        sibling = self._make_run("run_sibling")
        legacy = self.root / ".trash" / self.run.name
        legacy.mkdir(parents=True)
        (legacy / "cover.wav").write_bytes(b"legacy")
        library.delete_work(self.run.name, self.root)
        self.assertFalse(self.run.exists())
        self.assertTrue((sibling / "cover.wav").is_file())
        self.assertEqual((legacy / "cover.wav").read_bytes(), b"legacy")

    def test_incomplete_run_is_not_deletable(self):
        (self.run / "cover.wav").unlink()
        with self.assertRaises(ValueError):
            library.delete_work(self.run.name, self.root)
        self.assertTrue((self.run / "job.json").is_file())

    def test_bulk_delete_validates_all_ids_then_reports_each_work(self):
        with self.assertRaises(ValueError):
            library.delete_selected([self.run.name, "../run_other"], output_root=self.root)
        self.assertTrue(self.run.is_dir())
        deleted, failed = library.delete_selected([self.run.name, self.run.name, "run_missing"], output_root=self.root)
        self.assertEqual(deleted, [self.run.name])
        self.assertEqual(list(failed), ["run_missing"])
        self.assertFalse(self.run.exists())

    def test_busy_audio_stays_in_the_library_until_retry(self):
        library.rename_work(self.run.name, "Take 2", self.root)
        cover = (self.run / "cover.wav").resolve()
        real_unlink = Path.unlink

        def busy(path):
            if path == cover:
                raise PermissionError("in use")
            real_unlink(path)
        with patch.object(Path, "unlink", busy):
            deleted, failed = library.delete_selected([self.run.name], output_root=self.root)
        self.assertEqual(deleted, [])
        self.assertIn(self.run.name, failed)
        rows = library.list_works(self.root)
        self.assertEqual([(work["run_id"], work["title"]) for work in rows], [(self.run.name, "Take 2")])
        self.assertEqual(library.delete_selected([self.run.name], output_root=self.root), ([self.run.name], {}))

    def test_delete_rejects_linked_run_and_leaves_nested_link_target_intact(self):
        outside = Path(self.temp.name) / "other_files"
        outside.mkdir()
        marker = outside / "keep.wav"
        marker.write_bytes(b"keep")
        alias = self.root / "run_alias"
        def link_directory(link, target):
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(target), str(link))
            else:
                link.symlink_to(target, target_is_directory=True)

        link_directory(alias, outside)
        with self.assertRaises(ValueError):
            library.delete_work(alias, self.root)
        link_directory(self.run / "external_dir", outside)
        library.delete_work(self.run.name, self.root)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_export_contains_only_selected_finished_mixes_with_unique_safe_names(self):
        second = self._make_run("run_other")
        for run in (self.run, second):
            library.rename_work(run.name, "../同名:作品", self.root)
        target = Path(self.temp.name) / "selected.zip"
        library.export_selected([self.run.name, second.name, second.name], target, output_root=self.root)
        with ZipFile(target) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), 2)
            self.assertEqual(len(set(names)), 2)
            for name in names:
                self.assertTrue(name.endswith(".wav"))
                self.assertNotRegex(name, r'[/\\:]')
                self.assertEqual(archive.read(name), b"cover")

    def test_export_track_limits_sources_and_preserves_existing_download(self):
        folder = Path(self.temp.name)
        tracks = library.list_export_tracks(self.run.name, output_root=self.root)
        self.assertEqual(set(tracks), {"mix", "instrumental", "backing"})
        self.assertEqual(tracks["instrumental"][0], (self.run / "instrumental.wav").resolve())
        with self.assertRaises(ValueError):
            library.export_track(self.run.name, "../job.json", folder, output_root=self.root)
        with self.assertRaises(ValueError):
            library.export_track(self.run.name, "mix", folder, audio_format="../wav", output_root=self.root)
        library.rename_work(self.run.name, "CON.1", self.root)
        track = library.export_track(self.run.name, "instrumental", folder, output_root=self.root)
        self.assertEqual(track.name, "_CON.1 伴奏.wav")
        self.assertEqual(track.read_bytes(), b"audio")
        track.write_bytes(b"previous download")
        with self.assertRaises(FileExistsError):
            library.export_track(self.run.name, "instrumental", folder, output_root=self.root)
        self.assertEqual(track.read_bytes(), b"previous download")

    def test_failed_conversion_leaves_no_partial_download(self):
        folder = Path(self.temp.name) / "downloads"
        folder.mkdir()
        extra = self._make_run("run_second")
        calls = []

        def encode(source, target):
            calls.append(source.parent.name)
            target.write_bytes(b"incomplete")
            if source.parent.name == extra.name:
                raise ValueError("conversion failed")

        with patch.object(library, "write_audio", encode):
            with self.assertRaisesRegex(ValueError, "conversion failed"):
                library.export_selected([self.run.name, extra.name], folder / "mixes.zip",
                                        audio_format="mp3", output_root=self.root)
            with self.assertRaisesRegex(ValueError, "conversion failed"):
                library.export_track(extra.name, "mix", folder, audio_format="mp3", output_root=self.root)
        self.assertEqual(calls, [self.run.name, extra.name, extra.name])
        self.assertEqual(list(folder.iterdir()), [])

    @unittest.skipUnless(audio_export.FFMPEG.is_file(), "requires installed FFmpeg")
    def test_download_formats_and_batch_contents_decode_with_correct_channels(self):
        import numpy as np
        import soundfile as sf

        sr = 44100
        t = np.arange(sr) / sr
        reference = np.stack((.2 * np.sin(2*np.pi*300*t), .15 * np.sin(2*np.pi*540*t)), axis=1)
        master = self.run / "cover.wav"
        sf.write(master, reference, sr, subtype="PCM_24")
        source = master.read_bytes()
        expected, _ = sf.read(master, dtype="float32")
        folder = Path(self.temp.name)
        for fmt in library.EXPORT_FORMATS:
            with self.subTest(fmt=fmt):
                exported = library.export_track(self.run.name, "mix", folder, audio_format=fmt, output_root=self.root)
                decoded = subprocess.check_output(
                    [str(audio_export.FFMPEG), "-v", "error", "-i", str(exported),
                     "-ar", str(sr), "-f", "f32le", "-"])
                samples = np.frombuffer(decoded, dtype="<f4").reshape(-1, 2)
                # AAC may append up to one codec frame of padding.
                self.assertLess(abs(len(samples) - sr), 1024)
                for channel in range(2):
                    self.assertGreater(np.corrcoef(samples[1500:40000, channel], expected[1500:40000, channel])[0, 1], .99)
                if fmt in {"wav", "aiff", "flac"}:
                    np.testing.assert_array_equal(samples, expected)
                if fmt == "wav":
                    self.assertEqual(exported.read_bytes(), source)
        archive = folder / "library.zip"
        library.export_selected([self.run.name], archive, output_root=self.root, audio_format="mp3")
        with ZipFile(archive) as zipped:
            self.assertEqual(zipped.namelist(), ["001 - My Song.mp3"])
            self.assertEqual(zipped.read(zipped.namelist()[0]), (folder / "My Song 翻唱.mp3").read_bytes())
        self.assertEqual(master.read_bytes(), source)

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
