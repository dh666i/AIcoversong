"""Duet separator routing and additive-stem invariants, without model downloads."""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import duet_separator
from cover_pipeline import LEAD_CONFIG, LEAD_MODEL, _duet, _write_json, stage_fraction


class DuetSeparatorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_missing_model_has_install_command_and_does_not_create_files(self):
        with patch.object(duet_separator, "MODEL_DIR", self.root):
            self.assertFalse(duet_separator.duet_available())
            with self.assertRaisesRegex(FileNotFoundError, "download_assets.py --duet"):
                duet_separator.split_duet(self.root / "vocals.wav", self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_mono_mixture_solos_overlap_and_swapped_assignments(self):
        rate = 8000
        n = duet_separator.BLOCK_FRAMES + 1024
        time = np.arange(n, dtype=np.float32) / rate
        first = (np.sin(2 * np.pi * 170 * time) * .25).astype(np.float32)
        second = (np.sin(2 * np.pi * 280 * time) * .22).astype(np.float32)
        first[n * 2 // 3:] = 0
        second[:n // 3] = 0
        remainder = (.012 * np.cos(2 * np.pi * 37 * time)).astype(np.float32)
        mixture = first + second + remainder
        source = self.root / "vocals.wav"
        sf.write(source, mixture, rate, subtype="FLOAT")

        (self.root / LEAD_MODEL).touch()
        (self.root / LEAD_CONFIG).touch()
        calls = []

        def estimate(src, model, mapping, scratch, stage, **kwargs):
            calls.append((src.name, model))
            result = {}
            if model == LEAD_MODEL:
                samples = (("Vocals", first), ("Instrumental", second))
            elif src == source:
                samples = (("male", first + second), ("female", second * 0))
            else:
                samples = (("male", first if src.stem == "lead" else first * 0),
                           ("female", second if src.stem == "backing" else second * 0))
            for name, sample in samples:
                target = scratch / (mapping[name] + ".wav")
                sf.write(target, np.column_stack((sample, sample)), rate, subtype="FLOAT")
                result[mapping[name]] = str(target)
            return result, 1

        # The reconciliation algorithm has its own signal tests. Supply corrected
        # outputs here to exercise the full worker wiring, including late swap.
        def reconcile(male, female, lead, backing, out_male, out_female):
            self.assertEqual((male.stem, female.stem), ("whole-male", "whole-female"))
            self.assertEqual([lead[r].stem for r in ("male", "female")], ["lead-male", "lead-female"])
            self.assertEqual([backing[r].stem for r in ("male", "female")], ["backing-male", "backing-female"])
            for path, target in ((lead["male"], out_male), (backing["female"], out_female)):
                path.replace(target)
            return {"checked": True}

        with patch.object(duet_separator, "duet_available", return_value=True), \
                patch.object(duet_separator, "MODEL_DIR", self.root), \
                patch("cover_pipeline._two_stems", side_effect=estimate), \
                patch("duet_harmony.reconcile_harmony", side_effect=reconcile):
            for swap in (False, True):
                calls.clear()
                paths = duet_separator.split_duet(source, self.root, swap=swap)
                self.assertEqual(calls, [("vocals.wav", duet_separator.MODEL), ("vocals.wav", LEAD_MODEL),
                                         ("lead.wav", duet_separator.MODEL), ("backing.wav", duet_separator.MODEL)])
                self.assertEqual(paths["duet_harmony_review"], {"checked": True})
                data = []
                for name in ("male", "female", "residual"):
                    stem, sr = sf.read(paths["duet_" + name], dtype="float32")
                    self.assertEqual((stem.shape, sr), ((n,), rate))
                    data.append(stem)
                np.testing.assert_array_equal(data[0], second if swap else first)
                np.testing.assert_array_equal(data[1], first if swap else second)
                np.testing.assert_allclose(data[2], remainder, atol=1e-7)
                np.testing.assert_allclose(sum(data), mixture, atol=1e-7)

    def test_missing_harmony_model_reports_base_installer(self):
        with patch.object(duet_separator, "duet_available", return_value=True), \
                patch.object(duet_separator, "MODEL_DIR", self.root):
            with self.assertRaisesRegex(FileNotFoundError, "主唱与和声"):
                duet_separator.split_duet(self.root / "vocals.wav", self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_progress_across_harmony_review_and_optional_dereverb(self):
        log = self.root / "duet.log"
        contents = b""
        for i in range(1, 7):
            contents += f"duet-separate-pass {i}/6\n".encode()
            log.write_bytes(contents)
            self.assertAlmostEqual(stage_fraction("duet", log), (i - 1) / 6)
            contents += b"0123456789" * 2000 + b" 5/10 [0:01]\n"
            log.write_bytes(contents)
            self.assertAlmostEqual(stage_fraction("duet", log), (i - .5) / 6)
            contents += b"10/10 [0:02]\n"

    def test_worker_uses_uploaded_roles_directly(self):
        sample = (.2 * np.sin(np.arange(16000) * .013)).astype(np.float32)
        stems = {}
        for role, scale in (("male", .5), ("female", 1.0)):
            path = self.root / (role + ".wav")
            sf.write(path, sample * scale, 44100, subtype="FLOAT")
            stems[role] = str(path)
        _write_json(self.root / "separate_result.json", {"vocals": stems["male"]})
        with patch("duet_separator.split_duet", side_effect=AssertionError("unexpected model call")):
            result = _duet({"duet_mode": "male", "duet_sources": stems, "duet_swap": True}, self.root)
        self.assertEqual(result["duet_separator"]["method"], "uploaded")
        retained, _ = sf.read(result["duet_preserved"], dtype="float32")
        np.testing.assert_array_equal(retained[:, 0], sample)

    @unittest.skipUnless(find_spec("scipy"), "sample-rate conversion uses separator-env")
    def test_resampled_estimates_follow_original_stereo_grid(self):
        rate, model_rate = 48000, 44100
        frames = rate // 10 + 1
        source = self.root / "vocals.wav"
        reference = np.column_stack((.2 * np.sin(np.arange(frames) * .035),
                                     .1 * np.sin(np.arange(frames) * .04))).astype(np.float32)
        sf.write(source, reference, rate, subtype="FLOAT")
        scratch = self.root / "scratch"
        scratch.mkdir()
        paths = []
        for name, factor in (("first", .25), ("second", .75)):
            estimate = scratch / (name + ".wav")
            sf.write(estimate, reference[:model_rate // 10] * factor, model_rate, subtype="FLOAT")
            paths.append(estimate)
        results = duet_separator._write_outputs(source, *paths, scratch, self.root)
        reconstructed = np.zeros_like(reference)
        for result in results.values():
            samples, sr = sf.read(result, dtype="float32", always_2d=True)
            self.assertEqual((sr, samples.shape), (rate, reference.shape))
            reconstructed += samples
        np.testing.assert_allclose(reconstructed, reference, atol=1e-7)

    def test_large_estimated_balance_difference_reports_weaker_track(self):
        scratch = self.root / "estimates"
        scratch.mkdir()
        male, female = scratch / "male.wav", scratch / "female.wav"
        source = self.root / "vocals.wav"
        sf.write(male, np.full(1200, .008), 24000, subtype="FLOAT")
        sf.write(female, np.full(1200, .16), 24000, subtype="FLOAT")
        sf.write(source, np.full(1200, .168), 24000, subtype="FLOAT")
        note = duet_separator._write_outputs(source, male, female, scratch, self.root)["duet_balance_note"]
        self.assertIn("男声", note)
        self.assertIn("26 dB", note)
