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

        def estimate(src, model, mapping, scratch, stage, **kwargs):
            self.assertEqual((src, model), (source, duet_separator.MODEL))
            result = {}
            for name, sample in (("male", first), ("female", second)):
                target = scratch / (mapping[name] + ".wav")
                sf.write(target, np.column_stack((sample, sample)), rate, subtype="FLOAT")
                result[mapping[name]] = str(target)
            return result, 1

        with patch.object(duet_separator, "duet_available", return_value=True), \
                patch("cover_pipeline._two_stems", side_effect=estimate):
            for swap in (False, True):
                paths = duet_separator.split_duet(source, self.root, swap=swap)
                data = []
                for name in ("male", "female", "residual"):
                    stem, sr = sf.read(paths["duet_" + name], dtype="float32")
                    self.assertEqual((stem.shape, sr), ((n,), rate))
                    data.append(stem)
                np.testing.assert_array_equal(data[0], second if swap else first)
                np.testing.assert_array_equal(data[1], first if swap else second)
                np.testing.assert_allclose(data[2], remainder, atol=1e-7)
                np.testing.assert_allclose(sum(data), mixture, atol=1e-7)

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
