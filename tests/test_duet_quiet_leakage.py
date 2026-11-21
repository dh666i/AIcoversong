"""Quiet duet residue must not become an audible RVC note between phrases."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from duet_mix import _attenuate_quiet_leakage, repair_solo_bleed


class QuietLeakageTests(unittest.TestCase):
    def test_reduces_only_excess_in_quiet_intervals(self):
        sr = 8000
        t = np.arange(sr * 2 + 17) / sr
        tone = np.sin(2 * np.pi * 220 * t)
        phrase = (t < .8) | (t >= 1.5)
        ref = tone * np.where(phrase, .25, .0005)
        voice = tone * np.where(phrase, .25, .015)
        ref = np.column_stack((ref, -ref)).astype(np.float32)
        voice = np.column_stack((voice, -voice)).astype(np.float32)
        before = voice.copy()
        report = _attenuate_quiet_leakage(voice, ref, sr)
        # Opposite-polarity stereo is not silence. Last, partial frame is valid.
        np.testing.assert_array_equal(voice[phrase], before[phrase])
        np.testing.assert_array_equal(voice[t < .82], before[t < .82])
        mid = (t > 1) & (t < 1.3)
        self.assertLess(np.linalg.norm(voice[mid]), .1 * np.linalg.norm(before[mid]))
        self.assertGreater(report["attenuated_seconds"], .4)

    def test_keeps_genuinely_soft_phrase_and_silences_an_absent_stem(self):
        sr = 8000
        t = np.arange(sr) / sr
        data = np.sin(2 * np.pi * 180 * t) * np.where(t < .6, .18, .00075)
        ref = np.column_stack((data, data)).astype(np.float32)
        voice = ref.copy()
        _attenuate_quiet_leakage(voice, ref, sr)
        np.testing.assert_array_equal(voice, ref)
        _attenuate_quiet_leakage(voice, np.zeros_like(ref), sr)
        self.assertFalse(np.any(voice))

    def test_solo_reassignment_preserves_mixture_and_independent_voice(self):
        sr = 16000
        t = np.arange(4 * sr) / sr
        x = .17*np.sin(2*np.pi*180*t) + .07*np.sin(2*np.pi*1200*t)
        # Related leakage, then unrelated partner with the same fundamental.
        partner = .07*np.sin(2*np.pi*180*t) + .045*np.sin(2*np.pi*1540*t)
        y = np.where(t < 2, .45*x, partner)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "target.wav", root / "kept.wav"
            sf.write(a, x, sr, subtype="FLOAT")
            sf.write(b, y, sr, subtype="FLOAT")
            result = repair_solo_bleed(a, b, root, solo_spans=[(.5, 1.5), (2.4, 3.8)])
            new_x = sf.read(result["selected"], dtype="float32")[0]
            new_y = sf.read(result["preserved"], dtype="float32")[0]
            np.testing.assert_allclose(new_x + new_y, x + y, atol=3e-8)
            mid = (.7 < t) & (t < 1.3)
            self.assertLess(np.linalg.norm(new_y[mid]), .2 * np.linalg.norm(y[mid]))
            untouched = (t < .5) | (t >= 1.5)
            np.testing.assert_array_equal(new_y[untouched], y[untouched])
            np.testing.assert_array_equal(new_x[untouched], x[untouched])
            self.assertGreater(result["report"]["affected_seconds"], .85)
            with self.assertRaises(FileExistsError):
                repair_solo_bleed(a, b, root, solo_spans=[(.5, 1.5)])

    def test_overlapping_and_out_of_bounds_windows_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "target.wav", root / "kept.wav"
            for file in (a, b):
                sf.write(file, np.zeros(8000, dtype=np.float32), 8000)
            for invalid in ([(0, .6), (.5, .9)], [(0, 1.1)], [(-.1, .8)]):
                with self.subTest(spans=invalid), self.assertRaises(ValueError):
                    repair_solo_bleed(a, b, root, solo_spans=invalid)
            self.assertFalse((root / "solo_selected.wav").exists())
