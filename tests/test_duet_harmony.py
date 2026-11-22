"""Overlapping partners survive role assignment without doubling the original."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from tools.duet_harmony import reconcile_harmony


class DuetHarmonyTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.sr = 12000
        t = np.arange(4 * self.sr + 17) / self.sr
        # Work across an internal chunk boundary, with stereo and a partial frame.
        lead = .14 * np.sin(2*np.pi*186*t) + .035 * np.sin(2*np.pi*744*t)
        harmony = .075 * np.sin(2*np.pi*337*t) + .03 * np.sin(2*np.pi*1348*t)
        self.lead = np.column_stack((lead, .8 * lead)).astype("float32")
        self.harmony = np.column_stack((.6 * harmony, -harmony)).astype("float32")
        self.silence = np.zeros_like(self.lead)

    def run_pair(self, male, female, lead_m, lead_f, back_m, back_f, swap=False):
        files = {}
        for key, data in (("male", male), ("female", female), ("lead_m", lead_m), ("lead_f", lead_f),
                          ("back_m", back_m), ("back_f", back_f)):
            files[key] = self.folder / (key + ".wav")
            sf.write(files[key], data, self.sr, subtype="FLOAT")
        paths = [self.folder / (f"out_{key}.wav") for key in ("male", "female")]
        for output in paths:
            output.unlink(missing_ok=True)
        report = reconcile_harmony(files["male"], files["female"],
                                   {"male": files["lead_m"], "female": files["lead_f"]},
                                   {"male": files["back_m"], "female": files["back_f"]}, *paths, swap=swap)
        out_m, out_f = (sf.read(file, dtype="float32", always_2d=True)[0] for file in paths)
        np.testing.assert_allclose(out_m + out_f, male + female, rtol=0, atol=4e-8)
        return out_m, out_f, report["reassigned_seconds"]

    def test_partial_and_complete_loss_through_chunk_boundary(self):
        lead, h, zero = self.lead, self.harmony, self.silence
        preserved = .45 * h
        preserved[2*self.sr:] = 0
        m, f, report = self.run_pair(lead + h - preserved, preserved, lead, zero, zero, h)
        for region in (slice(self.sr//2, self.sr), slice(2*self.sr+self.sr//2, 3*self.sr+self.sr//2)):
            self.assertLess(np.linalg.norm(f[region] - h[region]), .025 * np.linalg.norm(h[region]))
            self.assertLess(np.linalg.norm(m[region] - lead[region]), .025 * np.linalg.norm(h[region]))
        self.assertGreater(report["female"], 3)
        self.assertEqual(report["male"], 0)

    def test_already_present_ambiguous_same_singer_and_solo_are_untouched(self):
        lead, h, zero = self.lead, self.harmony, self.silence
        cases = [(lead, h, lead, zero, zero, h),
                 (lead + h, zero, lead, zero, h, zero),
                 (lead + h, zero, lead, zero, .8*h, h),
                 (lead, zero, lead, zero, zero, zero),
                 (lead, zero, lead, zero, zero, .25*lead)]
        for case, value in enumerate(cases):
            with self.subTest(case=case):
                m, f, report = self.run_pair(*value)
                np.testing.assert_array_equal(m, value[0])
                np.testing.assert_array_equal(f, value[1])
                self.assertEqual(report, {"male": 0, "female": 0})

    def test_male_harmony_and_explicit_role_swap(self):
        lead, h, zero = self.lead, self.harmony, self.silence
        for swap in (False, True):
            streams = (lead, zero, zero, h) if swap else (zero, lead, h, zero)
            m, f, report = self.run_pair(zero, lead + h, *streams, swap=swap)
            mid = slice(self.sr, 3*self.sr)
            self.assertLess(np.linalg.norm(m[mid] - h[mid]), .025 * np.linalg.norm(h[mid]))
            self.assertGreater(report["male"], 3)

    def test_grid_error_leaves_no_outputs(self):
        lead, h, zero = self.lead, self.harmony, self.silence
        with self.assertRaisesRegex(ValueError, "time grid"):
            self.run_pair(lead, h, lead, zero[:-self.sr], zero, h)
        self.assertFalse((self.folder / "out_male.wav").exists())
