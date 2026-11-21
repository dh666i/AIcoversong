"""Consonant treatment is local to the selected voice and never generates missing phonemes."""
import unittest

import numpy as np
from scipy import signal

from tools.vocal_mix import restore_consonant_presence


class ArticulationTests(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        t = np.arange(self.sr * 2) / self.sr
        voiced = .07 * np.sin(2 * np.pi * 180 * t) + .03 * np.sin(2 * np.pi * 540 * t)
        consonants = ((.32 <= t) & (t < .42)) | ((1.05 <= t) & (t < 1.15))
        noise = np.random.default_rng(7).normal(size=len(t))
        sos = signal.butter(4, [2400, 6500], btype="bandpass", fs=self.sr, output="sos")
        noise = .12 * signal.sosfilt(sos, noise) * consonants
        voiced[(t >= 1.7) | consonants] = 0
        self.original = (voiced + noise).astype("float32")
        self.converted = (voiced + .3 * noise).astype("float32")
        self.syllable = (t > .335) & (t < .405)
        self.vowel = (t > .60) & (t < .85)

    def apply(self, converted, clarity):
        return restore_consonant_presence(converted, self.original, self.sr, clarity)

    def test_weak_consonants_improve_but_vowels_stay_unchanged(self):
        corrected, report = self.apply(self.converted, .75)
        db = 20 * np.log10(np.linalg.norm(corrected[self.syllable]) /
                            np.linalg.norm(self.converted[self.syllable]))
        self.assertGreater(db, 2.0)
        self.assertLessEqual(report["peak_gain_db"], .75 * 5)
        self.assertGreater(report["affected_seconds"], .12)
        np.testing.assert_array_equal(corrected[self.vowel], self.converted[self.vowel])
        self.assertEqual(len(corrected), len(self.original))

    def test_no_source_audio_copied_during_missing_consonants(self):
        silent = self.converted.copy()
        silent[int(.26 * self.sr):int(.50 * self.sr)] = 0
        corrected, _ = self.apply(silent, 1)
        np.testing.assert_array_equal(corrected[self.syllable], 0)

    def test_bypass_and_already_clear_vocal(self):
        for samples, clarity in [(self.converted, 0), (self.original, 1), (.6 * self.original, 1),
                                 (self.original * 0, 1)]:
            with self.subTest(clarity=clarity):
                corrected, report = self.apply(samples, clarity)
                np.testing.assert_array_equal(corrected, samples)
                self.assertEqual(report["peak_gain_db"], 0)


if __name__ == "__main__":
    unittest.main()
