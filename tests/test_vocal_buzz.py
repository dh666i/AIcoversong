"""Protect singing and normal frication when removing generator-rate ringing."""
import unittest

import numpy as np
from scipy import signal

from tools.vocal_buzz import reduce_periodic_buzz


class VocalBuzzTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.t = np.arange(self.sr) / self.sr
        self.rng = np.random.default_rng(71)
        self.sibilant = self.rng.normal(size=self.sr).astype("float32") * .035
        # Broadband pulse train whose high-frequency energy resembles a consonant.
        wave = self.rng.normal(size=self.sr // 100)
        self.pulses = (np.tile(wave, 100) * .035).astype("float32")
        self.region = (self.t > .35) & (self.t < .48)

    def test_pulses_on_unvoiced_consonant(self):
        voice = self.sibilant.copy()
        voice[self.region] += self.pulses[self.region] * 3
        corrected, report = reduce_periodic_buzz(voice, self.sibilant, self.sr)
        self.assertTrue(0 < report["affected_seconds"] <= .16)
        np.testing.assert_array_equal(corrected[~self.region], voice[~self.region])
        sos = signal.butter(3, [1400, 7000], fs=self.sr, btype="bandpass", output="sos")
        band = signal.sosfilt(sos, [voice, corrected], axis=1)[:, 6000:7300]
        correlations = [np.corrcoef(x[:-160], x[160:])[0, 1] for x in band]
        self.assertGreater(correlations[0], .8)
        self.assertLess(correlations[1], .45)
        self.assertLess(abs(20 * np.log10(np.linalg.norm(band[0]) / np.linalg.norm(band[1]))), 3)

    def test_true_100hz_note_and_transposed_reference(self):
        phases = (100 * self.t, 200 * self.t, 205 * self.t + 2 * self.t**2)
        for phase in phases:
            with self.subTest(pitch_increment=phase[1]):
                source = np.sin(2 * np.pi * phase).astype("float32") * .15
                source += .035 * np.sin(2 * np.pi * 21 * phase)
                voice = self.pulses + .15 * np.sin(2 * np.pi * 100 * self.t).astype("float32")
                corrected, report = reduce_periodic_buzz(voice, source, self.sr)
                np.testing.assert_array_equal(corrected, voice)
                self.assertEqual(report["spans"], [])

    def test_broadband_unvoiced_and_empty_source(self):
        for voice, source in [(self.sibilant, self.sibilant * .8), (self.pulses, np.zeros_like(self.pulses))]:
            corrected, report = reduce_periodic_buzz(voice, source, self.sr)
            np.testing.assert_array_equal(corrected, voice)
            self.assertEqual(report["affected_seconds"], 0)
