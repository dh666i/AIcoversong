"""Upload validation and decoding into the clocks used by the cover workers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tools import audio_input, cover_pipeline, duet_mix


class InputFormatTests(unittest.TestCase):
    def test_extensions_accept_case_insensitively(self):
        with tempfile.TemporaryDirectory() as folder:
            for ext in audio_input.AUDIO_EXTENSIONS:
                sample = Path(folder) / ("sample" + ext.upper())
                sample.write_bytes(b"test")
                self.assertEqual(cover_pipeline._audio(sample), sample.resolve())
            bad = Path(folder) / "sample.txt"
            bad.write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "扩展名"):
                cover_pipeline._audio(bad)

    @unittest.skipUnless(audio_input.FFMPEG.is_file(), "requires the installed FFmpeg asset")
    def test_wavpack_song_and_aligned_mka_stem(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            time = np.arange(48000) / 48000
            audio = .2 * np.sin(2 * np.pi * 247 * time)
            source = run / "original.wav"
            sf.write(source, audio, 48000, subtype="PCM_24")

            def encode(filename, *options):
                target = run / filename
                subprocess.run([str(audio_input.FFMPEG), "-v", "error", "-i", str(source),
                                *options, str(target)], check=True)
                return target

            song = encode("原曲.WV", "-c:a", "wavpack")
            decoded = cover_pipeline._separator_input(cover_pipeline._audio(song), run)
            self.assertEqual((sf.info(decoded).channels, sf.info(decoded).samplerate), (2, 44100))
            # Both roles retain their original mono waveform at its native sample rate.
            stem = encode("女声.mka", "-c:a", "flac")
            result = duet_mix.prepare_uploaded({"male": song, "female": stem}, source, run, 0)
            expected = sf.read(source, dtype="float32")[0]
            for role in ("male", "female"):
                samples, sr = sf.read(result[f"duet_{role}"], dtype="float32")
                self.assertEqual(sr, 48000)
                np.testing.assert_array_equal(samples, np.column_stack((expected, expected)))

            # FFmpeg must preserve the channel count so a surround stem gets rejected.
            invalid = encode("surround.mka", "-ac", "6", "-c:a", "flac")
            old = (run / "duet_female.wav").read_bytes()
            with self.assertRaisesRegex(ValueError, "单声道或双声道"):
                duet_mix.prepare_uploaded({"male": song, "female": invalid}, source, run, 0)
            self.assertEqual((run / "duet_female.wav").read_bytes(), old)
            bad = run / "corrupt.webm"
            bad.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "无法解码"):
                duet_mix.prepare_uploaded({"male": bad, "female": stem}, source, run, 0)
            self.assertFalse(list(run.glob("duet-input-*")))
