"""Isolated duet routing and mix checks. No model downloads or real recordings."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import cover_library
import cover_pipeline as pipeline
import duet_mix


class DuetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / "run_20261120_180000_1111"
        self.run.mkdir()
        self.sr = 44100
        t = np.arange(self.sr * 2) / self.sr
        x = 0.1 * np.sin(2 * np.pi * 140 * t) * (t < 1.6)
        y = 0.09 * np.sin(2 * np.pi * 269 * t) * (t > 0.6)
        self.voices = {"male": np.column_stack((x, x * .7)), "female": np.column_stack((y * .5, y))}
        self.residual = np.column_stack((x, -x)) * .007
        self.stems = {}
        for role, audio in {**self.voices, "residual": self.residual}.items():
            path = self.run / f"duet_{role}.wav"
            sf.write(path, audio, self.sr, subtype="FLOAT")
            self.stems[f"duet_{role}"] = str(path)
        self.music = self.run / "instrumental.wav"
        sf.write(self.music, np.zeros_like(self.residual), self.sr, subtype="FLOAT")

    def test_unselected_voice_and_unassigned_residue_are_preserved_in_chorus_and_solos(self):
        for mode in ("male", "female", "both"):
            p = duet_mix.write_preserved(self.stems, mode, self.run)
            wanted = self.residual.copy()
            for role in duet_mix.ROLES:
                if role not in duet_mix.selected_roles(mode):
                    wanted += self.voices[role]
            np.testing.assert_allclose(sf.read(p)[0], wanted, atol=1e-8)

    def test_uploaded_stems_are_aligned_and_have_explicit_roles(self):
        stems = {k: self.stems[f"duet_{k}"] for k in duet_mix.ROLES}
        out = self.root / "upload"
        out.mkdir()
        results = duet_mix.prepare_uploaded(stems, self.music, out, -6.020599913)
        for role in stems:
            np.testing.assert_allclose(sf.read(results[f"duet_{role}"])[0], .5 * self.voices[role], atol=1e-8)
        sf.write(self.run / "short.wav", np.zeros(5000), self.sr)
        with self.assertRaisesRegex(ValueError, "时长"):
            duet_mix.prepare_uploaded(dict(stems, female=self.run / "short.wav"), self.music, out, 0)

    def test_independent_model_conversion_once_per_selected_role(self):
        calls = []
        def load_engine(model, index):
            class Engine:
                def convert(self, source, output, **options):
                    calls.append((model.name, Path(source).name, Path(output).name, options["pitch"]))
                    shutil.copyfile(source, output)
                    return {"seconds": 2}
            return Engine()

        for mode in ("male", "female", "both"):
            calls.clear()
            job = dict(duet_mode=mode, model="alpha.pth", index="", index_rate=0,
                       duet_second_model="beta.pth", duet_second_index="", pitch=0, voice_range="keep")
            with patch.object(pipeline, "_load_engine", side_effect=load_engine):
                result = pipeline._convert_duet(job, self.run, self.stems)
            for role, call in zip(duet_mix.selected_roles(mode), calls, strict=True):
                self.assertEqual(call, ("beta.pth" if role == "female" and mode == "both" else "alpha.pth",
                                        f"duet_{role}.wav", f"duet_{role}_converted.wav", 0))
                self.assertTrue(Path(result[f"duet_{role}_converted"]).is_file())

    def test_optional_dereverb_only_changes_selected_rvc_input(self):
        import types
        (self.run / "separate_result.json").write_text(json.dumps({"vocals": str(self.music)}))
        job = dict(duet_mode="male", duet_dereverb=True, duet_sources=self.stems, index="",
                   model="self.pth", index_rate=.1, pitch=0, voice_range="keep")
        def dry_and_room(source, model, names, folder, *args, **kwargs):
            files = {}
            for key in names.values():
                path = folder / (key + ".wav")
                shutil.copyfile(source, path)
                files[key] = str(path)
            return files, 1.0

        def convert(source, dest, **kwargs):
            self.assertEqual(Path(source).name, "duet_male_dry.wav")
            shutil.copyfile(source, dest)
            return {}

        with patch.object(duet_mix, "prepare_uploaded", return_value=self.stems.copy()), \
             patch.object(pipeline, "_two_stems", side_effect=dry_and_room) as separator:
            stems = pipeline._duet(job, self.run)
        separator.assert_called_once()
        np.testing.assert_allclose(sf.read(stems["duet_preserved"])[0],
                                   self.voices["female"] + self.residual, atol=1e-8)
        with patch.object(pipeline, "_load_engine", return_value=types.SimpleNamespace(convert=convert)):
            pipeline._convert_duet(job, self.run, stems)

    def test_remix_preserves_other_voice_after_parent_is_trashed(self):
        duet_mix.write_preserved(self.stems, "male", self.run)
        sf.write(self.run / "duet_male_converted.wav", self.voices["male"] * .8, self.sr, subtype="FLOAT")
        sf.write(self.run / "cover.wav", self.voices["male"] + self.voices["female"], self.sr)
        (self.run / "job.json").write_text(json.dumps({"duet_mode":"male","original_song":"test.wav"}))
        with self.assertRaisesRegex(ValueError, "和声"):
            cover_library.remix_work(self.run.name, output_root=self.root, backing_mode="remove")
        first = cover_library.remix_work(self.run.name, output_root=self.root, clarity=0, reverb=0)
        cover_library.delete_work(self.run.name, self.root)
        second = cover_library.remix_work(first["run_id"], output_root=self.root, clarity=0, reverb=0)
        for remix in (first, second):
            out = Path(remix["path"])
            audio = sf.read(out / "cover.wav")[0]
            converted_only = sf.read(out / "my_vocals.wav")[0]
            np.testing.assert_allclose(audio - converted_only, self.voices["female"] + self.residual, atol=3e-7)
            self.assertLess(np.max(np.abs(converted_only[int(1.7*self.sr):])), 1e-6)

    def test_invalid_mode_and_half_uploaded_pair_rejected_before_creating_a_run(self):
        song = self.stems["duet_male"]
        model = self.root / "voice.pth"
        model.touch()
        with patch.object(pipeline, "MODEL_ROOT", self.root), patch.object(pipeline, "_new_run") as new_run:
            for args in ({"duet_mode":"unknown"},
                         {"duet_mode":"both", "duet_male_stem":song},
                         {"duet_mode":"male", "duet_male_stem":song, "duet_female_stem":song}):
                with self.assertRaises(ValueError):
                    next(pipeline.run_cover(song, "voice.pth", **args))
            new_run.assert_not_called()
