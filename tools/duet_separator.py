"""Two simultaneous vocal estimates and an additive residual for duet covers."""
from __future__ import annotations

from contextlib import ExitStack
from math import log10
from pathlib import Path
from tempfile import TemporaryDirectory

MODEL_DIR = Path(__file__).resolve().parents[1] / "Applio" / "rvc" / "models" / "uvr"
MODEL = "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt"
CONFIG = "config_chorus_male_female_bs_roformer.yaml"
BLOCK_FRAMES = 65536
SEPARATION_PASSES = 4


def duet_available() -> bool:
    """Cheap UI check. The asset installer verifies both SHA-256 signatures."""
    try:
        return ((MODEL_DIR / MODEL).stat().st_size == 527119779 and
                (MODEL_DIR / CONFIG).stat().st_size == 1847)
    except OSError:
        return False


def _match_grid(estimate: Path, reference, scratch: Path) -> Path:
    """Normalize the estimator's 44.1 kHz grid to the input rate if necessary."""
    import soundfile as sf

    info = sf.info(estimate)
    if info.channels not in (1, 2):
        raise ValueError("对唱模型输出的声道数无效。")
    if abs(info.duration - reference.frames / reference.samplerate) > 0.02:
        raise ValueError("对唱模型输出长度与输入不匹配。")
    if info.samplerate == reference.samplerate:
        return estimate

    from math import gcd
    from scipy.signal import resample_poly

    samples, rate = sf.read(estimate, dtype="float32", always_2d=True)
    factor = gcd(rate, reference.samplerate)
    aligned = resample_poly(samples, reference.samplerate // factor, rate // factor, axis=0)
    target = scratch / (estimate.stem + "-aligned.wav")
    sf.write(target, aligned, reference.samplerate, subtype="FLOAT")
    return target


def _write_outputs(source: Path, male: Path, female: Path, scratch: Path, run: Path) -> dict:
    """Preserve the original channels, alignment, level and unassigned signal."""
    import numpy as np
    import soundfile as sf

    names = ("duet_male", "duet_female", "duet_residual")
    energy = [0.0, 0.0]
    with ExitStack() as stack:
        original = stack.enter_context(sf.SoundFile(source))
        streams = [stack.enter_context(sf.SoundFile(_match_grid(path, original, scratch)))
                   for path in (male, female)]
        writers = [stack.enter_context(sf.SoundFile(scratch / (name + ".wav"), mode="w",
                                                    channels=original.channels, samplerate=original.samplerate,
                                                    subtype="FLOAT")) for name in names]
        for mixture in original.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
            residual = mixture.copy()
            for index, (stream, writer) in enumerate(zip(streams, writers[:2])):
                samples = stream.read(len(mixture), dtype="float32", always_2d=True, fill_value=0)
                if original.channels == 1 and stream.channels == 2:
                    samples = samples.mean(axis=1, keepdims=True)
                elif original.channels == 2 and stream.channels == 1:
                    samples = np.repeat(samples, 2, axis=1)
                if not np.isfinite(samples).all():
                    raise ValueError("对唱模型返回了无效的音频数据。")
                writer.write(samples)
                energy[index] += float(np.sum(np.square(samples), dtype=np.float64))
                residual -= samples
            writers[2].write(residual)
    paths = {}
    for name in names:
        target = run / (name + ".wav")
        (scratch / (name + ".wav")).replace(target)
        paths[name] = str(target)
    if max(energy) == 0:
        paths["duet_balance_note"] = "两个估计声部均为静音，请核对原曲的人声或使用独立人声分轨。"
    elif min(energy) <= max(energy) * 0.01:
        role = "男声" if energy[0] < energy[1] else "女声"
        difference = (f"平均电平比另一轨低 {10 * log10(max(energy) / min(energy)):.0f} dB"
                      if min(energy) > 0 else "为静音")
        paths["duet_balance_note"] = f"估计的{role}分轨{difference}。可能是该声部出场较少或分轨不准，请试听核对。"
    return paths


def split_duet(vocals_path: str | Path, run_dir: str | Path, *, swap: bool = False,
               extra_passes: int = 0) -> dict:
    """Estimate both roles, then check the lead and backing layers independently."""
    if not duet_available():
        raise FileNotFoundError("缺少男女人声分离模型。请在项目根目录运行 "
                                r".\env\Scripts\python.exe tools/download_assets.py --duet，"
                                "或提供两条已对齐的人声分轨。")

    import soundfile as sf
    if __package__:
        from .cover_pipeline import LEAD_CONFIG, LEAD_MODEL, _two_stems
        from .duet_harmony import reconcile_harmony
    else:
        from cover_pipeline import LEAD_CONFIG, LEAD_MODEL, _two_stems
        from duet_harmony import reconcile_harmony

    if not all((MODEL_DIR / name).is_file() for name in (LEAD_MODEL, LEAD_CONFIG)):
        raise FileNotFoundError("合唱复核需要基础模型中的主唱与和声分离模型，"
                                "请运行 tools/download_assets.py 安装基础模型，或上传已对齐的男女声分轨。")

    source, run = Path(vocals_path).resolve(), Path(run_dir).resolve()
    info = sf.info(source)
    if info.channels not in (1, 2) or not info.frames:
        raise ValueError("对唱分离需要非空的单声道或立体声人声。")
    with TemporaryDirectory(prefix="duet-work-", dir=run) as working:
        scratch = Path(working)
        total = SEPARATION_PASSES + extra_passes

        def role_estimates(audio: Path, layer: str, pass_number: int) -> dict:
            print(f"duet-separate-pass {pass_number}/{total}", flush=True)
            names = {role: f"{layer}-{role}" for role in ("male", "female")}
            paths, _ = _two_stems(audio, MODEL, names, scratch, layer, overlap=4, autocast=False)
            return {role: Path(paths[name]) for role, name in names.items()}

        baseline = role_estimates(source, "whole", 1)
        print(f"duet-separate-pass 2/{total}", flush=True)
        layers, _ = _two_stems(source, LEAD_MODEL, {"Vocals": "lead", "Instrumental": "backing"},
                               scratch, "layers", overlap=4, autocast=True)
        lead = role_estimates(Path(layers["lead"]), "lead", 3)
        backing = role_estimates(Path(layers["backing"]), "backing", 4)
        male, female = scratch / "routed-male.wav", scratch / "routed-female.wav"
        report = reconcile_harmony(baseline["male"], baseline["female"], lead, backing, male, female)
        if swap:
            male, female = female, male
        result = _write_outputs(source, male, female, scratch, run)
    return {**result, "duet_separator": "BS RoFormer Male/Female (aufr33)", "duet_swapped": bool(swap),
            "duet_harmony_review": report}
