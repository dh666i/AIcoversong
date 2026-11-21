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


def split_duet(vocals_path: str | Path, run_dir: str | Path, *, swap: bool = False) -> dict:
    """Run inside separator-env on the complete vocals, before any karaoke split."""
    if not duet_available():
        raise FileNotFoundError("缺少男女人声分离模型。请在项目根目录运行 "
                                r".\env\Scripts\python.exe tools/download_assets.py --duet，"
                                "或提供两条已对齐的人声分轨。")

    import soundfile as sf
    from cover_pipeline import _two_stems

    source, run = Path(vocals_path).resolve(), Path(run_dir).resolve()
    info = sf.info(source)
    if info.channels not in (1, 2) or not info.frames:
        raise ValueError("对唱分离需要非空的单声道或立体声人声。")
    with TemporaryDirectory(prefix="duet-work-", dir=run) as working:
        scratch = Path(working)
        estimates, _ = _two_stems(source, MODEL, {"male": "male-estimate", "female": "female-estimate"},
                                  scratch, "duet", overlap=4, autocast=False)
        male, female = Path(estimates["male-estimate"]), Path(estimates["female-estimate"])
        if swap:
            male, female = female, male
        result = _write_outputs(source, male, female, scratch, run)
    return {**result, "duet_separator": "BS RoFormer Male/Female (aufr33)", "duet_swapped": bool(swap)}
