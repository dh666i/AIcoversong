"""Prepare aligned duet stems and mix independently converted singers."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

if __package__:
    from .audio_input import decode_audio
    from .vocal_mix import MIX_SR, add_space, fit_length, limit, mono, read_audio, stereo, treat_vocal
else:
    from audio_input import decode_audio
    from vocal_mix import MIX_SR, add_space, fit_length, limit, mono, read_audio, stereo, treat_vocal

ROLES = ("male", "female")
ROLE_NAMES = {"male": "男声", "female": "女声"}


def selected_roles(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return ROLES
    if mode in ROLES:
        return (mode,)
    raise ValueError("请选择要翻唱的对唱声部。")


def automatically_separated(job: dict, separator) -> bool:
    if job.get("duet_sources"):
        return False
    if isinstance(separator, dict):
        separator = separator.get("method")
    if separator:
        return separator in ("auto", "automatic", "BS RoFormer Male/Female (aufr33)")
    # The original job schema writes an empty map only for automatic duets.
    return job.get("duet_sources") == {}


def prepare_uploaded(stems: dict, vocals: str, run: Path, gain_db: float) -> dict:
    """Use source stems as authoritative voices; the separated instrumental is the bed."""
    reference = sf.info(vocals)
    gain = 10 ** (gain_db / 20)
    result = {}
    with TemporaryDirectory(dir=run, prefix="duet-input-") as scratch:
        working = Path(scratch)
        for role in ROLES:
            source = Path(stems[role])
            info = None
            if source.suffix.lower() in {".wav", ".wave", ".flac"}:
                try:
                    info = sf.info(source)
                except sf.LibsndfileError:
                    pass  # Some WAV codecs require FFmpeg.
            if info is None:
                decoded = working / f"decoded_{role}.wav"
                decode_audio(source, decoded)
                source, info = decoded, sf.info(decoded)
            if info.channels not in (1, 2):
                raise ValueError(f"请上传单声道或双声道的{ROLE_NAMES[role]}分轨。")
            if abs(info.duration - reference.duration) > 0.1:
                raise ValueError(f"{ROLE_NAMES[role]}分轨与原曲时长不一致；请从同一起点导出整首歌，保留前后静音。")
            audio = read_audio(source, sr=reference.samplerate)
            if not np.isfinite(audio).all() or not np.any(np.abs(audio) > 1e-5):
                raise ValueError(f"{ROLE_NAMES[role]}分轨为空或包含无效采样，请检查上传文件。")
            filename = f"duet_{role}.wav"
            sf.write(working / filename, stereo(fit_length(audio, reference.frames)) * gain,
                     reference.samplerate, subtype="FLOAT")
            result[f"duet_{role}"] = str(run / filename)
        for target in result.values():
            (working / Path(target).name).replace(target)
    result["duet_separator"] = {"method": "uploaded", "stem_gain_db": gain_db}
    return result


def write_preserved(stems: dict, mode: str, run: Path) -> str:
    """Sum unselected and unassigned audio exactly once, retaining stereo and phase."""
    from contextlib import ExitStack

    roles = selected_roles(mode)
    keep = [stems[f"duet_{role}"] for role in ROLES if role not in roles]
    if stems.get("duet_residual"):
        keep.append(stems["duet_residual"])
    info = sf.info(stems["duet_male"])
    path = run / "duet_preserved.wav"
    with ExitStack() as stack:
        inputs = [stack.enter_context(sf.SoundFile(p)) for p in keep]
        if any((source.frames, source.samplerate, source.channels) !=
               (info.frames, info.samplerate, info.channels) for source in inputs):
            raise ValueError("对唱分轨采样格式或时长不一致。")
        with sf.SoundFile(path, "w", samplerate=info.samplerate, channels=info.channels, subtype="FLOAT") as target:
            for start in range(0, info.frames, 65536):
                n = min(65536, info.frames - start)
                block = np.zeros((n, info.channels), dtype=np.float32)
                for source in inputs:
                    block += source.read(n, dtype="float32", always_2d=True)
                target.write(block)
    return str(path)


def _attenuate_quiet_leakage(voice: np.ndarray, reference: np.ndarray, sr: int) -> dict:
    """Suppress RVC tones amplified out of quiet separation residue, before reverb."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    length = len(reference)
    hop = max(1, sr // 100)
    frames = (length + hop - 1) // hop
    peaks = np.empty(frames, dtype=np.float32)
    levels = np.empty((2, frames), dtype=np.float32)
    for start in range(0, frames, 256):
        end = min(start + 256, frames)
        region = slice(start * hop, end * hop)
        for row, audio in enumerate((reference, voice)):
            block = audio[region]
            missing = (end - start) * hop - len(block)
            if missing:
                block = np.pad(block, ((0, missing), (0, 0)), mode="edge")
            block = block.reshape(end - start, hop, -1)
            power = np.square(block).mean(axis=(1, 2))
            levels[row, start:end] = 10 * np.log10(np.maximum(power, 1e-14))
            if row == 0:
                peaks[start:end] = np.max(np.abs(block), axis=(1, 2))

    ref_db, out_db = levels
    # Measure channel power before summing, preserving anti-phase stereo cues.
    # Hold around nearby singing so close consonants and attacks remain intact.
    active_level = float(np.percentile(ref_db, 98))
    active = ref_db >= active_level - 12
    offset = np.clip(np.median(out_db[active] - ref_db[active]), -3, 3)
    protected = maximum_filter1d(ref_db, size=9)
    quiet_weight = np.clip((active_level - 29 - protected) / 12, 0, 1)
    excess = np.clip(out_db - ref_db - offset - 4, 0, 24)
    reduction = gaussian_filter1d(quiet_weight * excess, sigma=.9) * (quiet_weight > 0)
    gain = np.power(10., -reduction / 20.) * np.minimum(peaks / 1e-4, 1)
    centers = (np.arange(frames) + .5) * hop
    for start in range(0, length, 65536):
        end = min(start + 65536, length)
        voice[start:end] *= np.interp(np.arange(start, end), centers, gain)[:, None]
    return {"attenuated_seconds": round(float((reduction >= 3).sum() * hop / sr), 2),
            "peak_reduction_db": round(float(reduction.max()), 2)}


def _shared_voice_gain(selected: np.ndarray, kept: np.ndarray, sr: int) -> np.ndarray:
    """Positive, sustained zero-lag copies in two bands, dominated by selected."""
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d

    hop = max(1, round(.02 * sr))
    radius = max(1, round(.055 * sr))
    centers = np.arange(0, len(selected), hop)
    weights = np.zeros(len(centers))
    # Nearby singers can sing the same pitch. Require the overtone waveform to
    # match independently as well; strong fundamental correlation is insufficient.
    sos = signal.butter(2, [700, min(4800, sr * .45)], btype="bandpass", fs=sr, output="sos")
    upper_x = signal.sosfilt(sos, selected, axis=0)
    upper_y = signal.sosfilt(sos, kept, axis=0)
    for i, center in enumerate(centers):
        sl = slice(max(0, center-radius), min(len(selected), center+radius))
        x, y = selected[sl].reshape(-1), kept[sl].reshape(-1)
        xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
        if xx < x.size * 10**(-43/10) or not (.0004 * xx < yy < .9 * xx):
            continue
        h, k = upper_x[sl].reshape(-1), upper_y[sl].reshape(-1)
        hh, kk = float(h @ h), float(k @ k)
        if hh < xx * .001 or kk < yy * .001:
            continue
        corr = xy / np.sqrt(xx * yy)
        corr_upper = float(h @ k) / np.sqrt(hh * kk)
        confidence = np.clip((corr - .93) / .05, 0, 1) * np.clip((corr_upper - .79) / .16, 0, 1)
        weights[i] = .85 * max(xy / xx, 0) * confidence
    # Reject isolated high-correlation coincidences, then ramp in and out.
    envelope = gaussian_filter1d(minimum_filter1d(weights, size=3), .85)
    return np.interp(np.arange(len(selected)), centers, envelope)


def _attenuate_partner_leakage(voice: np.ndarray, reference: np.ndarray, partner: np.ndarray, sr: int) -> dict:
    """Limit excess RVC level when a weak input duplicates the dominant singer."""
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

    hop = max(1, round(.02 * sr))
    count = (len(voice) + hop - 1) // hop
    # 100 ms of both waveforms: a shared note is insufficient evidence. Also
    # require the two stems' upper harmonics to agree at the same phase.
    sos = signal.butter(2, [700, min(4500, sr * .45)], btype="bandpass", fs=sr, output="sos").astype(np.float32)
    state = np.zeros((len(sos), 2, 4), dtype=np.float32)
    moments = np.empty((7, count), dtype=np.float64)
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        part = slice(start * hop, stop * hop)
        pair = np.column_stack((stereo(reference[part]), stereo(partner[part])))
        upper, state = signal.sosfilt(sos, pair, axis=0, zi=state)
        output = voice[part]
        padding = (stop - start) * hop - len(output)
        if padding:
            pair, upper, output = (np.pad(x, ((0, padding), (0, 0))) for x in (pair, upper, output))

        def average(x):
            return x.reshape(stop - start, -1).mean(axis=1, dtype=np.float64)

        for i, data in ((0, pair), (3, upper)):
            x, y = data[:, :2], data[:, 2:]
            moments[i:i + 3, start:stop] = (average(x * x), average(y * y), average(x * y))
        moments[6, start:stop] = average(output * output)
    xx, yy, xy, hh, kk, hk, vv = uniform_filter1d(moments, 5, axis=1, mode="nearest")
    ref_db, partner_db, out_db = 10 * np.log10(np.maximum([xx, yy, vv], 1e-14))
    usual = float(np.percentile(ref_db, 98))
    offset = np.clip(np.median((out_db - ref_db)[ref_db >= usual - 12]), -3, 3)
    corr = xy / np.sqrt(np.maximum(xx * yy, 1e-25))
    upper_corr = hk / np.sqrt(np.maximum(hh * kk, 1e-25))
    confidence = np.clip(np.minimum((corr - .42) / .22, (upper_corr - .38) / .22), 0, 1)
    # Nearly unison singers pass through short intervals of matching phase.
    # Check half a second too: copied overtones stay aligned as pitch moves.
    slow_x, slow_y, slow_xy, slow_h, slow_k, slow_hk = uniform_filter1d(
        moments[:6], 25, axis=1, mode="constant")
    slow_corr = slow_xy / np.sqrt(np.maximum(slow_x * slow_y, 1e-25))
    slow_upper = slow_hk / np.sqrt(np.maximum(slow_h * slow_k, 1e-25))
    confidence *= np.clip(np.minimum((slow_corr - .38) / .20, (slow_upper - .35) / .20), 0, 1)
    confidence *= (hh > .001 * xx) & (kk > .001 * yy)
    dominance = np.clip((partner_db - ref_db - 9) / 6, 0, 1)
    quiet = np.clip((usual - 12 - ref_db) / 7, 0, 1)
    excess = np.clip(out_db - ref_db - offset - 2, 0, 15)
    reduction = gaussian_filter1d(median_filter(excess * confidence * dominance * quiet, 3), .8)
    centers = (np.arange(count) + .5) * hop
    gain = 10 ** (-reduction / 20)
    for start in range(0, len(voice), 65536):
        stop = min(start + 65536, len(voice))
        voice[start:stop] *= np.interp(np.arange(start, stop), centers, gain)[:, None]
    return {"attenuated_seconds": round(float((reduction >= 3).sum() * hop / sr), 2),
            "peak_reduction_db": round(float(reduction.max()), 2)}


def repair_solo_bleed(selected_path, preserved_path, output_dir, *, solo_spans) -> dict:
    """Reassign only coherent leakage inside explicitly known solo spans (seconds).

    Returns independent float WAVs whose sum matches the two inputs. Keep this
    off for unknown sections: identical waveforms cannot establish singer identity.
    """
    with sf.SoundFile(selected_path) as lead, sf.SoundFile(preserved_path) as other:
        sr = lead.samplerate
        if (lead.frames, sr, lead.channels) != (other.frames, other.samplerate, other.channels):
            raise ValueError("串声修复需要从同一起点导出的同格式、等长分轨。")
        if sr < 8000 or lead.channels not in (1, 2):
            raise ValueError("串声修复支持 8 kHz 及以上的单声道和立体声分轨。")
        spans = sorted((round(a*sr), round(b*sr)) for a, b in solo_spans)
        end = 0
        for a, b in spans:
            if not end <= a < b <= lead.frames:
                raise ValueError("独唱时间段必须落在音频内且互不重叠。")
            end = b
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        selected, preserved = dest / "solo_selected.wav", dest / "solo_preserved.wav"
        if selected.exists() or preserved.exists():
            raise FileExistsError("输出分轨已存在。")
        count = 0
        config = dict(samplerate=sr, channels=lead.channels, subtype="FLOAT")
        with sf.SoundFile(selected, "x", **config) as out_x, sf.SoundFile(preserved, "x", **config) as out_y:
            def copy_until(stop):
                while lead.tell() < stop:
                    size = min(65536, stop - lead.tell())
                    out_x.write(lead.read(size, dtype="float32"))
                    out_y.write(other.read(size, dtype="float32"))

            hop = max(1, round(sr * .02))
            for a, b in spans:
                copy_until(a)
                for start in range(a, b, 400 * hop):
                    stop = min(b, start + 400 * hop)
                    lo, hi = max(a, start - 12 * hop), min(b, stop + 12 * hop)
                    lead.seek(lo)
                    other.seek(lo)
                    x = lead.read(hi - lo, dtype="float32", always_2d=True)
                    y = other.read(hi - lo, dtype="float32", always_2d=True)
                    region = slice(start - lo, stop - lo)
                    gain = _shared_voice_gain(x, y, sr)[region]
                    idx = np.arange(start, stop)
                    gain *= np.minimum(1, np.minimum(idx - a, b - 1 - idx) / (.05 * sr))
                    count += int((gain > .025).sum())
                    shared = (x[region] * gain[:, None]).astype("float32")
                    out_x.write(x[region] + shared)
                    out_y.write(y[region] - shared)
                lead.seek(b)
                other.seek(b)
            copy_until(lead.frames)
    return {"selected": str(selected), "preserved": str(preserved),
            "report": {"affected_seconds": round(count/sr, 3),
                       "solo_spans": [[a/sr, b/sr] for a, b in spans]}}


def mix_duet(files: dict, mode: str, output: Path, *, vocal_gain_db: float, music_gain_db: float,
             clarity: float, dynamics: float, reverb: float, auto_separated: bool = False) -> dict:
    music = stereo(read_audio(files["instrumental"]))
    length = len(music)
    voices = np.zeros((length, 2), dtype=np.float32)
    reports = {}
    for role in selected_roles(mode):
        ref = fit_length(read_audio(files[f"duet_{role}"]), length)
        converted = fit_length(read_audio(files[f"duet_{role}_converted"]), length)
        voice, report = treat_vocal(converted, ref, clarity=clarity, dynamics=dynamics, reverb=0)
        report["quiet_leakage"] = _attenuate_quiet_leakage(voice, ref, MIX_SR)
        if reverb > 0:
            send = -25 + 20 * math.log10(reverb)
            voice += add_space(mono(voice), MIX_SR, wet_ratio_db=send, rt60=0.5)
            report["space"] = {"send_db": send, "rt60": 0.5}
        # Older single-role remixes retain the complete original partner and
        # residual together. In both-role mode compare the two input singers.
        if auto_separated:
            partner_key = "duet_preserved"
            if mode == "both":
                partner_key = "duet_female" if role == "male" else "duet_male"
            partner = fit_length(read_audio(files[partner_key]), length)
            report["partner_leakage"] = _attenuate_partner_leakage(voice, ref, partner, MIX_SR)
            del partner
        voices += voice
        reports[role] = report
        del ref, converted, voice
    # The vocal gain applies to the converted performance; original voices
    # bypass pitch, treatment, reverb and this gain control.
    voices *= 10 ** (vocal_gain_db / 20)
    solo, _ = limit(voices, MIX_SR, ceiling=0.98)
    sf.write(output / "my_vocals.wav", solo, MIX_SR, subtype="PCM_24")
    del solo
    voices += fit_length(stereo(read_audio(files["duet_preserved"])), length)
    voices += music * 10 ** (music_gain_db / 20)
    del music
    if not np.isfinite(voices).all():
        raise ValueError("对唱混音包含无效采样。")
    peak = float(np.percentile(np.abs(voices).max(axis=1), 99.99))
    headroom = min(1, 0.95 / peak) if peak else 1
    mixed, reduction = limit(voices * headroom, MIX_SR)
    sf.write(output / "cover.wav", mixed, MIX_SR, subtype="PCM_24")
    return {"duet_mode": mode, "roles": reports, "mix_gain_db": 20 * math.log10(headroom),
            "limiter_reduction_db": reduction}


def remix_inputs(mode: str) -> dict:
    inputs = {"instrumental": "instrumental.wav", "duet_preserved": "duet_preserved.wav"}
    for role in ROLES:
        inputs[f"duet_{role}"] = f"duet_{role}.wav"
    for role in selected_roles(mode):
        inputs[f"duet_{role}_converted"] = f"duet_{role}_converted.wav"
    return inputs


def retain_stems(files: dict, output: Path) -> dict:
    retained = {}
    for key, source in files.items():
        target = output / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        retained[key] = str(target)
    return retained
