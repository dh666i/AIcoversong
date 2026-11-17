"""Mix-ready treatment of a converted vocal and the final cover mix.

Pure numpy / scipy / soundfile: importing or running this loads no model.

A voice model reproduces the tone of the recordings it was trained on. Raw
home recordings are darker than a produced vocal and their consonants are
softer, so an untreated conversion sits muffled inside a commercial backing
track. The steps below use the original lead vocal of the same song as the
reference, because the backing track was mixed around exactly that vocal:

1. tone       - broadband equaliser towards the long-term spectrum of the
                reference (third-octave smoothed: brightness and body, not
                formants)
2. dynamics   - follow the level contour of the reference (restores weak word
                onsets; bounded gain, never a free boost)
3. level      - same level as the original dry lead, so the vocal/backing
                balance of the record is kept
4. space      - a synthetic reverb of the *converted* voice, as wet as the
                original production was
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MIX_SR = 44100


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def read_audio(path: str | Path, sr: int = MIX_SR) -> np.ndarray:
    """Return float32 audio shaped (samples, channels) at ``sr``."""
    import soundfile as sf
    import soxr

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != sr:
        audio = soxr.resample(audio, rate, sr, quality="VHQ").astype(np.float32)
    return audio


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    pad = [(0, length - len(audio))] + [(0, 0)] * (audio.ndim - 1)
    return np.pad(audio, pad)


def envelope_db(signal_: np.ndarray, sr: int, window: float = 0.03, hop: float = 0.005) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    size = max(1, int(sr * window))
    power = uniform_filter1d(signal_.astype(np.float64) ** 2, size=size, mode="constant")
    return 10 * np.log10(np.maximum(power[::max(1, int(sr * hop))], 1e-14))


def active_rms(signal_: np.ndarray, reference_db: np.ndarray, sr: int, hop: float = 0.005) -> float:
    """RMS over the frames where the reference vocal is actually singing."""
    step = max(1, int(sr * hop))
    active = reference_db > np.percentile(reference_db, 95) - 35
    mask = np.repeat(active, step)[:len(signal_)]
    mask = np.pad(mask, (0, len(signal_) - len(mask)))
    chosen = signal_[mask]
    return float(np.sqrt(np.mean(chosen.astype(np.float64) ** 2))) if len(chosen) else 0.0


# --------------------------------------------------------------------------
# 1. tone
# --------------------------------------------------------------------------

def _band_spectrum_db(signal_: np.ndarray, sr: int, grid: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Power density per log-frequency cell (dB) over active frames; returns (spectrum, frame mask)."""
    from scipy import signal

    freqs, _, spec = signal.stft(signal_.astype(np.float64), fs=sr, nperseg=4096, noverlap=2048,
                                 window="hann", boundary=None, padded=False)
    power = np.abs(spec) ** 2
    if keep is None:
        energy = power.sum(0)
        keep = energy > np.percentile(energy, 95) * 1e-4
    count = min(len(keep), power.shape[1])
    if not keep[:count].any():
        return np.full(len(grid), -200.0), keep
    mean = power[:, :count][:, keep[:count]].mean(1)
    cumulative = np.concatenate([[0.0], np.cumsum(mean)])
    edges = np.concatenate([[0.0], (freqs[1:] + freqs[:-1]) / 2, [freqs[-1]]])
    half = 2 ** (1 / 48)
    low, high = grid / half, np.minimum(grid * half, freqs[-1])
    band = np.interp(high, edges, cumulative) - np.interp(low, edges, cumulative)
    return 10 * np.log10(np.maximum(band / np.maximum(high - low, 1e-9), 1e-20)), keep


def match_tone(converted: np.ndarray, reference: np.ndarray, sr: int, *, strength: float = 0.75) -> tuple[np.ndarray, dict]:
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d

    grid = 20.0 * 2 ** (np.arange(0, int(24 * np.log2(sr / 2 / 20.0))) / 24)  # 1/24 octave cells
    ref_db, keep = _band_spectrum_db(reference, sr, grid)
    if keep.sum() < 8 or not np.any(reference):
        return converted, {"strength": strength, "skipped": "reference is silent"}
    conv_db, _ = _band_spectrum_db(converted, sr, grid, keep)
    sigma = (1 / 3) * 24 / 2.355  # third-octave FWHM
    difference = gaussian_filter1d(ref_db, sigma) - gaussian_filter1d(conv_db, sigma)
    body = (grid >= 150) & (grid <= 1200)
    difference -= difference[body].mean()
    ceiling = np.interp(grid, [20, 100, 1000, 8000, 12000, 16000, sr / 2], [0, 6, 12, 12, 9, 6, 0])
    curve = strength * np.clip(difference, -9.0, ceiling)
    # Rumble below the voice is never wanted, whatever the reference contains.
    curve += -10 * np.log10(1 + (65.0 / grid) ** 4)
    curve[grid > 18000] = np.minimum(curve[grid > 18000], 0.0)
    nyquist = sr / 2
    inside = grid < nyquist
    points = np.concatenate([[0.0], grid[inside], [nyquist]])
    gains = np.concatenate([[curve[0]], curve[inside], [min(curve[inside][-1], 0.0)]])
    fir = signal.firwin2(4095, points, np.power(10.0, gains / 20.0), fs=sr)
    result = signal.fftconvolve(converted.astype(np.float64), fir, mode="same").astype(np.float32)
    marks = [100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000]
    stats = {"strength": strength, "gain_db": {str(f): round(float(np.interp(f, grid, curve)), 1) for f in marks}}
    return result, stats


# --------------------------------------------------------------------------
# 2. dynamics
# --------------------------------------------------------------------------

def match_dynamics(converted: np.ndarray, reference: np.ndarray, sr: int, *, strength: float = 0.6,
                   max_boost_db: float = 8.0, max_cut_db: float = 6.0) -> tuple[np.ndarray, dict]:
    from scipy.ndimage import gaussian_filter1d, median_filter

    if strength <= 0:
        return converted, {"strength": 0.0}
    step = max(1, int(sr * 0.005))
    ref_db, conv_db = envelope_db(reference, sr), envelope_db(converted, sr)
    count = min(len(ref_db), len(conv_db))
    ref_db, conv_db = ref_db[:count], conv_db[:count]
    active = ref_db > np.percentile(ref_db, 95) - 50
    if active.sum() < 20:
        return converted, {"strength": strength, "skipped": "reference is silent"}
    body = active & (ref_db > np.percentile(ref_db[active], 50) - 6)
    offset = float(np.median((ref_db - conv_db)[body]))
    gain = np.clip(strength * (ref_db - conv_db - offset), -max_cut_db, max_boost_db)
    gain[~active] = 0.0
    # Never lift frames where the model produced next to nothing: that is noise, not a consonant.
    empty = conv_db < np.percentile(conv_db[active], 95) - 60
    gain[empty] = np.minimum(gain[empty], 0.0)
    gain = gaussian_filter1d(median_filter(gain, size=3), sigma=2.0)
    curve = np.interp(np.arange(len(converted)), np.arange(count) * step, gain)
    stats = {"strength": strength, "level_offset_db": offset,
             "boost_p95_db": float(np.percentile(gain[active], 95)), "cut_p05_db": float(np.percentile(gain[active], 5)),
             "mean_abs_db": float(np.mean(np.abs(gain[active])))}
    return (converted * np.power(10.0, curve / 20.0)).astype(np.float32), stats


# --------------------------------------------------------------------------
# 4. space
# --------------------------------------------------------------------------

def estimate_rt60(wet: np.ndarray, dry: np.ndarray, sr: int, default: float = 1.5) -> float:
    """Median decay of the original reverb in the pauses right after a phrase ends."""
    hop = 0.01
    wet_db, dry_db = envelope_db(wet, sr, 0.05, hop), envelope_db(dry, sr, 0.05, hop)
    count = min(len(wet_db), len(dry_db))
    wet_db, dry_db = wet_db[:count], dry_db[:count]
    quiet = dry_db < np.percentile(dry_db, 95) - 45
    padded = np.concatenate([[False], quiet, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    slopes = []
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < 45 or start < 20:
            continue
        segment = wet_db[start + 5:min(end, start + 85)]
        if segment[0] < np.percentile(wet_db, 95) - 30 or segment[0] - segment.min() < 12:
            continue
        stop = int(np.argmax(segment < segment[0] - 25)) or len(segment)
        if stop < 20:
            continue
        slope = np.polyfit(np.arange(stop) * hop, segment[:stop], 1)[0]
        if slope < -8:
            slopes.append(-60.0 / slope)
    if len(slopes) < 3:
        return default
    return float(np.clip(np.median(slopes), 0.8, 2.6))


def reverb_impulse(sr: int, rt60: float, seed: int) -> np.ndarray:
    from scipy import signal

    rng = np.random.default_rng(seed)
    length = int(sr * min(rt60 * 1.4, 3.6))
    time = np.arange(length) / sr
    impulse = np.zeros(length)
    for low, high, scale in ((180, 600, 1.15), (600, 2500, 1.0), (2500, 7000, 0.6)):
        sos = signal.butter(2, [low, high], btype="band", fs=sr, output="sos")
        impulse += signal.sosfilt(sos, rng.standard_normal(length)) * np.power(10.0, -3 * time / (rt60 * scale))
    fade = int(0.015 * sr)
    impulse[:fade] *= np.linspace(0, 1, fade) ** 2
    impulse = np.concatenate([np.zeros(int(0.03 * sr)), impulse])
    return impulse / np.sqrt(np.sum(impulse ** 2))


def add_space(lead: np.ndarray, sr: int, *, wet_ratio_db: float, rt60: float) -> np.ndarray:
    """Stereo reverb return for a mono lead, ``wet_ratio_db`` below the energy of the lead."""
    from scipy import signal

    wet = np.stack([signal.fftconvolve(lead.astype(np.float64), reverb_impulse(sr, rt60, seed), mode="full")[:len(lead)]
                    for seed in (11, 23)], axis=1)
    energy_dry = float(np.mean(lead.astype(np.float64) ** 2))
    energy_wet = float(np.mean(wet ** 2))
    if energy_wet <= 0 or energy_dry <= 0:
        return np.zeros((len(lead), 2), dtype=np.float32)
    return (wet * np.sqrt(energy_dry / energy_wet) * 10 ** (wet_ratio_db / 20)).astype(np.float32)


# --------------------------------------------------------------------------
# Vocal chain and final mix
# --------------------------------------------------------------------------

def treat_vocal(converted: np.ndarray, reference_dry: np.ndarray, sr: int = MIX_SR, *, clarity: float = 0.75,
                dynamics: float = 0.6, reference_wet: np.ndarray | None = None,
                reverb: float = 1.0) -> tuple[np.ndarray, dict]:
    """Return the treated stereo vocal (lead + reverb), level-matched to ``reference_dry``."""
    reference = mono(reference_dry)
    lead = fit_length(mono(converted), len(reference)).astype(np.float32)
    report: dict = {}
    if clarity > 0:
        lead, report["tone"] = match_tone(lead, reference, sr, strength=clarity)
    else:
        report["tone"] = {"strength": 0.0}
    lead, report["dynamics"] = match_dynamics(lead, reference, sr, strength=dynamics)
    ref_db = envelope_db(reference, sr)
    target, current = active_rms(reference, ref_db, sr), active_rms(lead, ref_db, sr)
    # More than +12 dB would mean the model produced almost nothing: do not turn that into noise.
    gain = float(np.clip(target / current, 10 ** (-24 / 20), 10 ** (12 / 20))) if current > 0 else 1.0
    lead = lead * gain
    report["level_gain_db"] = float(20 * np.log10(max(gain, 1e-9)))
    out = stereo(lead).copy()
    if reference_wet is not None and reverb > 0:
        wet = mono(fit_length(reference_wet, len(reference)))
        energy_dry = float(np.mean(reference.astype(np.float64) ** 2))
        energy_wet = float(np.mean(wet.astype(np.float64) ** 2))
        ratio_db = 10 * np.log10(max(energy_wet, 1e-14) / max(energy_dry, 1e-14))
        rt60 = estimate_rt60(wet, reference, sr)
        send_db = float(np.clip(ratio_db, -30.0, -9.0) + 20 * np.log10(reverb))
        out = out + add_space(lead, sr, wet_ratio_db=send_db, rt60=rt60)
        report["space"] = {"original_wet_db": float(ratio_db), "send_db": send_db, "rt60": rt60}
    else:
        report["space"] = {"send_db": None}
    return out.astype(np.float32), report


def solo_backing_mask(lead: np.ndarray, backing: np.ndarray, sr: int, *, frame: float = 0.05,
                      min_seconds: float = 0.5, phrase_gap: float = 0.3, min_alone: float = 0.7,
                      min_level_db: float = -12.0, pad: float = 0.1, ramp: float = 0.12):
    """Find whole phrases that the backing stem sings on its own.

    A lead/backing separation model calls everything that is not the main voice
    "backing": harmonies under the lead, but also lines sung while the lead rests
    (one test song ends with 19 s of them). Kept in the original voice, such a
    line exposes the original singer. The decision is made per phrase of the
    backing stem, never per moment: switching a voice between two singers inside
    one phrase is worse than either choice. A phrase is taken when at least
    ``min_alone`` of its energy is sung while the lead is 10 dB quieter and its
    level is within ``min_level_db`` of the usual singing level of the lead.

    Returns (weight per sample in 0..1, [(start, end) seconds], info). ``info``
    counts the seconds where the backing is exposed and how many remain.
    """
    from scipy.ndimage import uniform_filter1d

    hop = max(1, int(sr * frame))
    count = min(len(lead), len(backing)) // hop
    weight = np.zeros(len(backing), dtype=np.float32)
    info = {"exposed_seconds": 0.0, "replaced_seconds": 0.0, "left_exposed_seconds": 0.0}
    if count < 4:
        return weight, [], info

    def power(audio: np.ndarray) -> np.ndarray:
        return (audio[:count * hop].astype(np.float64) ** 2).reshape(count, hop, -1).mean(axis=(1, 2))

    lead_power, back_power = power(lead), power(backing)
    lead_db = 10 * np.log10(np.maximum(lead_power, 1e-14))
    back_db = 10 * np.log10(np.maximum(back_power, 1e-14))
    singing = lead_db > np.percentile(lead_db, 95) - 30
    if singing.sum() >= 20 and np.percentile(lead_db, 95) > -70:
        usual = float(np.median(lead_db[singing]))
    else:  # no lead at all: judge the backing stem against its own loud passages
        usual = float(np.percentile(back_db, 95)) - 6
    active = (back_db > usual - 30) & (back_db > -70)
    alone = lead_db < back_db - 10
    exposed = active & alone & (back_db > usual + min_level_db)
    padded = np.concatenate([[False], active, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    phrases: list[list[int]] = []
    for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
        if phrases and (start - phrases[-1][1]) * frame <= phrase_gap:
            phrases[-1][1] = end
        else:
            phrases.append([start, end])
    extra = int(round(pad / frame))
    chosen = np.zeros(count, dtype=bool)
    spans = []
    for start, end in phrases:
        energy = back_power[start:end]
        if (end - start) * frame < min_seconds or energy.sum() <= 0:
            continue
        if energy[alone[start:end]].sum() < min_alone * energy.sum():
            continue
        if 10 * np.log10(energy.mean()) < usual + min_level_db:
            continue
        start, end = max(0, start - extra), min(count, end + extra)
        chosen[start:end] = True
        spans.append((start * hop / sr, end * hop / sr))
    info = {"exposed_seconds": float(exposed.sum() * frame), "replaced_seconds": float(chosen.sum() * frame),
            "left_exposed_seconds": float((exposed & ~chosen).sum() * frame)}
    if not spans:
        return weight, [], info
    weight[:count * hop] = np.repeat(chosen.astype(np.float32), hop)
    weight = uniform_filter1d(weight, size=max(1, int(sr * ramp)), mode="nearest")
    return weight.astype(np.float32), spans, info


def solo_backing_spans(lead_path: str | Path, backing_path: str | Path) -> tuple[list[tuple[float, float]], dict]:
    """Phrases the backing stem sings alone (and the exposure counts), read from the two stem files."""
    lead, backing = read_audio(lead_path), read_audio(backing_path)
    length = min(len(lead), len(backing))
    _, spans, info = solo_backing_mask(lead[:length], backing[:length], MIX_SR)
    return spans, info


def limit(audio: np.ndarray, sr: int, ceiling: float = 0.95, lookahead: float = 0.005,
          release_db_per_s: float = 20.0) -> tuple[np.ndarray, float]:
    """Look-ahead peak limiter; returns the audio and the largest gain reduction in dB.

    The gain is fully down ``lookahead`` before a peak and recovers at a fixed
    rate afterwards. The slow recovery matters: a limiter that lets go within a
    few milliseconds changes its gain inside single cycles of a bass note, which
    is heard as distortion.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    peak = np.abs(audio).max(axis=1) if audio.ndim == 2 else np.abs(audio)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    if need.min() >= 1.0:
        return audio, 0.0
    block = max(1, sr // 1000)  # 1 ms control rate
    count = -(-len(need) // block)
    control = np.pad(need, (0, count * block - len(need)), constant_values=1.0).reshape(count, block).min(axis=1)
    ahead = max(2, int(round(lookahead * sr / block)))
    held_db = 20 * np.log10(minimum_filter1d(control, size=2 * ahead + 1, mode="nearest"))
    slope = release_db_per_s * block / sr * np.arange(count)
    released_db = np.minimum.accumulate(held_db - slope) + slope  # recover at a constant dB per second
    # Each averaged value comes from blocks whose hold window still covers the
    # peak, so the ramp reaches full reduction before the peak arrives.
    smooth_db = uniform_filter1d(released_db, size=ahead, mode="nearest")
    centres = (np.arange(count) + 0.5) * block
    gain = np.power(10.0, np.interp(np.arange(len(need)), centres, smooth_db) / 20.0)
    gain = np.minimum(gain, need)
    out = audio * (gain[:, None] if audio.ndim == 2 else gain)
    return out.astype(np.float32), float(-20 * np.log10(max(gain.min(), 1e-9)))


def mix_cover(*, instrumental: str | Path, converted: str | Path, reference_dry: str | Path, output_mix: str | Path,
              output_vocal: str | Path | None = None, reference_wet: str | Path | None = None,
              backing: str | Path | None = None, backing_converted: str | Path | None = None,
              backing_solo: str | Path | None = None,
              vocal_gain_db: float = 0.0, music_gain_db: float = 0.0, backing_gain_db: float = 0.0,
              clarity: float = 0.75, dynamics: float = 0.6, reverb: float = 1.0) -> dict:
    """Treat the converted lead and write the finished cover (and the treated vocal).

    ``backing`` is the original backing stem. With ``backing_converted`` the whole
    stem is replaced by its conversion; with ``backing_solo`` (also a conversion of
    the stem) only the passages where the stem sings alone are replaced.
    """
    import soundfile as sf

    music = stereo(read_audio(instrumental))
    length = len(music)
    reference = fit_length(read_audio(reference_dry), length)
    wet = fit_length(read_audio(reference_wet), length) if reference_wet else None
    vocal, report = treat_vocal(fit_length(read_audio(converted), length), reference, MIX_SR,
                                clarity=clarity, dynamics=dynamics, reference_wet=wet, reverb=reverb)
    vocal = vocal * 10 ** (vocal_gain_db / 20)
    bed = music * 10 ** (music_gain_db / 20)
    if backing and backing_converted:
        original = fit_length(read_audio(backing), length)
        harmony, report["backing_treatment"] = treat_vocal(
            fit_length(read_audio(backing_converted), length), original, MIX_SR,
            clarity=clarity, dynamics=dynamics, reverb=0.0)
        bed = bed + harmony * 10 ** (backing_gain_db / 20)
    elif backing:
        original = stereo(fit_length(read_audio(backing), length))
        harmony = original
        lead_stem = reference + wet if wet is not None else reference
        weight, spans, exposure = solo_backing_mask(lead_stem, original, MIX_SR)
        report["backing_solo"] = {**exposure, "spans": [], "replaced_seconds": 0.0,
                                  "left_exposed_seconds": exposure["exposed_seconds"]}
        if backing_solo and spans:
            solo, _ = treat_vocal(fit_length(read_audio(backing_solo), length), original, MIX_SR,
                                  clarity=clarity, dynamics=dynamics, reverb=0.0)
            # match the level inside the replaced phrases themselves, where the line is heard
            wanted = float(np.sum((mono(original).astype(np.float64) * weight) ** 2))
            current = float(np.sum((mono(solo).astype(np.float64) * weight) ** 2))
            gain = float(np.clip(np.sqrt(wanted / current), 10 ** (-6 / 20), 10 ** (6 / 20))) if wanted > 0 and current > 0 else 1.0
            line = solo * gain * weight[:, None]
            if report["space"].get("send_db") is not None:  # same room as the lead
                line = line + add_space(mono(line), MIX_SR, wet_ratio_db=report["space"]["send_db"],
                                        rt60=report["space"]["rt60"])
            harmony = original * (1.0 - weight)[:, None] + line
            report["backing_solo"] = {**exposure, "spans": [[round(start, 2), round(end, 2)] for start, end in spans],
                                      "gain_db": float(20 * np.log10(gain))}
        bed = bed + harmony * 10 ** (backing_gain_db / 20)
    mixed = bed + vocal
    # A backing track separated from a loud master overshoots full scale as soon as
    # anything is added to it. Turning the whole mix down is inaudible; squashing it
    # with the limiter is not, so only the rarest peaks are left for the limiter.
    busy = float(np.percentile(np.abs(mixed).max(axis=1), 99.99))
    headroom = min(1.0, 0.95 / busy) if busy > 0 else 1.0
    mixed, reduction = limit(mixed * headroom, MIX_SR)
    if not np.isfinite(mixed).all():
        raise RuntimeError("混音结果出现无效数值，已停止写入成品；请保留本次文件夹以便排查。")
    sf.write(str(output_mix), mixed, MIX_SR, subtype="PCM_24")
    if output_vocal:
        solo, _ = limit(vocal, MIX_SR, ceiling=0.98)
        sf.write(str(output_vocal), solo, MIX_SR, subtype="PCM_24")
    report.update({"mix_gain_db": float(20 * np.log10(headroom)), "limiter_reduction_db": reduction,
                   "vocal_gain_db": vocal_gain_db,
                   "music_gain_db": music_gain_db, "backing": "converted" if backing_converted else bool(backing),
                   "backing_gain_db": backing_gain_db})
    return report
