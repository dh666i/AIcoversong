"""Repair very short artificial 100 Hz pulse trains in converted fricatives."""
import numpy as np


def _periodicity(audio, band):
    """10 ms and 5 ms periodicity, strongest vocal periodicity, and band level."""
    from scipy import fft, signal

    rate, size, hop = 16000, 640, 80
    sos = signal.butter(3, band, fs=rate, btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos.astype("float32"), audio)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(filtered, size // 2), size)[::hop]
    window = np.hanning(size).astype("float32")
    correction = signal.correlate(window, window, mode="full")[size - 1:]
    values = np.empty((len(frames), 4), dtype="float32")
    for start in range(0, len(frames), 512):
        block = frames[start:start + 512] * window
        power = np.abs(fft.rfft(block, n=2 * size, axis=1)) ** 2
        ac = fft.irfft(power, axis=1)[:, :241]
        energy = np.maximum(ac[:, :1], 1e-10)
        r = ac / energy * (correction[0] / correction[:241])
        values[start:start + len(block)] = np.column_stack((
            r[:, 160], r[:, 80], r[:, 20:].max(axis=1),
            10 * np.log10(energy[:, 0] / correction[0]),
        ))
    return values.T


def _diffuse_pulses(high, start, end, sr):
    """Estimate the repeating waveform and spread its line energy into frication."""
    from scipy import ndimage, signal

    center = np.arange(start, end)
    offsets = np.rint(np.arange(-2, 3) * sr / 100).astype(int)
    cycles = np.stack([high[np.clip(center + off, 0, len(high) - 1)] for off in offsets])
    # A nearby vowel can have far more energy than the fricative. Project onto
    # the current cycle locally so a loud neighbour cannot enter this phrase.
    radius = max(3, round(sr / 100))
    current = high[start:end]
    cross = ndimage.uniform_filter1d((cycles * current).astype("float64"), radius, axis=1)
    other = ndimage.uniform_filter1d(cycles.astype("float64") ** 2, radius, axis=1)
    coherent = np.clip(cross / np.maximum(other, 1e-12), 0, 1.2)
    periodic = (cycles * coherent).mean(axis=0).astype("float32")

    # Preserve the spectral envelope of the *converted* consonant, including
    # its energy, while removing the repeated excitation. Use a deterministic
    # noise excitation so the same cover remixes identically.
    size = min(len(periodic), max(16, round(sr * .02)))
    args = dict(fs=sr, nperseg=size, noverlap=size * 3 // 4)
    freqs, _, spectrum = signal.stft(periodic, **args)
    noise = np.random.default_rng(start).standard_normal(len(periodic)).astype("float32")
    _, _, excitation = signal.stft(noise, **args)
    width = max(1, 80 * size / sr)  # smooth the 100 Hz harmonic comb
    shape = ndimage.gaussian_filter1d(np.abs(spectrum) ** 2, width, axis=0)
    noise_power = ndimage.gaussian_filter1d(np.abs(excitation) ** 2, width, axis=0)
    excitation *= np.sqrt(shape / np.maximum(noise_power, 1e-12))
    excitation[freqs < 1000] = 0
    _, diffuse = signal.istft(excitation, **args)
    diffuse = diffuse[:len(periodic)]
    replacement_power = ndimage.uniform_filter1d(diffuse.astype("float64") ** 2, radius)
    nominal_power = ndimage.uniform_filter1d(periodic.astype("float64") ** 2, radius)
    diffuse *= np.clip(np.sqrt(np.maximum(nominal_power, 0) / np.maximum(replacement_power, 1e-12)), .5, 2)
    return diffuse - periodic


def reduce_periodic_buzz(lead, reference, sr):
    """Touch only phase-locked upper harmonics with an unvoiced source."""
    import soxr
    from scipy import signal

    report = {"spans": [], "affected_seconds": 0.0}
    if sr < 16000 or len(lead) < sr // 12 or not np.any(lead):
        return lead, report
    converted = soxr.resample(lead, sr, 16000) if sr != 16000 else lead
    source = soxr.resample(reference, sr, 16000) if sr != 16000 else reference
    hi = (1000, 7500)
    grid, half, _, level = _periodicity(converted, hi)
    source_grid, _, source_high_pitch, source_level = _periodicity(source, hi)
    _, _, source_pitch, _ = _periodicity(source, (65, 2200))
    # A voiced source vetoes correction at any pitch, even after transposition.
    unvoiced = (source_pitch < .76) & (source_high_pitch < .74) & (source_level > -63)
    signature = unvoiced & (grid - half > .28) & (grid - source_grid > .42) & (level > -54)
    strong = signature & (grid > .78)
    candidate = signature & (grid > .62)
    spans = np.flatnonzero(np.diff(np.r_[False, candidate, False])).reshape(-1, 2)
    spans = [(a, b) for a, b in spans if strong[a:b].sum() >= 4]
    if not spans:
        return lead, report

    # Linear phase prevents a timing shift between the subtracted and original
    # waveforms. All changes are tapered within the detected intervals.
    taps = signal.firwin(129, 1200, pass_zero=False, fs=sr).astype("float32")
    out = lead.copy()
    for a, b in spans:
        start = max(0, round((a - .5) * .005 * sr))
        end = min(len(lead), round((b - .5) * .005 * sr))
        context = round(.03 * sr) + len(taps)
        lo, hi_sample = max(0, start - context), min(len(lead), end + context)
        high = signal.oaconvolve(lead[lo:hi_sample], taps, mode="same")
        change = _diffuse_pulses(high, start - lo, end - lo, sr)
        idx = np.arange(end - start)
        fade = np.clip(np.minimum(idx, idx[::-1]) / (.006 * sr), 0, 1)
        out[start:end] += (.90 * np.sin(fade * np.pi / 2) ** 2 * change).astype("float32")
        report["spans"].append([round(start / sr, 4), round(end / sr, 4)])
    report["affected_seconds"] = round(sum(b - a for a, b in report["spans"]), 3)
    return out, report
