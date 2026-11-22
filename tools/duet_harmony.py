"""Reassign harmony shared with the wrong duet stem before voice conversion."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def _ramp(value, low, high):
    return np.clip((value - low) / (high - low), 0, 1)


def _cross(x, y):
    """Local covariance over frequency, both channels, and about 90 ms."""
    moment = np.einsum("fct,fct->t", x.real, y.real) + np.einsum("fct,fct->t", x.imag, y.imag)
    return uniform_filter1d(moment.astype("float64") / x.shape[1], 9, mode="nearest")


def _levels(power):
    return 10 * np.log10(np.maximum(power, 1e-14))


def _projections(lead, harmony, donor, receiver):
    # Regress the two baseline stems onto the harmony after projecting out
    # the lead. An octave or an overlapping harmonic alone proves no leakage.
    ll = np.maximum(_cross(lead, lead), 1e-14)
    hh = np.maximum(_cross(harmony, harmony), 1e-14)
    lh = _cross(lead, harmony)
    ld, hd = _cross(lead, donor), _cross(harmony, donor)
    lr, hr = _cross(lead, receiver), _cross(harmony, receiver)
    independent = np.clip(1 - lh**2 / (ll * hh), 0, 1)
    denom = np.maximum(hh * independent, 1e-14)
    lost = (hd - lh * ld / ll) / denom
    already = (hr - lh * lr / ll) / denom
    donor_lead = (ld - lh * hd / hh) / np.maximum(ll * independent, 1e-14)
    return lost, already, donor_lead, independent, ll, hh


def _missing_harmony(donor, receiver, lead, other_lead, harmony, other_harmony, frequencies):
    lost, already, donor_lead, independent, lead_power, harmony_power = _projections(
        lead, harmony, donor, receiver)
    lead_db, harmony_db = _levels(lead_power), _levels(harmony_power)
    lead_margin = lead_db - _levels(_cross(other_lead, other_lead))
    harmony_margin = harmony_db - _levels(_cross(other_harmony, other_harmony))
    # Require opposite role assignments from the two isolated layers, and
    # evidence that these layers describe the actual, synchronous baseline.
    confidence = _ramp(lead_margin, 6, 14) * _ramp(harmony_margin, 6, 14)
    confidence *= _ramp(harmony_db, -66, -54) * _ramp(lead_db, -66, -54)
    confidence *= _ramp(harmony_db - lead_db, -34, -23)
    confidence *= _ramp(lead_db - harmony_db, -28, -16)
    confidence *= _ramp(independent, .14, .44) * _ramp(donor_lead, .30, .65)
    confidence *= 1 - _ramp(np.abs(lost + already - 1), .18, .42)
    if not np.any(confidence > 0):
        return np.zeros_like(harmony)

    # Track the missing fraction separately by register. A separator often
    # retains the soprano's overtones but sends her fundamental to the tenor.
    band_edges = [0, 300, 600, 1100, 2000, 3800, 7500, np.inf]
    above_floor = np.maximum(0.00012 * harmony_power, 1e-13)
    gain = np.zeros((len(frequencies), len(confidence)), dtype="float32")
    for low, high in zip(band_edges, band_edges[1:]):
        bins = (frequencies >= low) & (frequencies < high)
        if not bins.any():
            continue
        borrowed, present, _, distinct, _, power = _projections(
            lead[bins], harmony[bins], donor[bins], receiver[bins])
        missing = np.clip(np.minimum(borrowed, 1 - present), 0, 1)
        missing *= _ramp(missing, .035, .12) * _ramp(distinct, .10, .35)
        missing *= _ramp(power / above_floor, .5, 2)
        missing *= 1 - _ramp(abs(borrowed + present - 1), .25, .55)
        gain[bins] = gaussian_filter1d(confidence * missing, 1.1)
    # Taper the transition between adjacent analysis bands as well as frames.
    gain = gaussian_filter1d(gain, 1.25, axis=0)
    return harmony * gain[:, None, :]


def _read(handle, start, stop, channels):
    handle.seek(min(start, handle.frames))
    block = handle.read(stop - start, always_2d=True, fill_value=0, dtype="float32")
    if not np.isfinite(block).all():
        raise ValueError("Non-finite samples in the duet separation.")
    if block.shape[1] != channels:
        return block.mean(axis=1, keepdims=True) if channels == 1 else np.repeat(block, 2, axis=1)
    return block


def reconcile_harmony(male, female, lead_roles, backing_roles, output_male, output_female, *, swap=False):
    """Write a mixture-preserving pair from independent lead/backing role estimates.

    The baseline pair is already labelled; swap exchanges only the candidate
    labels. Process with overlapping context so memory is independent of song
    duration. The two exclusive-create outputs may be used after a successful return.
    """
    roles = ["female", "male"] if swap else ["male", "female"]
    inputs = [male, female, *(lead_roles[r] for r in roles), *(backing_roles[r] for r in roles)]
    paths = tuple(map(Path, (output_male, output_female)))
    totals = dict.fromkeys(("male", "female"), 0)
    created = []
    try:
        with ExitStack() as stack:
            tracks = [stack.enter_context(sf.SoundFile(p)) for p in inputs]
            frames, sr, channels = tracks[0].frames, tracks[0].samplerate, tracks[0].channels
            if not frames or any(t.channels not in (1, 2) or t.samplerate != sr or
                                 abs(t.frames - frames) > .02 * sr for t in tracks):
                raise ValueError("Harmony estimates must share the baseline sample rate and time grid.")
            writers = []
            for path in paths:
                writers.append(stack.enter_context(sf.SoundFile(
                    path, mode="x", format="WAV", subtype="FLOAT", samplerate=sr, channels=channels)))
                created.append(path)
            hop = max(1, round(.010 * sr))
            options = dict(fs=sr, nperseg=4 * hop, noverlap=3 * hop)
            for start in range(0, frames, 300 * hop):
                end = min(frames, start + 300 * hop)
                left, right = max(0, start - 20 * hop), min(frames, end + 20 * hop)
                blocks = [_read(track, left, right, channels) for track in tracks]
                spectra = []
                for block in blocks:
                    if len(block) < 4 * hop:
                        block = np.pad(block, ((0, 4 * hop - len(block)), (0, 0)))
                    frequencies, _, spectrum = signal.stft(block, axis=0, **options)
                    spectra.append(spectrum)
                m, f, lm, lf, bm, bf = spectra
                corrections = []
                for role, args in (("female", (m, f, lm, lf, bf, bm)), ("male", (f, m, lf, lm, bm, bf))):
                    delta = _missing_harmony(*args, frequencies)
                    _, audio = signal.istft(delta, freq_axis=0, time_axis=2, **options)
                    part = audio.T[start - left:end - left]
                    corrections.append(part)
                    totals[role] += int(np.count_nonzero(np.max(np.abs(part), axis=1) > 1e-4))
                transfer = corrections[0] - corrections[1]
                interior = slice(start - left, end - left)
                writers[0].write(blocks[0][interior] - transfer)
                writers[1].write(blocks[1][interior] + transfer)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {"reassigned_seconds": {role: round(count / sr, 3) for role, count in totals.items()}}
