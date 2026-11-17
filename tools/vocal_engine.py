"""Song-cover voice conversion on top of Applio's RVC modules.

Importing this module loads no model. It replaces Applio's stock
``Pipeline.pipeline`` for covers:

* Chunk cuts are placed in pauses on the 20 ms feature grid and the chunks are
  cross-faded at their true timestamps to reduce discontinuities and preserve
  alignment when a chunk has an odd number of frames.
* The pitch contour that drives the voice model is tidied first (octave flips,
  few-frame spikes, voicing flicker); every change is counted in the report.
  These conservative rules do not guarantee pitch accuracy or pronunciation.
  For harmonies, separating the lead first also matters: see the
  "refine" stage of cover_pipeline.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "Applio"

SR = 16000          # rate of everything fed to the pitch tracker and ContentVec
HOP = 160           # 10 ms pitch frames
FEAT_HOP = 320      # 20 ms ContentVec frames
PAD_SECONDS = 2     # context given to every chunk on both sides
BINS_PER_OCTAVE = 60  # RMVPE salience bins are 20 cents apart


# --------------------------------------------------------------------------
# Pitch contour repair (pure numpy)
# --------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _octave_support(salience: np.ndarray, shift: int) -> np.ndarray:
    """Salience found one octave away from each frame's main peak, relative to that peak."""
    frames, bins = salience.shape
    main = salience.argmax(1)
    peak = np.maximum(salience[np.arange(frames), main], 1e-9)
    target = main + shift * BINS_PER_OCTAVE
    best = np.zeros(frames, dtype=np.float32)
    for offset in range(-3, 4):
        index = target + offset
        valid = (index >= 0) & (index < bins)
        values = np.zeros(frames, dtype=np.float32)
        values[valid] = salience[np.flatnonzero(valid), index[valid]]
        best = np.maximum(best, values)
    return best / peak


def fix_octaves(f0: np.ndarray, salience: np.ndarray | None = None, *, hold_cost: float = 0.08,
                max_gap: int = 6, min_support: float = 0.08) -> tuple[np.ndarray, int]:
    """Undo short octave flips with a Viterbi search over {-1, 0, +1} octave shifts.

    Keeping the tracker's value is free; holding a shifted value costs ``hold_cost``
    per frame; moving costs the distance in octaves. A flipped stretch is therefore
    folded back only when it is short (two jumps of one octave outweigh about
    2 / hold_cost frames) and, when the salience map is available, only if the
    tracker itself saw energy at the other octave.
    """
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) < 2:
        return f0.copy(), 0
    logf = np.log2(f0[voiced])
    shifts = np.array([-1.0, 0.0, 1.0])
    unary = np.zeros((len(voiced), 3))
    unary[:, 0] = unary[:, 2] = hold_cost
    if salience is not None:
        for column, shift in ((0, -1), (2, 1)):
            support = _octave_support(salience, shift)[voiced]
            unary[support < min_support, column] = np.inf
    cost = unary[0].copy()
    back = np.zeros((len(voiced), 3), dtype=np.int8)
    for i in range(1, len(voiced)):
        if voiced[i] - voiced[i - 1] > max_gap + 1:
            best = int(np.argmin(cost))
            back[i] = best
            cost = cost[best] + unary[i]
            continue
        previous = logf[i - 1] + shifts
        current = logf[i] + shifts
        move = np.minimum(np.abs(current[None, :] - previous[:, None]), 1.25)
        total = cost[:, None] + move
        back[i] = np.argmin(total, axis=0)
        cost = total[back[i], np.arange(3)] + unary[i]
    state = int(np.argmin(cost))
    chosen = np.zeros(len(voiced), dtype=np.int8)
    for i in range(len(voiced) - 1, -1, -1):
        chosen[i] = state
        state = int(back[i, state])
    out = f0.copy()
    out[voiced] = f0[voiced] * np.power(2.0, shifts[chosen])
    return out, int(np.count_nonzero(chosen != 1))


def remove_spikes(f0: np.ndarray, *, max_len: int = 4, leave: float = 250.0, back: float = 150.0) -> tuple[np.ndarray, int]:
    """Bridge excursions of at most ``max_len`` frames that jump away and come straight back."""
    out = f0.copy()
    fixed = 0
    n = len(out)
    i = 1
    while i < n - 1:
        if out[i] > 0 and out[i - 1] > 0 and abs(1200 * np.log2(out[i] / out[i - 1])) > leave:
            anchor = out[i - 1]
            for length in range(1, max_len + 1):
                j = i + length
                if j >= n or out[j] <= 0:
                    break
                if abs(1200 * np.log2(out[j] / anchor)) <= back:
                    ramp = np.linspace(np.log2(anchor), np.log2(out[j]), length + 2)[1:-1]
                    out[i:j] = np.power(2.0, ramp)
                    fixed += length
                    i = j
                    break
        i += 1
    return out, fixed


def trim_edges(f0: np.ndarray, *, max_len: int = 3, leave: float = 250.0, steady: float = 120.0) -> tuple[np.ndarray, int]:
    """Pull the first/last frames of a note onto it when they start or end far off pitch."""
    out = f0.copy()
    fixed = 0
    for start, end in _runs(out > 0):
        if end - start < max_len + 4:
            continue
        for length in range(max_len, 0, -1):
            body = out[start + length:start + length + 4]
            if (abs(1200 * np.log2(out[start + length - 1] / body[0])) > leave
                    and np.ptp(1200 * np.log2(body / body[0])) < steady):
                out[start:start + length] = body[0]
                fixed += length
                break
        for length in range(max_len, 0, -1):
            body = out[end - length - 4:end - length]
            if (abs(1200 * np.log2(out[end - length] / body[-1])) > leave
                    and np.ptp(1200 * np.log2(body / body[-1])) < steady):
                out[end - length:end] = body[-1]
                fixed += length
                break
    return out, fixed


def fill_gaps(f0: np.ndarray, level_db: np.ndarray | None = None, *, max_len: int = 3,
              max_cents: float = 200.0, max_drop_db: float = 12.0) -> tuple[np.ndarray, int]:
    """Close voicing drop-outs of a few frames inside a held note."""
    out = f0.copy()
    fixed = 0
    for start, end in _runs(out <= 0):
        if start == 0 or end >= len(out) or end - start > max_len:
            continue
        left, right = out[start - 1], out[end]
        if abs(1200 * np.log2(right / left)) > max_cents:
            continue
        if level_db is not None:
            around = min(level_db[start - 1], level_db[end])
            if level_db[start:end].min() < around - max_drop_db:
                continue
        ramp = np.linspace(np.log2(left), np.log2(right), end - start + 2)[1:-1]
        out[start:end] = np.power(2.0, ramp)
        fixed += end - start
    return out, fixed


def drop_islands(f0: np.ndarray, confidence: np.ndarray | None = None, *, max_len: int = 3,
                 max_confidence: float = 0.3) -> tuple[np.ndarray, int]:
    """Unvoice isolated low-confidence blips: they render as short buzzes."""
    if confidence is None:
        return f0.copy(), 0
    out = f0.copy()
    fixed = 0
    for start, end in _runs(out > 0):
        if end - start <= max_len and confidence[start:end].max() < max_confidence:
            out[start:end] = 0
            fixed += end - start
    return out, fixed


def count_jumps(f0: np.ndarray, threshold: float = 300.0) -> int:
    voiced = f0 > 0
    both = voiced[1:] & voiced[:-1]
    step = np.zeros(len(f0) - 1)
    step[both] = np.abs(1200 * np.log2(f0[1:][both] / f0[:-1][both]))
    return int(np.count_nonzero(step > threshold))


def refine_f0(f0: np.ndarray, salience: np.ndarray | None = None, level_db: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Repair a 10 ms pitch contour. Returns the new contour and what was changed."""
    f0 = np.asarray(f0, dtype=np.float64)
    confidence = salience.max(1) if salience is not None else None
    report = {"frames": int(len(f0)), "voiced_before": int(np.count_nonzero(f0 > 0)),
              "jumps_before": count_jumps(f0)}
    out, report["spike_frames"] = remove_spikes(f0)
    out, report["octave_frames"] = fix_octaves(out, salience)
    out, more = remove_spikes(out)
    report["spike_frames"] += more
    out, report["edge_frames"] = trim_edges(out)
    out, report["gap_frames"] = fill_gaps(out, level_db)
    out, report["island_frames"] = drop_islands(out, confidence)
    report["jumps_after"] = count_jumps(out)
    report["voiced_after"] = int(np.count_nonzero(out > 0))
    return out, report


# --------------------------------------------------------------------------
# Does the song fit the voice?
# --------------------------------------------------------------------------

def training_pitch_range(model_path: str | Path) -> tuple[float, float] | None:
    """(5th, 99th) percentile in Hz of the pitch the voice model was trained on.

    Read from the f0 files Applio keeps next to the model; None when they are
    gone (a model copied in from elsewhere).
    """
    folder = Path(model_path).resolve().parent / "f0_voiced"
    files = sorted(folder.glob("*.npy")) if folder.is_dir() else []
    if len(files) < 20:
        return None
    step = max(1, len(files) // 400)
    values = [np.load(str(path), mmap_mode="r") for path in files[::step]]
    pitch = np.concatenate([np.asarray(v)[np.asarray(v) > 0] for v in values])
    if len(pitch) < 2000:
        return None
    low, high = np.percentile(pitch, [5, 99])
    return float(low), float(high)


def suggest_octave(f0: np.ndarray, trained: tuple[float, float] | None, *, min_gain: float = 0.10,
                   max_below: float = 0.05) -> tuple[int, dict]:
    """Semitones (0 or -12) to transpose a vocal so that it fits the voice model.

    A voice model only sounds like its speaker inside the pitch range it was
    trained on: on notes above it the output turns thin and anonymous (measured
    on a song whose chorus sits a fourth above the training range: one third of
    those frames came out more than 12 dB too quiet, and speaker similarity fell
    from 0.89 to 0.48). An octave keeps the melody in key with the backing track.
    The drop is suggested only when it brings clearly more of the melody into
    the trained range and pushes almost nothing below it.
    """
    voiced = np.asarray(f0, dtype=np.float64)
    voiced = voiced[voiced > 0]
    info = {"trained_low_hz": None, "trained_high_hz": None, "in_range_now": None, "in_range_octave_down": None,
            "above_range_now": None, "below_range_octave_down": None, "median_hz": float(np.median(voiced)) if len(voiced) else None}
    if trained is None or len(voiced) < 200:
        return 0, info
    low, high = trained
    now = float(np.mean((voiced >= low) & (voiced <= high)))
    down = float(np.mean((voiced / 2 >= low) & (voiced / 2 <= high)))
    below = float(np.mean(voiced / 2 < low))
    info.update({"trained_low_hz": low, "trained_high_hz": high, "in_range_now": now, "in_range_octave_down": down,
                 "above_range_now": float(np.mean(voiced > high)), "below_range_octave_down": below})
    return (-12 if down - now >= min_gain and below <= max_below else 0), info


# --------------------------------------------------------------------------
# Chunk boundaries
# --------------------------------------------------------------------------

def frame_level_db(audio: np.ndarray, hop: int) -> np.ndarray:
    count = len(audio) // hop
    frames = audio[:count * hop].reshape(count, hop)
    return 10 * np.log10(np.maximum(np.mean(frames ** 2, axis=1), 1e-14))


def choose_cuts(audio: np.ndarray, f0: np.ndarray, *, target: float = 30.0, search: float = 8.0) -> list[int]:
    """Cut positions (samples, multiples of FEAT_HOP), preferably in the middle of a pause."""
    total = len(audio) // FEAT_HOP
    level = frame_level_db(audio, FEAT_HOP)
    silent = level < np.percentile(level, 95) - 45
    voiced = np.zeros(total, dtype=bool)
    usable = min(total, len(f0) // 2)
    voiced[:usable] = (f0[0:2 * usable:2] > 0) | (f0[1:2 * usable:2] > 0)
    per_second = SR // FEAT_HOP
    step, radius = int(target * per_second), int(search * per_second)
    cuts = [0]
    position = step
    while position < total - radius:
        low, high = position - radius, min(position + radius, total)
        pauses = [(a + low, b + low) for a, b in _runs(silent[low:high]) if b - a >= 5]
        if pauses:
            a, b = max(pauses, key=lambda item: (item[1] - item[0], -abs((item[0] + item[1]) // 2 - position)))
            cut = (a + b) // 2
        else:
            candidates = np.flatnonzero(~voiced[low:high])
            if len(candidates) == 0:
                candidates = np.arange(high - low)
            cut = low + int(candidates[np.argmin(level[low:high][candidates])])
        cuts.append(cut * FEAT_HOP)
        position = cut + step
    cuts.append(total * FEAT_HOP)
    return cuts


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def _enter_applio() -> None:
    os.chdir(APPLIO)
    if str(APPLIO) not in sys.path:
        sys.path.insert(0, str(APPLIO))


def _salience(predictor, audio: np.ndarray, chunk_seconds: int = 10) -> np.ndarray:
    """RMVPE salience map on the frame clock Applio uses for its own pitch extraction."""
    import torch

    chunk_frames = chunk_seconds * SR // HOP
    frame_count = len(audio) // HOP + 1
    out = np.zeros((frame_count, 360), dtype=np.float32)
    context = SR // HOP

    def infer(window: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(window).float().to(predictor.device).unsqueeze(0)
            hidden = predictor.mel2hidden(predictor.mel_extractor(tensor, center=True))
            return hidden.squeeze(0).float().cpu().numpy()

    if len(audio) <= chunk_frames * HOP:
        hidden = infer(audio)
        usable = min(len(hidden), frame_count)
        out[:usable] = hidden[:usable]
        return out
    for start in range(0, frame_count - 1, chunk_frames):
        end = min(start + chunk_frames, frame_count - 1)
        if end == frame_count - 1:
            end = frame_count
        left = max(0, start - context)
        right = min(len(audio), (end + context) * HOP)
        hidden = infer(audio[left * HOP:right])
        offset = start - left
        out[start:end] = hidden[offset:offset + end - start]
    return out


class VocalEngine:
    """Loads one voice model and converts any number of vocal stems with it."""

    def __init__(self, model_path: str, index_path: str = "", embedder: str = "contentvec"):
        _enter_applio()
        import faiss
        import torch
        from rvc.infer.infer import VoiceConverter
        from rvc.lib.predictors.f0 import RMVPE

        self.torch = torch
        self.converter = VoiceConverter()
        self.converter.get_vc(model_path, 0)
        self.converter.load_hubert(embedder, None)
        self.vc = self.converter.vc
        self.tgt_sr = int(self.converter.tgt_sr)
        self.device = self.vc.device
        self.rmvpe = RMVPE(device=self.device, sample_rate=SR, hop_size=HOP, high_register={"enabled": False})
        self.index = self.big_npy = None
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

    # -- pitch ---------------------------------------------------------------
    def track_pitch(self, audio: np.ndarray, threshold: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        scaled = audio.astype(np.float32)
        scale = np.quantile(np.abs(scaled), 0.999)
        if scale > 0:
            scaled = scaled / scale
        salience = _salience(self.rmvpe.model, scaled)
        f0 = self.rmvpe.model.decode(salience, thred=threshold)
        return f0, salience

    def _coarse(self, f0: np.ndarray) -> np.ndarray:
        vc = self.vc
        mel = 1127 * np.log(1 + f0 / 700)
        mel[mel > 0] = (mel[mel > 0] - vc.f0_mel_min) * 254 / (vc.f0_mel_max - vc.f0_mel_min) + 1
        mel[mel <= 1] = 1
        mel[mel > 255] = 255
        return np.rint(mel).astype(int)

    # -- conversion ----------------------------------------------------------
    def melody(self, input_path: str) -> np.ndarray:
        """Pitch contour (10 ms frames, 0 = unvoiced) of a vocal, as the conversion would hear it."""
        from scipy import signal
        from rvc.infer.pipeline import ah, bh
        from rvc.lib.utils import load_audio_infer

        audio = load_audio_infer(input_path, SR)
        peak = np.abs(audio).max() / 0.95
        if peak > 1:
            audio = audio / peak
        audio = signal.filtfilt(bh, ah, audio)
        f0, salience = self.track_pitch(audio)
        level = frame_level_db(np.pad(audio, (HOP // 2, HOP // 2)), HOP)[:len(f0)]
        level = np.pad(level, (0, len(f0) - len(level)), constant_values=-140.0)
        return refine_f0(f0, salience, level)[0]

    def convert(self, input_path: str, output_path: str, *, pitch: int = 0, index_rate: float = 0.25,
                protect: float = 0.33, repair_pitch: bool = True, seed: int = 0,
                crossfade_ms: float = 30.0, progress=None) -> dict:
        import soundfile as sf
        from scipy import signal
        from rvc.infer.pipeline import ah, bh
        from rvc.lib.utils import load_audio_infer

        torch = self.torch
        audio = load_audio_infer(input_path, SR)
        peak = np.abs(audio).max() / 0.95
        if peak > 1:
            audio = audio / peak
        audio = signal.filtfilt(bh, ah, audio)
        source_len = len(audio)
        # Reflect padding must stay shorter than the audio itself.
        pad = min(PAD_SECONDS * SR, (source_len - FEAT_HOP - HOP - 1) // FEAT_HOP * FEAT_HOP)
        if pad < FEAT_HOP:
            raise ValueError("人声太短（不足 0.1 秒），无法转换。")
        grid_len = -(-source_len // FEAT_HOP) * FEAT_HOP
        padded = np.pad(audio, (pad, pad + grid_len - source_len + HOP), mode="reflect")

        f0_raw, salience = self.track_pitch(padded)
        level = frame_level_db(np.pad(padded, (HOP // 2, HOP // 2)), HOP)[:len(f0_raw)]
        level = np.pad(level, (0, len(f0_raw) - len(level)), constant_values=-140.0)
        if repair_pitch:
            f0, report = refine_f0(f0_raw, salience, level)
        else:
            f0, report = f0_raw.astype(np.float64), {"jumps_before": count_jumps(f0_raw), "jumps_after": count_jumps(f0_raw)}
        f0 = f0 * pow(2, pitch / 12)
        pitch_coarse = torch.tensor(self._coarse(f0.copy()), device=self.device).unsqueeze(0).long()
        pitch_float = torch.tensor(f0, device=self.device).unsqueeze(0).float()

        inner_f0 = f0[pad // HOP:pad // HOP + grid_len // HOP]
        cuts = choose_cuts(np.pad(audio, (0, grid_len - source_len)), inner_f0)
        ratio = self.tgt_sr / SR
        out_len = int(round(grid_len * ratio))
        mixed = np.zeros(out_len, dtype=np.float64)
        weight = np.zeros(out_len, dtype=np.float64)
        half = int(round(crossfade_ms / 1000 * self.tgt_sr / 2))
        sid = torch.tensor(0, device=self.device).unsqueeze(0).long()
        use_f0 = bool(self.converter.use_f0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        pieces = list(zip(cuts[:-1], cuts[1:]))
        for number, (start, end) in enumerate(pieces):
            if progress:
                progress(number, len(pieces))
            segment = padded[start:end + 2 * pad + HOP]
            frames = slice(start // HOP, (end + 2 * pad) // HOP)
            raw = self.vc.voice_conversion(
                self.converter.hubert_model, self.converter.net_g, sid, segment,
                pitch_coarse[:, frames] if use_f0 else None, pitch_float[:, frames] if use_f0 else None,
                self.index if index_rate > 0 else None, self.big_npy, index_rate,
                self.converter.version, protect,
            )
            expected = int(round((end - start + 2 * pad) * ratio))
            if len(raw) < expected:
                raw = np.pad(raw, (0, expected - len(raw)))
            lead = half if number else 0
            tail = half if number + 1 < len(pieces) else 0
            a = int(round(start * ratio)) - lead
            b = int(round(end * ratio)) + tail
            local = int(round(pad * ratio)) - lead
            piece = raw[local:local + b - a].astype(np.float64)
            ramp = np.ones(len(piece))
            if lead:
                ramp[:2 * lead] = np.linspace(0, 1, 2 * lead, endpoint=False) + 0.5 / (2 * lead)
            if tail:
                ramp[-2 * tail:] = ramp[-2 * tail:] * (np.linspace(1, 0, 2 * tail, endpoint=False) - 0.5 / (2 * tail))
            mixed[a:b] += piece * ramp
            weight[a:b] += ramp
        if progress:
            progress(len(pieces), len(pieces))
        mixed /= np.maximum(weight, 1e-9)
        mixed = mixed[:int(round(source_len * ratio))]
        gain = 1.0
        peak = np.abs(mixed).max() / 0.99
        if peak > 1:
            gain = 1 / peak
            mixed *= gain
        sf.write(output_path, mixed.astype(np.float32), self.tgt_sr, subtype="PCM_24")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        voiced = inner_f0[inner_f0 > 0]
        report.update({
            "seconds": source_len / SR, "sample_rate": self.tgt_sr, "chunks": len(pieces),
            "cut_times": [c / SR for c in cuts[1:-1]], "output_gain": gain,
            "pitch_shift": pitch, "index_rate": index_rate, "protect": protect,
            "pitch_percentiles_hz": {str(p): float(np.percentile(voiced, p)) for p in (5, 50, 95, 99)} if len(voiced) else {},
        })
        return report
