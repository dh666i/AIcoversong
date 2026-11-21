"""Local song-cover workers. Importing this module never loads an audio model.

Stages (each runs in its own process so the GPU is fully released in between):

  separate  song            -> vocals + instrumental
  refine    vocals          -> lead / backing vocals, then lead without reverb
  convert   dry lead        -> the voice model's rendition (tools/vocal_engine.py)
  mix       everything      -> treated vocal + finished cover (tools/vocal_mix.py)

``refine`` exists because feeding harmonies and reverb to the voice model made
its pitch flip between voices and octaves, which is heard as an electronic
stutter. Separating a dry lead before conversion reduces competing voices
in the pitch track; the result still depends on the source material.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "Applio"
OUTPUT = ROOT / "output"
MODEL_ROOT = APPLIO / "logs"
UVR_MODELS = APPLIO / "rvc" / "models" / "uvr"
SEPARATOR_MODEL = "mel_band_roformer_kim_ft_unwa.ckpt"
SEPARATOR_CONFIG = "config_mel_band_roformer_kim_ft_unwa.yaml"
LEAD_MODEL = "bs_roformer_karaoke_frazer_becruily.ckpt"
LEAD_CONFIG = "config_bs_roformer_karaoke_frazer_becruily.yaml"
DEREVERB_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
DEREVERB_CONFIG = "dereverb_mel_band_roformer_anvuew.yaml"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
NO_INDEX = "不使用索引"
BACKING_MODES = ("original", "convert", "remove")
# How the lead is fitted to the pitch range of the voice model:
#   auto    one octave down when clearly more of the melody then lies inside the range the model was trained on
#   keep    never transpose
#   down    always one octave down
RANGE_MODES = ("auto", "keep", "down")
STAGE_LABELS = {"separate": "分离人声与伴奏", "refine": "分出主唱、去掉混响",
                "duet": "分开对唱声部",
                "convert": "换成你的声音", "mix": "处理人声并合成"}
# Rough share of the total time each stage takes (measured on six songs of 3.5-5.5 minutes).
STAGE_WEIGHTS = {"separate": 0.60, "refine": 0.25, "duet": 0.30, "convert": 0.10, "mix": 0.05}


def _inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def _audio(path: str | Path | None) -> Path:
    if not path:
        raise ValueError("请先上传音频文件。")
    result = Path(path).resolve()
    if not result.is_file() or result.stat().st_size == 0:
        raise ValueError("音频文件不存在或为空，请重新上传。")
    if result.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError("请上传 WAV、MP3、FLAC、M4A、AAC、OGG 等音频文件。")
    return result


def voice_models() -> list[str]:
    if not MODEL_ROOT.exists():
        return []
    return sorted(
        str(p.relative_to(MODEL_ROOT))
        for p in MODEL_ROOT.rglob("*.pth")
        if p.is_file() and _inside(p, MODEL_ROOT) and not p.name.lower().startswith(("g_", "d_"))
    )


def _voice_path(relative: str | None) -> Path:
    if not relative:
        raise ValueError("还没有选择自己的音色模型。请先在 Applio 的训练页面训练，或先点击“只分离人声和伴奏”。")
    path = (MODEL_ROOT / relative).resolve()
    if not _inside(path, MODEL_ROOT) or not path.is_file() or path.suffix.lower() != ".pth":
        raise ValueError("音色模型必须是 Applio/logs 中已有的 .pth 文件，请刷新列表。")
    if path.name.lower().startswith(("g_", "d_")):
        raise ValueError("请选择导出的音色模型，不要选择 G_ 或 D_ 训练检查点。")
    return path


def voice_indices(relative: str | None) -> list[str]:
    if not relative:
        return [NO_INDEX]
    model = _voice_path(relative)
    return [NO_INDEX] + sorted(
        str(p.relative_to(MODEL_ROOT))
        for p in model.parent.rglob("*.index")
        if p.is_file() and _inside(p, model.parent) and "trained" not in p.name.lower()
    )


def _index_path(relative: str | None, model: Path) -> str:
    if not relative or relative == NO_INDEX:
        return ""
    path = (MODEL_ROOT / relative).resolve()
    if not path.is_file() or path.suffix.lower() != ".index" or not _inside(path, model.parent):
        raise ValueError("索引必须在所选模型的目录中。请选择同一音色训练生成的 .index，或不使用索引。")
    return str(path)


def _number(value, low: float, high: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{label}必须在 {low:g} 到 {high:g} 之间。")
    return result


def refine_available() -> bool:
    return all((UVR_MODELS / name).is_file() for name in (LEAD_MODEL, LEAD_CONFIG, DEREVERB_MODEL, DEREVERB_CONFIG))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_run() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def _environment() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["AUDIO_SEPARATOR_MODEL_DIR"] = str(UVR_MODELS)
    env["PATH"] = str(APPLIO) + os.pathsep + env.get("PATH", "")
    return env


def _process_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# What usually goes wrong, in words that need no English. Matched against the worker's own message.
_KNOWN_FAILURES = (
    (("out of memory", "alloc_failed"),
     "显卡内存不够用，多半是别的程序正占着显卡。请关掉游戏和其他 AI 软件，再试一次。"),
    (("no space left", "not enough space", "disk full"),
     "硬盘空间不够了。请先清理硬盘（output 文件夹里不要的翻唱可以删掉），再试一次。"),
    (("无法把歌曲解码", "invalid data found", "cannot determine format", "format not recognised", "error opening",
      "nobackenderror", "could not open"),
     "这个文件里读不出声音。请换一个文件，或者先把它转成 mp3、flac 或 wav。"),
    (("memoryerror", "unable to allocate", "bad allocation"),
     "电脑内存不够用。请关掉一些占内存的程序再试；特别长的录音可以先剪短。"),
)


def _failure_text(stage: str, detail: str, log_path: Path, from_log: bool = False) -> str:
    """The message the page shows when a step fails.

    ``detail`` is the message the worker left behind, or (from_log) the end of its log.
    """
    label = STAGE_LABELS[stage]
    detail = (detail or "").strip()
    lowered = detail.lower()
    for needles, reason in _KNOWN_FAILURES:
        if any(needle in lowered for needle in needles):
            return f"“{label}”这一步没有完成。\n{reason}"
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    last = (lines[-1] if from_log else lines[0]) if lines else ""
    if last and any("\u4e00" <= ch <= "\u9fff" for ch in last):   # one of this program's own messages
        return f"“{label}”这一步没有完成：{last}"
    return (f"“{label}”这一步没有完成，原因没能自动认出来。\n再试一次；还是不行的话，把这个文件发给帮你维护的人："
            f"\n{log_path}" + (f"\n程序的原话：{last[:300]}" if last else ""))


def _tail(path: Path, length: int = 2600) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - length * 4))
        return stream.read().decode("utf-8", errors="replace")[-length:]


_BAR = re.compile(rb"(\d+)/(\d+) \[")
_CHUNKS = re.compile(r"(主唱|和声)：已转换 (\d+)/(\d+) 段".encode("utf-8"))
_DUET_CHUNKS = re.compile(rb"duet-pass (\d+)/(\d+) chunks (\d+)/(\d+)")
_HALF = "第 2/2 步".encode("utf-8")
_TWO_PASSES = "本步共 2 遍转换".encode("utf-8")   # printed by the convert worker before its first pass
_LEAD, _BACKING = "主唱".encode("utf-8"), "和声".encode("utf-8")


def stage_fraction(stage: str, log_path: Path) -> float:
    """How far a stage is (0..1), read from the log its worker writes. Never decreases while the log grows."""
    if not log_path.exists():
        return 0.0
    with log_path.open("rb") as stream:
        if stage == "convert":   # a few hundred bytes: read it all
            data = stream.read(400_000)
        else:
            size = log_path.stat().st_size
            stream.seek(max(0, size - 6000))
            tail = stream.read()
            stream.seek(0)
            head = stream.read(200_000) if stage == "refine" else b""

    def last(data: bytes) -> float:
        found = _BAR.findall(data)
        if not found:
            return 0.0
        done, total = int(found[-1][0]), int(found[-1][1])
        return min(1.0, done / total) if total else 0.0

    if stage in ("separate", "duet"):
        return last(tail)
    if stage == "refine":  # two models in a row, each about half of the stage
        if _HALF not in head and _HALF not in tail:
            return 0.5 * last(tail)
        position = tail.rfind(_HALF)
        return 0.5 + 0.5 * last(tail[position:] if position >= 0 else tail)
    if stage == "convert":  # loading the model (about a third of the stage), the lead, then perhaps the backing stem
        duet = _DUET_CHUNKS.findall(data)
        if duet:
            i, n, done_chunks, total = map(int, duet[-1])
            return (i + done_chunks / max(total, 1)) / n
        if "本步共".encode("utf-8") not in data:
            return 0.0
        done = {_LEAD: 0.0, _BACKING: 0.0}
        for who, count, total in _CHUNKS.findall(data):
            done[who] = max(done[who], int(count) / int(total) if int(total) else 0.0)
        share = 0.5 if _TWO_PASSES in data else 1.0
        return min(1.0, 0.35 + 0.65 * (share * done[_LEAD] + (1 - share) * done[_BACKING]))
    return 0.0


def _run_stage(stage: str, run: Path):
    """Yield (fraction, seconds) while a worker owns the GPU, then fully exit that worker."""
    log_path = run / f"{stage}.log"
    uses_separator = stage in ("separate", "refine", "duet")
    interpreter = ROOT / "separator-env" / "Scripts" / "python.exe" if uses_separator else Path(sys.executable)
    if not interpreter.is_file():
        raise FileNotFoundError(f"人声分离环境尚未安装完整，缺少：{interpreter}。请先完成 AICover 依赖安装。")
    process = None
    start = time.monotonic()
    best = 0.0
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(interpreter), "-u", str(Path(__file__).resolve()), stage, str(run / "job.json")],
                cwd=str(APPLIO), env=_environment(), stdout=log, stderr=subprocess.STDOUT,
                **_process_options(),
            )
            while process.poll() is None:
                best = max(best, stage_fraction(stage, log_path))   # the page never steps back
                yield best, time.monotonic() - start
                time.sleep(0.5)
        if process.returncode != 0:
            error_file = run / f"{stage}_error.json"
            detail = _read_json(error_file).get("error", "") if error_file.exists() else _tail(log_path)
            raise RuntimeError(_failure_text(stage, detail, log_path, from_log=not error_file.exists()))
        if not (run / f"{stage}_result.json").is_file():
            raise RuntimeError(_failure_text(stage, "", log_path))
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, **_process_options())
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if process is not None and process.returncode != 0:
            # A stopped or failed separator leaves its scaled input and half-written stems behind.
            for leftover in run.glob("tmp-*"):
                try:
                    leftover.unlink()
                except OSError:
                    pass


RESULT_FILES = ("mix", "converted", "vocals", "instrumental", "lead", "backing",
                "duet_male", "duet_female", "duet_male_converted", "duet_female_converted", "duet_preserved")


def _progress(stages: list[str], index: int, fraction: float, seconds: float, started: float, **extra) -> dict:
    """One progress snapshot for the page: which step, how far, how long."""
    weights = [STAGE_WEIGHTS[name] for name in stages]
    fraction = max(0.0, min(1.0, fraction))
    done = sum(weights[:index]) + (weights[index] * fraction if index < len(stages) else 0.0)
    return {"state": "running", "steps": [STAGE_LABELS[name] for name in stages], "step": index,
            "step_fraction": round(fraction, 4), "step_seconds": round(seconds, 1),
            "overall": round(done / sum(weights), 4), "elapsed": round(time.monotonic() - started, 1), **extra}


def run_cover(song, model=None, index=NO_INDEX, pitch=0, index_rate=0.25,
              vocal_gain=0, instrumental_gain=0, separate_only=False,
              refine=True, backing="original", clarity=0.75, reverb=1.0, voice_range="auto",
              duet_mode="off", duet_swap=False, duet_second_model=None, duet_male_stem=None, duet_female_stem=None,
              duet_dereverb=False):
    """User-triggered generator; creates a unique run and preserves every stage.

    Yields dicts with the result files found so far, ``status`` (one plain sentence)
    and ``progress`` (see _progress; its ``state`` is running, done or failed).
    """
    source = _audio(song)
    model_path = _voice_path(model) if not separate_only else None
    selected_index = _index_path(index, model_path) if model_path else ""
    pitch_value = int(_number(pitch, -24, 24, "升降调"))
    rate = _number(index_rate, 0, 1, "索引强度")
    vocal_db = _number(vocal_gain, -24, 12, "人声音量")
    instrumental_db = _number(instrumental_gain, -24, 12, "伴奏音量")
    if backing not in BACKING_MODES:
        raise ValueError("和声处理方式无效，请重新选择。")
    if voice_range not in RANGE_MODES:
        raise ValueError("音区处理方式无效，请重新选择。")
    use_duet = duet_mode != "off"
    second_model = None
    sources = {}
    if use_duet:
        from duet_mix import ROLES, selected_roles

        selected_roles(duet_mode)
        if duet_dereverb and not all((UVR_MODELS / name).is_file() for name in (DEREVERB_MODEL, DEREVERB_CONFIG)):
            raise ValueError("缺少去混响模型，请安装基础模型或关闭对唱声部去混响。")
        if separate_only:
            raise ValueError("只分离伴奏时不能指定对唱翻唱。")
        if bool(duet_male_stem) != bool(duet_female_stem):
            raise ValueError("使用已有分轨时，请同时上传对齐原曲的男声和女声。")
        if duet_male_stem:
            sources = {role: _audio(path) for role, path in zip(ROLES, (duet_male_stem, duet_female_stem))}
            if any(path.suffix.lower() not in {".wav", ".flac"} for path in sources.values()):
                raise ValueError("对唱人声分轨请使用 WAV 或 FLAC。")
            if sources["male"].samefile(sources["female"]):
                raise ValueError("男声和女声不能使用同一份分轨。")
        else:
            from duet_separator import duet_available
            if not duet_available():
                raise ValueError("自动分开对唱需要安装双人声分离模型，或上传两条对齐的人声分轨。")
        if duet_mode == "both" and duet_second_model:
            second_model = _voice_path(duet_second_model)
    use_refine = bool(refine) and not separate_only and not use_duet
    if use_refine and not refine_available():
        raise ValueError(f"精细处理所需的模型不完整（{UVR_MODELS}）。请取消勾选“精细处理”，或重新下载模型。")
    clarity_value = _number(clarity, 0, 1, "人声清晰度")
    reverb_value = _number(reverb, 0, 2, "混响量")
    run = _new_run()
    source_copy = run / ("source" + source.suffix.lower())
    job = {
        "song": str(source_copy), "original_song": str(source), "run_dir": str(run),
        "model": str(model_path) if model_path else "", "index": selected_index,
        "pitch": pitch_value, "index_rate": rate if selected_index else 0,
        "vocal_gain": vocal_db, "instrumental_gain": instrumental_db,
        "refine": use_refine, "backing": backing if use_refine else "original",
        "clarity": clarity_value, "reverb": reverb_value,
        "voice_range": voice_range,
    }
    try:
        shutil.copy2(source, source_copy)
        if use_duet:
            job.update(duet_mode=duet_mode, duet_swap=bool(duet_swap), duet_sources={}, duet_dereverb=bool(duet_dereverb))
            if second_model:
                choices = voice_indices(duet_second_model)
                job.update(duet_second_model=str(second_model),
                           duet_second_index=_index_path(choices[1], second_model) if len(choices) > 1 else "",
                           duet_second_index_rate=rate)
            for role, path in sources.items():
                target = run / ("input_" + role + path.suffix.lower())
                shutil.copy2(path, target)
                job["duet_sources"][role] = str(target)
        _write_json(run / "job.json", job)
    except Exception:
        shutil.rmtree(run, ignore_errors=True)
        raise
    stages = ["separate"]
    if not separate_only:
        if use_duet:
            stages.append("duet")
        elif use_refine:
            stages.append("refine")
        stages.extend(("convert", "mix"))
    started = time.monotonic()
    result = {"status": "准备开始", "vocals": None, "instrumental": None, "converted": None, "mix": None, "files": [],
              "run": str(run), "progress": _progress(stages, 0, 0.0, 0.0, started)}
    position = 0
    try:
        for position, stage in enumerate(stages):
            worker = _run_stage(stage, run)
            try:
                for fraction, seconds in worker:
                    yield {**result, "status": STAGE_LABELS[stage],
                           "progress": _progress(stages, position, fraction, seconds, started)}
            finally:
                worker.close()   # stopped from the page: the worker is gone before anything else happens
            result.update(_read_json(run / f"{stage}_result.json"))
            result["files"] = [result[k] for k in RESULT_FILES if result.get(k)]
        notes = [text for text in (result.get("duet_note"), result.get("range_note"), result.get("note")) if text]
        result["notes"] = notes
        result["status"] = "分离完成" if separate_only else "翻唱完成"
        result["progress"] = {**_progress(stages, len(stages), 1.0, 0.0, started), "state": "done", "notes": notes}
        _write_json(run / "result.json", result)
        yield result
    except GeneratorExit:
        # Stopped before it was finished (the stop button, or the page was closed). A half-made run is of
        # no use and holds a copy of the song, so its folder goes. A finished run is never touched here.
        if not (run / "result.json").is_file():
            for _ in range(12):   # a worker that was just killed can hold its log open for a moment
                shutil.rmtree(run, ignore_errors=True)
                if not run.exists():
                    break
                time.sleep(0.25)
        raise
    except Exception as exc:
        message = str(exc)
        yield {**result, "status": message,
               "progress": {**_progress(stages, position, 0.0, 0.0, started), "state": "failed", "error": message}}


def remix(vocals, instrumental, vocal_gain=0, instrumental_gain=0):
    voice = _audio(vocals)
    music = _audio(instrumental)
    run = _new_run()
    _write_json(run / "job.json", {
        "run_dir": str(run), "converted": str(voice), "instrumental": str(music),
        "vocal_gain": _number(vocal_gain, -24, 12, "人声音量"),
        "instrumental_gain": _number(instrumental_gain, -24, 12, "伴奏音量"),
    })
    try:
        for _fraction, seconds in _run_stage("mix", run):
            yield f"正在合成… {int(seconds)} 秒", None
        result = _read_json(run / "mix_result.json")
        yield f"合成完成：{result['mix']}", result["mix"]
    except Exception as exc:
        yield str(exc), None


def _output_audio(path: Path, run: Path) -> str:
    if not _inside(path, run) or not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError(f"未找到有效的输出音频：{path}")
    return str(path)


def _separator_input(source: Path, run: Path) -> str:
    """Decode compressed input to PCM so its codec tag is valid for WAV output."""
    import soundfile as sf

    ffmpeg = APPLIO / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"缺少 FFmpeg：{ffmpeg}")
    output = run / "separator_input.wav"
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
         "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(output)],
        cwd=str(APPLIO), env=_environment(), capture_output=True, text=True,
        **_process_options(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-1600:]
        raise RuntimeError(f"无法把歌曲解码为分离用 WAV：{detail or completed.returncode}")
    try:
        info = sf.info(str(output))
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("FFmpeg 未生成有效的分离用 WAV；原歌曲文件已保留。") from exc
    if info.format != "WAV" or info.subtype != "PCM_16" or info.samplerate != 44100 or info.channels != 2:
        raise RuntimeError("分离用 WAV 格式不符合要求；原歌曲文件已保留。")
    return str(output)


def _two_stems(source: Path, model_filename: str, names: dict[str, str], run: Path, label: str,
               *, overlap: int, autocast: bool, joint_peak: float | None = None) -> tuple[dict[str, str], float]:
    """Split ``source`` with one two-stem model, keeping the balance between the stems.

    audio-separator rescales each stem on its own once a peak passes its
    threshold, which silently changes the balance between stems. The input is
    therefore fed with 6 dB of float headroom and the stems are scaled back.
    With ``joint_peak`` both stems share one gain so that neither exceeds that
    peak: a stem of a loud master overshoots full scale on its own.
    Returns the written files and the gain that was applied.
    """
    import gc

    import numpy as np
    import soundfile as sf
    import torch
    from audio_separator.separator import Separator

    for filename in (model_filename,):
        if not (UVR_MODELS / filename).is_file():
            raise FileNotFoundError(f"缺少分离模型文件：{UVR_MODELS / filename}")
    audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
    scaled = run / f"tmp-{label}-input.wav"
    sf.write(str(scaled), audio * 0.5, rate, subtype="FLOAT")
    del audio
    separator = Separator(
        model_file_dir=str(UVR_MODELS), output_dir=str(run), output_format="WAV",
        normalization_threshold=1.0, amplification_threshold=0,
        sample_rate=44100, use_autocast=autocast, use_soundfile=True,
        mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": overlap,
                     "override_model_segment_size": False, "pitch_shift": 0},
    )
    separator.load_model(model_filename=model_filename)
    separator.separate(str(scaled), custom_output_names={stem: f"tmp-{label}-{name}" for stem, name in names.items()})
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stems = {}
    for name in dict.fromkeys(names.values()):
        produced = run / f"tmp-{label}-{name}.wav"
        if not produced.is_file():
            raise RuntimeError(f"分离模型没有输出 {name}：{produced}")
        data, out_rate = sf.read(str(produced), dtype="float32", always_2d=True)
        stems[name] = (data * 2.0, out_rate)
        produced.unlink()
    scaled.unlink()
    peak = max(float(np.abs(data).max()) for data, _ in stems.values())
    gain = joint_peak / peak if joint_peak and peak > joint_peak else 1.0
    outputs = {}
    for name, (data, out_rate) in stems.items():
        target = run / f"{name}.wav"
        # Float keeps a stem that peaks above full scale intact instead of clipping or rescaling it.
        sf.write(str(target), data * gain, out_rate, subtype="PCM_24" if peak * gain <= 1.0 else "FLOAT")
        outputs[name] = _output_audio(target, run)
    return outputs, gain


def _separate(job: dict, run: Path) -> dict:
    for filename in (SEPARATOR_MODEL, SEPARATOR_CONFIG):
        if not (UVR_MODELS / filename).is_file():
            raise FileNotFoundError(f"缺少人声分离文件：{UVR_MODELS / filename}")
    source = _audio(job["song"])
    separator_input = _separator_input(source, run)
    stems, gain = _two_stems(Path(separator_input), SEPARATOR_MODEL,
                             {"Vocals": "vocals", "Other": "instrumental", "Instrumental": "instrumental"},
                             run, "separate", overlap=8, autocast=False, joint_peak=0.95)
    return {"vocals": stems["vocals"], "instrumental": stems["instrumental"],
            "stem_gain_db": round(20 * math.log10(gain), 2)}


def _refine(job: dict, run: Path) -> dict:
    for filename in (LEAD_MODEL, LEAD_CONFIG, DEREVERB_MODEL, DEREVERB_CONFIG):
        if not (UVR_MODELS / filename).is_file():
            raise FileNotFoundError(f"缺少精细处理模型文件：{UVR_MODELS / filename}")
    separated = _read_json(run / "separate_result.json")
    print("第 1/2 步：分离主唱与和声", flush=True)
    voices, _ = _two_stems(Path(separated["vocals"]), LEAD_MODEL,
                           {"Vocals": "lead_vocals", "Instrumental": "backing_vocals"},
                           run, "lead", overlap=2, autocast=True)
    print("第 2/2 步：去除主唱混响", flush=True)
    dry, _ = _two_stems(Path(voices["lead_vocals"]), DEREVERB_MODEL,
                        {"noreverb": "lead_dry", "reverb": "lead_reverb"},
                        run, "dry", overlap=2, autocast=True)
    return {"lead": voices["lead_vocals"], "backing": voices["backing_vocals"],
            "lead_dry": dry["lead_dry"], "lead_reverb": dry["lead_reverb"]}


def _stage_result(run: Path, stage: str) -> dict:
    path = run / f"{stage}_result.json"
    return _read_json(path) if path.is_file() else {}


def _duet(job: dict, run: Path) -> dict:
    from duet_mix import prepare_uploaded, selected_roles, write_preserved

    separated = _read_json(run / "separate_result.json")
    if job.get("duet_sources"):
        stems = prepare_uploaded(job["duet_sources"], separated["vocals"], run, separated.get("stem_gain_db", 0))
    else:
        from duet_separator import split_duet
        stems = split_duet(separated["vocals"], run, swap=job.get("duet_swap", False))
    stems["duet_preserved"] = write_preserved(stems, job["duet_mode"], run)
    if job.get("duet_dereverb"):
        # Strip the room from the input to RVC, retaining aligned originals for mix and audition.
        for role in selected_roles(job["duet_mode"]):
            dry_key = f"duet_{role}_dry"
            wet_key = f"duet_{role}_room"
            parts, _ = _two_stems(Path(stems[f"duet_{role}"]), DEREVERB_MODEL,
                                  {"noreverb": dry_key, "reverb": wet_key}, run, dry_key,
                                  overlap=2, autocast=True)
            stems[dry_key] = parts[dry_key]
            Path(parts[wet_key]).unlink()
    if not job.get("duet_sources"):
        balance_note = stems.get("duet_balance_note", "")
        stems["duet_note"] = (balance_note + "已按估计的声部拆分对唱。两人的音区接近、同音合唱或有多人和声时可能串声；"
                               "可在“分开听”核对。角色反了可交换，局部分错可改用原始人声分轨。")
    return stems


def _load_engine(model: Path, index: str):
    _voice_path(str(model.relative_to(MODEL_ROOT)))
    import torch

    metadata = torch.load(str(model), map_location="cpu", weights_only=True)
    embedder = metadata.get("embedder_model") or "contentvec"
    if embedder != "contentvec":
        raise ValueError(f"此助手预装 ContentVec，所选模型使用 {embedder}。请在 Applio 推理页选择该模型对应的特征提取器。")
    del metadata
    from vocal_engine import VocalEngine
    return VocalEngine(str(model), index)


def _convert_duet(job: dict, run: Path, stems: dict) -> dict:
    import gc
    import soundfile as sf
    import numpy as np
    from duet_mix import selected_roles, ROLE_NAMES
    import vocal_engine

    roles = selected_roles(job["duet_mode"])
    reports = {}
    result = {"duet_mode": job["duet_mode"]}
    ranges = {}
    engine = None
    loaded = None
    for i, role in enumerate(roles):
        source = stems[f"duet_{role}_dry"] if job.get("duet_dereverb") else stems[f"duet_{role}"]
        with sf.SoundFile(source) as audio:
            if not any(np.max(np.abs(block)) > 1e-5 for block in audio.blocks(blocksize=65536, dtype="float32")):
                raise ValueError(f"没有分离出可转换的{ROLE_NAMES[role]}，请检查或上传对齐的人声分轨。")
        second = role == "female" and job["duet_mode"] == "both" and job.get("duet_second_model")
        model = Path(job["duet_second_model"] if second else job["model"]).resolve()
        index = job["duet_second_index"] if second else job["index"]
        if loaded != (model, index):
            del engine
            gc.collect()
            engine = _load_engine(model, index)
            loaded = (model, index)
        mode = job.get("voice_range", "keep")
        suggested, fit = (0, {})
        if mode == "auto":
            trained = vocal_engine.training_pitch_range(model)
            if trained:
                suggested, fit = vocal_engine.suggest_octave(engine.melody(source), trained)
        octave = -12 if mode == "down" else suggested
        ranges[role] = {"mode": mode, "octave": octave, "suggested": suggested, **fit}
        if octave and mode == "auto":
            result["range_note"] = (result.get("range_note", "") + f"{ROLE_NAMES[role]}已自动降低一个八度。")
        output = run / f"duet_{role}_converted.wav"
        def progress(done, total):
            print(f"duet-pass {i}/{len(roles)} chunks {done}/{total}", flush=True)
        reports[role] = engine.convert(source, str(output), pitch=int(job["pitch"]) + octave,
                                       index_rate=job.get("duet_second_index_rate", 0) if second else job["index_rate"],
                                       protect=0.33, progress=progress)
        result[f"duet_{role}_converted"] = _output_audio(output, run)
    result["duet_voice_ranges"] = ranges
    _write_json(run / "convert_report.json", reports)
    return result


def _convert(job: dict, run: Path) -> dict:
    if job.get("duet_mode", "off") != "off":
        return _convert_duet(job, run, _read_json(run / "duet_result.json"))
    separated = _read_json(run / "separate_result.json")
    refined = _stage_result(run, "refine")
    model = Path(job["model"]).resolve()
    def progress(label):
        return lambda done, total: print(f"{label}：已转换 {done}/{total} 段", flush=True)

    import vocal_engine

    engine = _load_engine(model, job["index"])
    output = run / "converted_vocals.wav"
    lead_input = refined.get("lead_dry") or separated["vocals"]
    # A voice model sounds like its speaker only inside the pitch range it was trained on.
    mode = job.get("voice_range", "keep")
    suggested, fit = vocal_engine.suggest_octave(engine.melody(lead_input), vocal_engine.training_pitch_range(model))
    octave = -12 if mode == "down" else suggested if mode == "auto" else 0
    lead_pitch = int(job["pitch"]) + octave
    result = {"voice_range": {"mode": mode, "octave": octave, "suggested": suggested, **fit}}
    if octave and mode == "auto":
        result["range_note"] = (f"这首歌有 {fit['above_range_now'] * 100:.0f}% 的音比你录过的最高音还高，那么高的地方模型唱不出你的音色，"
                                "所以人声自动降了一个八度（旋律不变，整体唱低了一截），伴奏没变。想保持原来的音高，在“更多设置”里把“歌太高时怎么办”"
                                "改成“保持原来的音高”，再做一次。")
    elif suggested and mode == "keep":
        result["range_note"] = (f"这首歌有 {fit['above_range_now'] * 100:.0f}% 的音比你录过的最高音还高，这些地方听起来会不太像你。"
                                "在“更多设置”里把“歌太高时怎么办”改成“自动”，再做一次，会更像。")
    if result.get("range_note"):
        print(result["range_note"], flush=True)
    backing_mode = job.get("backing")
    solo_spans = []
    if backing_mode == "original" and refined.get("backing"):
        # A line the "backing" stem sings while the lead rests (an outro, an answer phrase) would
        # stay in the original voice; the stem is converted only if such phrases exist.
        import vocal_mix

        solo_spans, _ = vocal_mix.solo_backing_spans(refined["lead"], refined["backing"])
        if solo_spans:
            print(f"和声轨里有 {sum(b - a for a, b in solo_spans):.1f} 秒是主唱不在时单独唱的句子，这些句子也换成你的声音。", flush=True)
    convert_backing = bool(refined.get("backing")) and (backing_mode == "convert" or bool(solo_spans))
    # Read by stage_fraction(): how many conversion passes this stage will make.
    print(f"本步共 {2 if convert_backing else 1} 遍转换", flush=True)
    reports = {"lead": engine.convert(
        lead_input, str(output), pitch=lead_pitch,
        index_rate=job["index_rate"], protect=0.33, progress=progress("主唱"),
    )}
    result["converted"] = _output_audio(output, run)
    if convert_backing:
        backing_output = run / "converted_backing.wav"
        reports["backing"] = engine.convert(
            refined["backing"], str(backing_output), pitch=lead_pitch,
            index_rate=job["index_rate"], protect=0.33, progress=progress("和声"),
        )
        result["converted_backing"] = _output_audio(backing_output, run)
    _write_json(run / "convert_report.json", reports)
    return result


def _plain_mix(job: dict, run: Path) -> dict:
    """Sum two user-supplied files (the re-mix tool): no treatment, just gains and a limiter."""
    ffmpeg = APPLIO / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"缺少 FFmpeg：{ffmpeg}")
    voice_gain = _number(job["vocal_gain"], -24, 12, "人声音量")
    music_gain = _number(job["instrumental_gain"], -24, 12, "伴奏音量")
    # Put accompaniment first: duration=first preserves the complete song length.
    filters = (
        f"[0:a]aresample=44100,aformat=channel_layouts=stereo,volume={music_gain:g}dB[m];"
        f"[1:a]aresample=44100,aformat=channel_layouts=stereo,volume={voice_gain:g}dB[v];"
        "[m][v]amix=inputs=2:duration=first:dropout_transition=0,volume=2,"
        "alimiter=limit=0.95:level=false[out]"
    )
    output = run / "cover.wav"
    completed = subprocess.run([
        str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-i", str(_audio(job["instrumental"])),
        "-i", str(_audio(job["converted"])), "-filter_complex", filters, "-map", "[out]",
        "-c:a", "pcm_s24le", str(output),
    ], cwd=str(APPLIO), env=_environment(), capture_output=True, text=True,
       **_process_options())
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-1600:]
        raise RuntimeError(f"FFmpeg 合成失败：{detail or completed.returncode}")
    return {"mix": _output_audio(output, run)}


def _mix(job: dict, run: Path) -> dict:
    if "converted" in job:
        return _plain_mix(job, run)
    import vocal_mix

    separated = _read_json(run / "separate_result.json")
    refined = _stage_result(run, "refine")
    converted = _read_json(run / "convert_result.json")
    if job.get("duet_mode", "off") != "off":
        from duet_mix import mix_duet

        stems = _read_json(run / "duet_result.json")
        report = mix_duet({**separated, **stems, **converted}, job["duet_mode"], run,
                          vocal_gain_db=job["vocal_gain"], music_gain_db=job["instrumental_gain"],
                          clarity=job["clarity"], dynamics=0.6, reverb=job["reverb"])
        _write_json(run / "mix_report.json", report)
        return {"mix": _output_audio(run / "cover.wav", run), "converted": _output_audio(run / "my_vocals.wav", run)}
    backing_mode = job.get("backing", "original")
    output, vocal = run / "cover.wav", run / "my_vocals.wav"
    report = vocal_mix.mix_cover(
        instrumental=separated["instrumental"], converted=converted["converted"],
        reference_dry=refined.get("lead_dry") or separated["vocals"],
        reference_wet=refined.get("lead_reverb"),
        backing=refined.get("backing") if backing_mode != "remove" else None,
        backing_converted=converted.get("converted_backing") if backing_mode == "convert" else None,
        backing_solo=converted.get("converted_backing") if backing_mode == "original" else None,
        output_mix=output, output_vocal=vocal,
        vocal_gain_db=_number(job["vocal_gain"], -24, 12, "人声音量"),
        music_gain_db=_number(job["instrumental_gain"], -24, 12, "伴奏音量"),
        clarity=float(job.get("clarity", 0.75)), reverb=float(job.get("reverb", 1.0)),
    )
    _write_json(run / "mix_report.json", report)
    result = {"mix": _output_audio(output, run), "converted": _output_audio(vocal, run),
              "raw_converted": converted["converted"]}
    solo = report.get("backing_solo") or {}
    notes = []
    if solo.get("replaced_seconds", 0) >= 0.5:
        notes.append(f"和声轨里有 {solo['replaced_seconds']:.0f} 秒是主唱不在时单独唱的句子，已换成你的声音。")
    if solo.get("left_exposed_seconds", 0) >= 8:
        notes.append(f"另有约 {solo['left_exposed_seconds']:.0f} 秒是原唱和声与主唱交替出现的段落，保留了原唱的声音；"
                     "如果介意，在“更多设置”里把“和声”改成“和声也换成我的声音”，再做一次，对比着听哪个更好。")
    if notes:
        result["note"] = "".join(notes)
        print(result["note"], flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="AICover isolated audio worker")
    parser.add_argument("stage", choices=["separate", "refine", "duet", "convert", "mix"])
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job_path = args.job.resolve()
    run = job_path.parent
    if not _inside(run, OUTPUT) or not run.name.startswith("run_") or job_path.name != "job.json":
        parser.error("Job must be an AICover/output/run_*/job.json file")
    try:
        os.chdir(APPLIO)
        sys.path.insert(0, str(APPLIO))
        os.environ.update(_environment())
        job = _read_json(job_path)
        stages = {"separate": _separate, "refine": _refine, "duet": _duet, "convert": _convert, "mix": _mix}
        result = stages[args.stage](job, run)
        _write_json(run / f"{args.stage}_result.json", result)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _write_json(run / f"{args.stage}_error.json", {"error": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
