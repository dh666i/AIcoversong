"""User-triggered voice training; importing this module never loads an ML model.

Public UI API: ``train_voice(terms_accepted)`` yields progress snapshots (dicts,
see ``_snapshot``); ``cancel_training()`` cancels only the task started by this
module. All worker output stays in a unique task log folder, and original
recordings are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "Applio"
RECORDINGS = ROOT / "recordings" / "MyVoice"
MODELS = APPLIO / "logs"
RUNS = ROOT / "logs" / "training"
PYTHON = ROOT / "env" / "Scripts" / "python.exe"
MIN_SECONDS = 5 * 60
RECOMMENDED_SECONDS = 10 * 60
EPOCHS = 200
SAMPLE_RATE = 40000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
STEPS = ["整理录音", "学习音高和发音", "训练你的音色", "制作音色索引"]
# Share of the total time (measured: 23.5 minutes of recordings took 1 h 47 min, of which
# the third step was 1 h 46 min).
STEP_WEIGHTS = [0.006, 0.006, 0.985, 0.003]
_TASK_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_ACTIVE = None


def _snapshot(step: int, fraction: float, step_seconds: float, started: float, *, state: str = "running",
              message: str = "", detail: str = "") -> dict:
    """One progress snapshot for the page: which step, how far, how long, and one plain sentence."""
    fraction = max(0.0, min(1.0, fraction))
    done = sum(STEP_WEIGHTS[:step]) + (STEP_WEIGHTS[step] * fraction if step < len(STEPS) else 0.0)
    return {"state": state, "steps": STEPS, "step": step, "step_fraction": round(fraction, 4),
            "step_seconds": round(step_seconds, 1), "overall": round(min(1.0, done / sum(STEP_WEIGHTS)), 4),
            "elapsed": round(time.monotonic() - started, 1), "message": message, "detail": detail}


@dataclass
class _Task:
    cancel: threading.Event = field(default_factory=threading.Event)
    stop_lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen | None = None
    run: Path | None = None
    model: Path | None = None
    started: float = field(default_factory=time.monotonic)
    stage: str = "准备录音"
    step: int = 0
    outcome: str = "running"


class _Cancelled(Exception):
    pass


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours}小时" if hours else "") + f"{minutes}分{seconds:02d}秒"


def _check_cancel(task: _Task) -> None:
    if task.cancel.is_set():
        raise _Cancelled()


def _recordings() -> tuple[list[dict], float, int]:
    import soundfile as sf

    if not RECORDINGS.is_dir():
        raise ValueError("还没有录音。请先按页面上的稿子录几段，并点击保存。")
    accepted = []
    skipped = 0
    base = RECORDINGS.resolve()
    for path in sorted(RECORDINGS.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if not path.resolve().is_relative_to(base):
            skipped += 1
            continue
        try:
            info = sf.info(str(path))
            seconds = float(info.frames) / info.samplerate
            if info.frames <= 0 or info.samplerate <= 0 or info.channels not in (1, 2):
                raise ValueError("Invalid audio metadata")
            if not math.isfinite(seconds) or seconds < 1:
                raise ValueError("Recording is too short")
            accepted.append({"source": path, "seconds": seconds,
                             "frames": info.frames, "sample_rate": info.samplerate,
                             "channels": info.channels})
        except (OSError, RuntimeError, ValueError, ZeroDivisionError):
            skipped += 1
    seconds = sum(item["seconds"] for item in accepted)
    if seconds < MIN_SECONDS:
        raise ValueError(
            f"目前有 {_duration(seconds)}可读取的录音，还差 {_duration(MIN_SECONDS - seconds)}达到 5 分钟。"
            "请继续录音并保存，推荐累计 10 分钟后再训练。"
            "录音时请实际朗读或清唱；长时间停顿可能在整理录音时被去掉。"
        )
    return accepted, seconds, skipped


def _prepare_task(task: _Task, sources: list[dict], seconds: float, skipped: int) -> None:
    required = [PYTHON,
                APPLIO / "rvc/train/preprocess/preprocess.py",
                APPLIO / "rvc/train/extract/extract.py",
                APPLIO / "rvc/train/train.py",
                APPLIO / "rvc/train/process/extract_index.py",
                APPLIO / "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
                APPLIO / "rvc/models/pretraineds/hifi-gan/f0D40k.pth",
                APPLIO / "rvc/models/predictors/rmvpe.pt",
                APPLIO / "rvc/models/embedders/contentvec/config.json",
                APPLIO / "rvc/models/embedders/contentvec/pytorch_model.bin"]
    if any(not item.is_file() or item.stat().st_size == 0 for item in required):
        raise ValueError("训练工具或基础模型缺失，请先修复安装；你的录音会保留。")
    _check_cancel(task)
    RUNS.mkdir(parents=True, exist_ok=True)
    task.run = RUNS / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    task.run.mkdir(exist_ok=False)
    snapshot = task.run / "recordings"
    snapshot.mkdir()
    copied = []
    for number, item in enumerate(sources, 1):
        _check_cancel(task)
        destination = snapshot / f"recording_{number:04d}{item['source'].suffix.lower()}"
        # A real copy keeps this run stable even when the user re-records later.
        shutil.copy2(item["source"], destination)
        copied.append({**item, "source": str(item["source"]), "snapshot": str(destination)})
    _check_cancel(task)
    MODELS.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        model = MODELS / ("MyVoice" if number == 1 else f"MyVoice{number}")
        try:
            model.mkdir(exist_ok=False)
            task.model = model
            break
        except FileExistsError:
            number += 1
    _write_json(task.run / "job.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": model.name, "model_dir": str(model), "dataset": str(snapshot),
        "recordings": copied, "readable_audio_seconds": seconds, "skipped_files": skipped,
        "settings": {"architecture": "RVC v2", "sample_rate": SAMPLE_RATE,
                     "vocoder": "HiFi-GAN", "embedder": "contentvec", "pitch": "rmvpe",
                     "epochs": EPOCHS, "batch_size": 4, "gpu": "0"},
        "terms_accepted": True,
    })


def _environment(log_path: Path) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "APPLIO_CONFIG_FILE": str(APPLIO / "assets/config.json"),
                "APPLIO_LOGS_DIR": str(MODELS), "APPLIO_ROOT": str(APPLIO),
                "APPLIO_CODE_ROOT": str(APPLIO),
                # All grandchildren append to this task log, not the UI terminal.
                "APPLIO_TERMINAL_LOG": str(log_path),
                "HF_HOME": str(ROOT / "cache/huggingface"),
                "TORCH_HOME": str(ROOT / "cache/torch"),
                "NUMBA_CACHE_DIR": str(ROOT / "cache/numba"),
                "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "*"})
    env["PATH"] = str(APPLIO) + os.pathsep + str(PYTHON.parent) + os.pathsep + env.get("PATH", "")
    return env


def _commands(task: _Task) -> list[tuple[str, list[str]]]:
    model = task.model
    cpu_count = max(1, min(os.cpu_count() or 1, 8))
    # On Windows every preprocess worker imports torch and maps the CUDA DLLs.
    # Eight simultaneous imports can exhaust the page file before any audio is
    # processed (WinError 1455), so keep this I/O/CPU stage conservative.
    preprocess_workers = 1 if os.name == "nt" else cpu_count
    prefix = [str(PYTHON), "-u"]
    return [
        ("整理录音", prefix + [str(APPLIO / "rvc/train/preprocess/preprocess.py"),
         str(model), str(task.run / "recordings"), str(SAMPLE_RATE), str(preprocess_workers),
         "Automatic", "False", "3.0", "0.3", "post"]),
        ("学习音高和发音特征", prefix + [str(APPLIO / "rvc/train/extract/extract.py"),
         str(model), "rmvpe", str(cpu_count), "0", str(SAMPLE_RATE), "contentvec", "None", "2"]),
        ("训练你的音色", prefix + [str(APPLIO / "rvc/train/train.py"), model.name,
         "10", str(EPOCHS), str(APPLIO / "rvc/models/pretraineds/hifi-gan/f0G40k.pth"),
         str(APPLIO / "rvc/models/pretraineds/hifi-gan/f0D40k.pth"), "0", "4", str(SAMPLE_RATE),
         "True", "True", "False", "False", "HiFi-GAN", "False"]),
        ("制作音色索引", prefix + [str(APPLIO / "rvc/train/process/extract_index.py"),
         str(model), "Auto"]),
    ]


def _process_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}


def _terminate_process(task: _Task, process: subprocess.Popen) -> None:
    """Cancel a process tree once, without relying on stale saved PID files."""
    with task.stop_lock:
        if process.poll() is not None:
            return
        descendants = []
        try:
            import psutil
            descendants = psutil.Process(process.pid).children(recursive=True)
        except (ImportError, OSError):
            pass
        except Exception:
            # The parent can finish between poll() and the process snapshot.
            if process.poll() is not None:
                return
        if os.name == "nt":
            try:
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=10, **_process_options())
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # psutil snapshots descendants while the task parent is still alive.
            try:
                import psutil
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                psutil.wait_procs(children + [parent], timeout=5)
            except Exception:
                if process.poll() is None:
                    process.kill()
            process.wait(timeout=5)
        # A child can briefly outlive its parent. Saved psutil objects check
        # creation time before kill(), so they do not target a reused PID.
        for child in descendants:
            try:
                if child.is_running():
                    child.kill()
            except Exception:
                pass
        if descendants:
            psutil.wait_procs(descendants, timeout=5)


def cancel_training() -> str:
    with _STATE_LOCK:
        task = _ACTIVE
        if task is None:
            return "当前没有正在运行的音色训练。"
        task.cancel.set()
        process = task.process
    if process is not None:
        try:
            _terminate_process(task, process)
        except Exception:
            # The generator also retries cleanup in its finally block.
            return "已请求停止，正在清理训练进程；已保存的录音会保留。"
    return "已请求停止本次训练，原始录音和已生成的文件都已保留。"


def _log_tail(log_path: Path, size: int = 32768) -> str:
    if not log_path.exists():
        return ""
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace")


# A progress bar of the training tools: " 48%|████▊     | 68/143 [01:31<00:54, ...". Bars with a caption in
# front ("Loading weights: 100%|...") belong to something else and are not counted.
_BAR_LINE = re.compile(r"(?m)^\s*\d+%\|[^|\n]*\|\s*(\d+)/(\d+) \[")


def _last_bar(text: str) -> float:
    """Position of the newest progress bar in a piece of log, 0..1."""
    bars = _BAR_LINE.findall(text)
    if not bars or not int(bars[-1][1]):
        return 0.0
    return min(1.0, int(bars[-1][0]) / int(bars[-1][1]))


def _stage_progress(number: int, log_path: Path) -> tuple[float, str]:
    """(fraction 0..1, short detail) of one training stage, read from its log.

    Checked against the logs of a real training by logs/audio-audit-20261116/train_progress_check.py.
    """
    if number == 3:
        text = _log_tail(log_path, 65536).replace("\r", "\n")
        # "epoch=12 | step=1716 | time=..." closes an epoch; "epoch=13 | step=1730 | batch=14/143" is inside one.
        # Only whole lines count: a line caught half written reads "batch=70/1".
        closed = re.findall(r"\bepoch=(\d+)\s*\|\s*step=\d+\s*\|\s*time=", text)
        inside = re.findall(r"\bepoch=(\d+)\s*\|\s*step=\d+\s*\|\s*batch=(\d+)/(\d+)[ \t]*\n", text)
        current = int(inside[-1][0]) if inside else 0
        done = max(int(closed[-1]) if closed else 0, current - 1, 0)
        part = 0.0
        if inside and current == done + 1 and int(inside[-1][2]):
            part = min(1.0, int(inside[-1][1]) / int(inside[-1][2]))
        done = min(done, EPOCHS)
        return min(1.0, (done + part) / EPOCHS), f"第 {min(done + 1, EPOCHS)}/{EPOCHS} 轮"
    text = _log_tail(log_path, 4_000_000).replace("\r", "\n")   # these logs are a few kilobytes: read all of it
    if number == 2:   # two passes over the slices: pitch, then pronunciation features
        second = text.rfind("Starting embedding extraction")
        if second < 0:
            return 0.5 * _last_bar(text), ""
        return 0.5 + 0.5 * _last_bar(text[second:]), ""
    return _last_bar(text), ""


def _stage(task: _Task, number: int, label: str, command: list[str]):
    task.stage = label
    task.step = number - 1
    log_path = task.run / f"{number:02d}.log"
    process = None
    started = time.monotonic()
    best = 0.0
    with log_path.open("ab") as log:
        log.write((f"Stage {number}: {label}\nCommand: {json.dumps(command, ensure_ascii=False)}\n").encode("utf-8"))
        log.flush()
        try:
            with _STATE_LOCK:
                _check_cancel(task)
                process = subprocess.Popen(command, cwd=str(APPLIO), env=_environment(log_path),
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           **_process_options())
                task.process = process
            while process.poll() is None:
                _check_cancel(task)
                fraction, detail = _stage_progress(number, log_path)
                best = max(best, fraction)   # whatever the log looks like for a moment, the page never steps back
                yield _snapshot(number - 1, best, time.monotonic() - started, task.started,
                                message=STEPS[number - 1], detail=detail)
                task.cancel.wait(1)
            _check_cancel(task)
            if process.returncode != 0:
                raise RuntimeError(f"{label}未能完成（退出码 {process.returncode}）。")
        finally:
            try:
                if process is not None and process.poll() is None:
                    _terminate_process(task, process)
            finally:
                with _STATE_LOCK:
                    if task.process is process:
                        task.process = None


def _check_slices(task: _Task) -> None:
    import soundfile as sf

    files = list((task.model / "sliced_audios").glob("*.wav"))
    if not files:
        raise RuntimeError("整理录音后没有得到可训练的人声片段，请检查录音内容和音量。")
    readable = 0
    seconds = 0.0
    for path in files:
        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate == SAMPLE_RATE:
            readable += 1
            seconds += info.frames / info.samplerate
    if readable != len(files):
        raise RuntimeError("整理后的部分录音无法读取，请检查本次任务日志。")
    if seconds < MIN_SECONDS:
        raise RuntimeError(
            f"整理后只剩 {_duration(seconds)}可训练片段，不足 5 分钟。"
            "请继续朗读或清唱并保存，再重新训练。"
        )


def _check_features(task: _Task) -> None:
    import numpy as np

    config = task.model / "config.json"
    filelist = task.model / "filelist.txt"
    if not config.is_file() or not filelist.is_file():
        raise RuntimeError("音色特征不完整，尚未开始训练；请查看任务日志。")
    json.loads(config.read_text(encoding="utf-8"))
    rows = [line.split("|") for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
    own_rows = []
    for fields in rows:
        if len(fields) != 5:
            raise RuntimeError("音色特征列表格式不完整，尚未开始训练。")
        paths = [Path(value) for value in fields[:4]]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            raise RuntimeError("音色特征文件缺失，尚未开始训练。")
        if paths[0].resolve().is_relative_to(task.model.resolve()):
            own_rows.append(paths)
    if not own_rows:
        raise RuntimeError("没有从你的录音中提取到可训练的特征，请检查录音后重试。")
    slices = {path.resolve() for path in (task.model / "sliced_audios").glob("*.wav")}
    if {paths[0].resolve() for paths in own_rows} != slices:
        raise RuntimeError("部分录音片段未能提取完整特征，尚未开始训练；请查看任务日志。")
    for paths in own_rows:
        features = np.load(paths[1], mmap_mode="r", allow_pickle=False)
        pitch = np.load(paths[2], mmap_mode="r", allow_pickle=False)
        voiced = np.load(paths[3], mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 768:
            raise RuntimeError("音色特征维度不正确，尚未开始训练。")
        if pitch.size == 0 or voiced.size == 0:
            raise RuntimeError("音高特征为空，尚未开始训练。")
        if not (np.isfinite(features).all() and np.isfinite(pitch).all() and np.isfinite(voiced).all()):
            raise RuntimeError("音色或音高特征包含无效数值，尚未开始训练。")


def _exported_model(task: _Task) -> Path:
    pattern = re.compile(re.escape(task.model.name) + rf"_{EPOCHS}e_\d+s\.pth$")
    candidates = [path for path in task.model.glob("*.pth")
                  if not path.name.lower().startswith(("g_", "d_")) and pattern.fullmatch(path.name)
                  and path.is_file() and path.stat().st_size > 1024]
    if not candidates:
        raise RuntimeError("训练进程已结束，但没有生成完整的个人音色模型；请查看任务日志。")
    model = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    try:
        with zipfile.ZipFile(model) as archive:
            names = archive.namelist()
            if not any(name.endswith("/data.pkl") for name in names) or archive.testzip() is not None:
                raise ValueError("Incomplete exported model archive")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise RuntimeError("导出的音色模型不完整，已保留本次训练文件。") from error
    return model


def _check_index(task: _Task) -> Path:
    import faiss

    path = task.model / f"{task.model.name}.index"
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("音色模型已生成，但索引未生成；请查看任务日志。")
    index = faiss.read_index(str(path))
    if index.ntotal <= 0 or index.d != 768:
        raise RuntimeError("音色索引不完整，已保留模型和本次训练文件。")
    return path


def train_voice(terms_accepted):
    """Run only after the user presses train; yield statuses without model calls at import."""
    global _ACTIVE
    began = time.monotonic()
    if terms_accepted is not True:
        yield _snapshot(0, 0, 0, began, state="failed", message="请先勾选“这些录音都是我自己的声音”那一项，再开始训练。")
        return
    if not _TASK_LOCK.acquire(blocking=False):
        yield _snapshot(0, 0, 0, began, state="failed", message="已经有一个训练在进行，请等它完成，或先点“停止训练”。")
        return
    task = _Task()
    with _STATE_LOCK:
        _ACTIVE = task
    try:
        yield _snapshot(0, 0, 0, task.started, message="正在检查已保存的录音")
        sources, seconds, skipped = _recordings()
        _check_cancel(task)
        yield _snapshot(0, 0, 0, task.started, message="整理录音",
                        detail=f"{len(sources)} 段录音，共 {_duration(seconds)}")
        _prepare_task(task, sources, seconds, skipped)
        for number, (label, command) in enumerate(_commands(task), 1):
            yield from _stage(task, number, label, command)
            _check_cancel(task)
            if number == 1:
                _check_slices(task)
            elif number == 2:
                _check_features(task)
            elif number == 3:
                _exported_model(task)
            elif number == 4:
                _check_index(task)
        _check_cancel(task)
        model = _exported_model(task)
        index = _check_index(task)
        task.outcome = "completed"
        _write_json(task.run / "result.json", {
            "status": "completed", "model_name": task.model.name,
            "model": str(model), "index": str(index),
            "elapsed_seconds": round(time.monotonic() - task.started),
        })
        yield _snapshot(len(STEPS), 1.0, 0, task.started, state="done",
                        message=f"训练完成，新音色“{task.model.name}”已经可以用了。",
                        detail=f"共用 {_duration(time.monotonic() - task.started)}。回到“翻唱”页就会自动选中它。")
    except _Cancelled:
        task.outcome = "cancelled"
        yield _snapshot(task.step, 0, 0, task.started, state="stopped", message="训练已停止。",
                        detail="你的录音都还在，可以稍后重新训练。")
    except Exception as error:
        task.outcome = "failed"
        if task.run is not None:
            (task.run / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else "这次训练没有完成。"
        location = f"详细记录在：{task.run}。" if task.run else ""
        yield _snapshot(task.step, 0, 0, task.started, state="failed", message=message,
                        detail=f"{location}你的录音没有改动。")
    finally:
        task.cancel.set()
        try:
            if task.process is not None:
                _terminate_process(task, task.process)
            if task.outcome == "running":
                task.outcome = "cancelled"
            if task.run is not None and task.outcome != "completed":
                _write_json(task.run / "result.json", {"status": task.outcome,
                            "stage": task.stage, "model_name": task.model.name if task.model else None,
                            "elapsed_seconds": round(time.monotonic() - task.started)})
        finally:
            with _STATE_LOCK:
                if _ACTIVE is task:
                    _ACTIVE = None
            _TASK_LOCK.release()
