"""The whole web page: make a cover, record and train a voice, look at earlier covers.

Three tabs, all in Chinese, built for someone who is not a developer:

  翻唱        upload a song, press one button, watch four steps, listen
  我的声音     read or sing the prompts into the microphone, then train
  历史        earlier covers, newest first

Importing this module loads no audio model.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import quote

import gradio as gr

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))
import cover_pipeline as pipeline
import voice_recording as recorder
import voice_training as trainer

BACKING_CHOICES = [("保留原唱的和声（推荐）", "original"), ("和声也换成我的声音", "convert"), ("不要和声", "remove")]
RANGE_CHOICES = [("自动（推荐）", "auto"), ("保持原来的音高", "keep"), ("降一个八度", "down")]
IDLE = {"state": "idle"}


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

def _voice_label(relative: str) -> str:
    """'MyVoice4\\MyVoice4_200e_28600s.pth' -> '我的声音 · 第 4 版'."""
    folder = Path(relative).parts[0]
    digits = "".join(ch for ch in folder if ch.isdigit())
    name = f"我的声音 · 第 {digits or 1} 版" if folder.lower().startswith("myvoice") else folder
    stem = Path(relative).stem
    rounds = next((part[:-1] for part in stem.split("_") if part.endswith("e") and part[:-1].isdigit()), "")
    return name if not rounds or rounds == "200" else f"{name}（练到第 {rounds} 轮）"


def _voices() -> list[tuple[str, str]]:
    """(label, relative path), newest training first; only the final model of each training."""
    by_folder: dict[str, str] = {}
    for relative in pipeline.voice_models():
        folder = Path(relative).parts[0]
        path = pipeline.MODEL_ROOT / relative
        if folder not in by_folder or path.stat().st_mtime > (pipeline.MODEL_ROOT / by_folder[folder]).stat().st_mtime:
            by_folder[folder] = relative
    ordered = sorted(by_folder.values(), key=lambda r: (pipeline.MODEL_ROOT / r).stat().st_mtime, reverse=True)
    return [(_voice_label(relative), relative) for relative in ordered]


def _index_for(model: str | None) -> str:
    choices = pipeline.voice_indices(model) if model else [pipeline.NO_INDEX]
    return choices[1] if len(choices) > 1 else pipeline.NO_INDEX


def _refresh_voices():
    voices = _voices()
    return gr.update(choices=voices, value=voices[0][1] if voices else None)


# --------------------------------------------------------------------------
# Progress component (shared by covers and training)
# --------------------------------------------------------------------------

PROGRESS_TEMPLATE = '<div class="stage" data-state="idle" aria-live="polite"></div>'

PROGRESS_JS = r"""
const root = element.querySelector('.stage');
const esc = (text) => String(text ?? '').replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const clock = (seconds) => {
  seconds = Math.max(0, Math.round(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
  if (h) return `${h} 小时 ${String(m).padStart(2, '0')} 分`;
  return m ? `${m} 分 ${String(s).padStart(2, '0')} 秒` : `${s} 秒`;
};
const rough = (seconds) => {
  if (seconds < 20) return '马上就好';
  if (seconds < 90) return '大约还要 1 分钟';
  if (seconds < 3600) return `大约还要 ${Math.round(seconds / 60)} 分钟`;
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60 / 5) * 5;
  return `大约还要 ${h} 小时${m ? ' ' + m + ' 分' : ''}`;
};
let shown = null;          // last snapshot from the server
let received = 0;          // performance.now() when it arrived
let display = 0;           // overall fraction currently drawn (eases towards the target)
let rail = 0;              // how much of the line between the dots is filled
let frame = 0;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function structure(data) {
  const steps = (data.steps || []).map((label, i) =>
    `<li data-i="${i}"><span class="dot"></span><span class="name">${esc(label)}</span></li>`).join('');
  root.innerHTML = `
    ${data.title ? `<h2 class="what">${esc(data.title)}</h2>` : ''}
    <ol class="steps" style="--n:${(data.steps || []).length}">${steps}<span class="rail"><i></i></span></ol>
    <p class="now"><b></b><span class="pct"></span></p>
    <p class="time"></p>
    <p class="hint"></p>
    <div class="notes"></div>`;
}

function paint() {
  frame = 0;
  if (!shown) return;
  const data = shown;
  const running = data.state === 'running';
  const since = running ? (performance.now() - received) / 1000 : 0;
  const elapsed = (data.elapsed || 0) + since;
  // glide between server updates so the line never jumps, and never passes what the server reported
  const target = data.state === 'done' ? 1 : Math.min(0.995, data.overall || 0);
  display = reducedMotion.matches ? target : display + (target - display) * 0.12;
  if (Math.abs(target - display) < 0.0005 || !running) display = target;
  // the line follows the steps: dot i is where step i starts, the stretch after it is that step
  const count = (data.steps || []).length;
  const along = count > 1 ? Math.min(1, ((data.step || 0) + (data.step_fraction || 0)) / (count - 1)) : (data.step_fraction || 0);
  const railTarget = data.state === 'done' ? 1 : running ? along : rail;
  rail = reducedMotion.matches ? railTarget : rail + (railTarget - rail) * 0.12;
  if (Math.abs(railTarget - rail) < 0.0005 || !running) rail = railTarget;
  root.querySelector('.rail i').style.transform = `scaleX(${rail})`;
  root.querySelectorAll('.steps li').forEach((li) => {
    const i = Number(li.dataset.i);
    li.dataset.s = data.state === 'done' || i < data.step ? 'done' : i === data.step ? (data.state === 'failed' ? 'failed' : data.state === 'stopped' ? 'stopped' : 'now') : 'todo';
  });
  const now = root.querySelector('.now b'), pct = root.querySelector('.pct'), time = root.querySelector('.time');
  root.querySelector('.hint').textContent = running ? (data.hint || '') : '';
  if (running) {
    const label = (data.steps || [])[data.step] || data.message || '';
    now.textContent = '正在' + label + (data.detail ? `（${data.detail}）` : '');
    pct.textContent = Math.round(display * 100) + '%';
    const reached = Math.min(0.995, data.overall || 0);
    const left = reached > 0.03 && elapsed > 8 ? elapsed * (1 - reached) / reached : null;
    time.textContent = `已用 ${clock(elapsed)}` + (left === null ? '' : `，${rough(left)}`);
    frame = requestAnimationFrame(paint);
  } else if (data.state === 'done') {
    now.textContent = data.message || '完成';
    pct.textContent = '';
    time.textContent = `共用 ${clock(data.elapsed || 0)}` + (data.detail ? `。${data.detail}` : '');
  } else {
    now.textContent = data.message || data.error || (data.state === 'stopped' ? '已停止' : '没有完成');
    pct.textContent = '';
    time.textContent = data.detail || '';
  }
  const notes = (data.notes || []).map((text) => `<p>${esc(text)}</p>`).join('');
  const box = root.querySelector('.notes');
  if (box.dataset.html !== notes) { box.innerHTML = notes; box.dataset.html = notes; }
}

function update() {
  let data = props.value;
  if (typeof data === 'string') { try { data = JSON.parse(data); } catch (e) { data = null; } }
  if (!data || !data.state || data.state === 'idle') {
    shown = null; cancelAnimationFrame(frame); frame = 0;
    root.dataset.state = 'idle'; root.innerHTML = ''; display = 0; rail = 0;
    return;
  }
  const newRun = shown && (shown.run !== data.run || (shown.state !== 'running' && data.state === 'running'));
  const fresh = !shown || newRun || (shown.steps || []).join('|') !== (data.steps || []).join('|') || shown.title !== data.title || !root.firstChild;
  if (fresh) { structure(data); if (!shown || newRun || data.overall < display) { display = 0; rail = 0; } }
  shown = data; received = performance.now();
  root.dataset.state = data.state;
  // a finished, failed or stopped state is drawn at once: a tab in the background gets no animation frames
  if (data.state !== 'running') { cancelAnimationFrame(frame); frame = 0; }
  if (!frame) paint();
}
watch('value', update);
update();
"""

PROGRESS_CSS = """
.stage { display: block; }
.stage[data-state="idle"] { display: none; }
.what { margin: 0 0 30px; font-family: var(--sans); font-weight: 650; font-size: 22px; line-height: 1.45; color: var(--ink); overflow-wrap: anywhere; }
.hint { margin: 2px 0 0; font-size: 14px; color: var(--ink-3); }
.hint:empty { display: none; }
.steps { position: relative; display: grid; grid-template-columns: repeat(var(--n, 4), 1fr); margin: 0; padding: 0 0 2px; list-style: none; }
.steps li { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center; }
.steps .dot { width: 14px; height: 14px; border-radius: 50%; background: var(--paper-3); box-shadow: 0 0 0 5px var(--paper); transition: background .4s, transform .4s; }
.steps .name { font-size: 14px; line-height: 1.35; color: var(--ink-3); max-width: 9em; transition: color .4s; }
.steps li[data-s="done"] .dot { background: var(--done); }
.steps li[data-s="done"] .name { color: var(--ink-2); }
.steps li[data-s="now"] .dot { background: var(--lamp); transform: scale(1.25); animation: breathe 1.8s ease-in-out infinite; }
.steps li[data-s="now"] .name { color: var(--ink); font-weight: 600; }
.steps li[data-s="failed"] .dot { background: var(--bad); }
.steps li[data-s="failed"] .name { color: var(--bad); }
.steps li[data-s="stopped"] .dot { background: var(--ink-3); }
.rail { position: absolute; left: calc(50% / var(--n, 4)); right: calc(50% / var(--n, 4)); top: 6px; height: 2px; background: var(--paper-3); border-radius: 2px; overflow: hidden; }
.rail i { display: block; height: 100%; background: linear-gradient(90deg, var(--done), var(--lamp)); transform-origin: left; transform: scaleX(0); }
.stage[data-state="done"] .rail i { background: var(--done); }
.now { margin: 26px 0 0; display: flex; align-items: baseline; justify-content: space-between; gap: 16px; font-size: 20px; line-height: 1.4; color: var(--ink); }
.now b { font-weight: 600; }
.now .pct { font-variant-numeric: tabular-nums; color: var(--lamp); font-size: 20px; }
.stage[data-state="failed"] .now b { color: var(--bad); font-size: 17px; font-weight: 500; white-space: pre-line; }
.time { margin: 6px 0 0; font-size: 14px; color: var(--ink-3); font-variant-numeric: tabular-nums; min-height: 1.5em; }
.notes p { margin: 14px 0 0; padding: 12px 14px; border-left: 3px solid var(--lamp); background: var(--paper-2); color: var(--ink-2); font-size: 14.5px; line-height: 1.7; border-radius: 0 8px 8px 0; }
@keyframes breathe { 0%, 100% { box-shadow: 0 0 0 5px var(--paper), 0 0 0 5px transparent; } 50% { box-shadow: 0 0 0 5px var(--paper), 0 0 0 10px color-mix(in srgb, var(--lamp) 16%, transparent); } }
@media (prefers-reduced-motion: reduce) { .steps li[data-s="now"] .dot { animation: none; } }
@media (max-width: 560px) { .steps .name { font-size: 12px; } .now { font-size: 17px; } .what { font-size: 20px; margin-bottom: 24px; } }
"""


def _progress_view(elem_id: str):
    return gr.HTML(value=IDLE, html_template=PROGRESS_TEMPLATE, css_template=PROGRESS_CSS, js_on_load=PROGRESS_JS,
                   elem_id=elem_id, elem_classes="stage-host", container=False, apply_default_css=False)


# --------------------------------------------------------------------------
# Players (results and history)
# --------------------------------------------------------------------------

def _file_url(path) -> str:
    """Address of a file under output/ as the page can play it (the server allows that folder)."""
    return "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:")


def _track(path, download_name: str, label: str = "", main: bool = False) -> str:
    """One player with its download link. The browser's own player: press play, click anywhere on the bar."""
    url = html.escape(_file_url(path), quote=True)
    name = f'<span class="track-name">{html.escape(label)}</span>' if label else ""
    return (f'<div class="track{" track-main" if main else ""}">{name}'
            f'<audio controls preload="metadata" controlslist="nodownload noplaybackrate" src="{url}"></audio>'
            f'<a class="save" href="{url}" download="{html.escape(download_name, quote=True)}">下载</a></div>')


def _result_html(item: dict, title: str) -> str:
    """What is shown when a cover is finished: the cover, and the separate stems behind a fold."""
    if not item.get("mix"):
        return ""
    parts = [("只有我的人声", item.get("converted"), "我的人声"), ("原唱的人声", item.get("vocals"), "原唱人声"),
             ("伴奏", item.get("instrumental"), "伴奏")]
    rows = "".join(_track(file, f"{title} {suffix}.wav", label) for label, file, suffix in parts if file)
    fold = f'<details class="parts"><summary>分开听</summary>{rows}</details>' if rows else ""
    return (f'<section class="result"><div class="result-heading"><span class="eyebrow">制作完成</span>'
            f'<h3>{html.escape(title)}</h3><p>试听你的作品，也可以单独下载人声与伴奏。</p></div>'
            f'{_track(item["mix"], f"{title} 翻唱.wav", main=True)}{fold}</section>')


# --------------------------------------------------------------------------
# Cover tab
# --------------------------------------------------------------------------

@dataclass
class _CoverTask:
    token: str
    title: str
    cancel: threading.Event = field(default_factory=threading.Event)
    item: dict = field(default_factory=dict)
    finished: bool = False
    updated: float = field(default_factory=time.monotonic)


# A short request owns a task; browser polling never owns the audio worker.
# The same lock also protects training, including requests from other tabs.
_GPU_WORK = threading.Lock()
_COVER_TASKS_LOCK = threading.Lock()
_COVER_TASKS: dict[str, _CoverTask] = {}


def _task_snapshot(token):
    with _COVER_TASKS_LOCK:
        task = _COVER_TASKS.get(token) if isinstance(token, str) else None
        if task is None:
            return None
        return task.title, dict(task.item), task.finished, task.cancel.is_set()


def _publish_cover(task, item, *, finished=False):
    with _COVER_TASKS_LOCK:
        task.item = item
        task.finished = finished
        task.updated = time.monotonic()


def _run_cover_task(task, arguments):
    """Only this thread advances/closes the generator and releases its GPU slot."""
    worker = None
    terminal = None
    try:
        worker = pipeline.run_cover(*arguments)
        while not task.cancel.is_set():
            try:
                item = next(worker)
            except StopIteration:
                break
            if task.cancel.is_set():
                break
            if item.get("progress", {}).get("state") in {"done", "failed", "stopped"}:
                terminal = item
                break
            _publish_cover(task, item)
    except Exception as exc:
        terminal = {"progress": {"state": "failed", "error": str(exc),
                                  "message": "翻唱未完成，请查看提示后重试。"}}
    finally:
        try:
            if worker is not None:
                worker.close()
        except Exception as exc:
            terminal = {"progress": {"state": "failed", "error": str(exc),
                                      "message": "任务清理未完成，请稍后重试。"}}
        finally:
            if task.cancel.is_set() and (terminal or {}).get("progress", {}).get("state") != "done":
                terminal = {"progress": {"state": "stopped", "message": "已停止，可以重新选择歌曲。"}}
            if terminal is None:
                terminal = {"progress": {"state": "failed", "message": "任务结束但未返回成品，请重试。"}}
            # Publish the terminal state only after generator cleanup has released
            # its subprocesses, and after a second request may take the GPU slot.
            _GPU_WORK.release()
            _publish_cover(task, terminal, finished=True)


def _cover_view(snapshot):
    """Progress/form/stage/result/again/start/stop/empty/timer updates."""
    # Gradio consumes update dictionaries in place during postprocessing.
    # Never share an update object between components or successive responses.
    def hidden():
        return gr.update(visible=False)

    def shown():
        return gr.update(visible=True)

    ready = gr.update(interactive=True, value="开始翻唱")
    if snapshot is None:
        return (dict(IDLE), gr.skip(), hidden(), gr.update(value="", visible=False), hidden(),
                ready, hidden(), shown(), gr.update(active=False))
    title, item, finished, cancelling = snapshot
    progress = {**item.get("progress", {}), "title": title, "run": item.get("run")}
    if not finished:
        progress["state"] = "running"
        progress["hint"] = "正在停止并释放资源，请稍等。" if cancelling else "可以切换页面或刷新；任务会继续，完成后可在作品库找到。"
        if cancelling:
            progress["detail"] = "正在停止"
        return (progress, gr.skip(), shown(), gr.update(value="", visible=False), hidden(),
                gr.update(interactive=False, value="正在停止…" if cancelling else "正在制作…"),
                gr.update(visible=True, interactive=not cancelling), hidden(), gr.update(active=True))
    state = progress.get("state")
    if state == "done":
        progress["message"] = "翻唱好了"
        return (progress, gr.skip(), shown(), gr.update(value=_result_html(item, title), visible=True),
                gr.update(visible=True, value="继续制作"), ready, hidden(), hidden(), gr.update(active=False))
    if state == "stopped":
        return (progress, gr.skip(), shown(), gr.update(value="", visible=False), hidden(),
                ready, hidden(), hidden(), gr.update(active=False))
    progress["message"] = progress.get("error") or progress.get("message") or "没有完成，请重试。"
    return (progress, gr.skip(), shown(), gr.update(value="", visible=False),
            gr.update(visible=True, value="继续制作"), ready, hidden(), hidden(), gr.update(active=False))


def _start_cover(song, model, pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate,
                 current_task=""):
    """Start a task immediately; /cover returns a task token, not a long stream."""
    if not song:
        raise gr.Error("先选一首歌。", title="还不能开始", print_exception=False)
    if not model:
        raise gr.Error("还没有训练过声音。先到“我的声音”里录音并训练一次。", title="还不能开始", print_exception=False)
    previous = _task_snapshot(current_task)
    if previous is not None and not previous[2]:
        view = _cover_view(previous)
        return (*view[:-1], current_task, view[-1])
    if not _GPU_WORK.acquire(blocking=False):
        raise gr.Error("已有翻唱或训练正在进行，请等它完成或先停止。", title="任务进行中", print_exception=False)
    task = None
    try:
        arguments = (song, model, _index_for(model), 0, index_rate, vocal_gain, music_gain, False,
                     refine, backing, clarity, reverb, pitch_mode)
        task = _CoverTask(uuid.uuid4().hex, _song_title(song))
        task.item = {"progress": {"state": "running", "steps": ["准备歌曲"], "step": 0,
                                  "overall": 0, "elapsed": 0, "detail": "任务已接收"}}
        with _COVER_TASKS_LOCK:
            # Keep recent refresh recovery without an unbounded per-song registry.
            expired = [key for key, value in _COVER_TASKS.items()
                       if value.finished and time.monotonic() - value.updated > 86400]
            for key in expired:
                del _COVER_TASKS[key]
            _COVER_TASKS[task.token] = task
        thread = threading.Thread(target=_run_cover_task, args=(task, arguments),
                                  name=f"cover-{task.token[:8]}", daemon=True)
        thread.start()
    except Exception:
        if task is not None:
            with _COVER_TASKS_LOCK:
                _COVER_TASKS.pop(task.token, None)
        _GPU_WORK.release()
        raise
    view = _cover_view(_task_snapshot(task.token))
    return (*view[:-1], task.token, view[-1])


def _poll_cover(current_task=""):
    return _cover_view(_task_snapshot(current_task))


def _stop_cover(current_task=""):
    with _COVER_TASKS_LOCK:
        task = _COVER_TASKS.get(current_task) if isinstance(current_task, str) else None
        if task is not None and not task.finished:
            task.cancel.set()
    return _poll_cover(current_task)


def _again(current_task=""):
    """Back to the form, with the song and the settings as they were."""
    snapshot = _task_snapshot(current_task)
    if snapshot is not None and not snapshot[2]:
        view = _cover_view(snapshot)
        return (*view[:-1], current_task, view[-1])
    view = _cover_view(None)
    return (*view[:-1], "", view[-1])


def _cover_tab():
    voices = _voices()
    refine_ready = pipeline.refine_available()
    with gr.Row(elem_id="cover-workspace", equal_height=False):
        with gr.Column(elem_id="cover-controls", elem_classes="studio-panel", scale=5, min_width=320):
            gr.HTML('<div class="panel-heading"><span class="eyebrow">01 / 创作设置</span>'
                    '<h2>从一首喜欢的歌开始</h2><p>选好歌曲和音色，剩下的交给 AIcoversong。</p></div>', container=False)
            with gr.Column(elem_id="cover-form") as form:
                song = gr.File(label="上传歌曲", file_types=["audio"], type="filepath", elem_id="song-drop",
                               height=170)
                with gr.Row(elem_id="voice-picker-row"):
                    model = gr.Dropdown(label="我的音色", choices=voices, value=voices[0][1] if voices else None,
                                        interactive=True, elem_id="voice-pick", scale=5,
                                        info=None if voices else "还没有音色？到“我的声音”录音并训练，完成后即可选择。")
                    refresh_voice = gr.Button("刷新音色", elem_id="refresh-voices", size="sm", scale=1, min_width=90)
                with gr.Accordion("声音设置", open=False, elem_id="more-settings"):
                    pitch_mode = gr.Radio(choices=RANGE_CHOICES, value="auto", label="音域适配",
                                          info="自动模式只在原唱明显超出你的录音音域时，将人声降低一个八度；伴奏保持原调。")
                    backing = gr.Radio(choices=BACKING_CHOICES, value="original", label="和声处理",
                                       info="保留原唱和声通常更自然，主唱仍会转换为你的声音。")
                    clarity = gr.Slider(0, 1, value=0.75, step=0.05, label="人声清晰度",
                                        info="歌词不清楚可调高；声音偏尖、偏薄可调低。")
                    reverb = gr.Slider(0, 2, value=1, step=0.1, label="混响", info="1 接近原曲的空间感，0 不添加混响。")
                    with gr.Row():
                        vocal_gain = gr.Slider(-12, 12, value=0, step=0.5, label="人声音量（dB）", info="0 保持原曲人声响度。")
                        music_gain = gr.Slider(-12, 12, value=0, step=0.5, label="伴奏音量（dB）", info="0 保持原曲伴奏响度。")
                    refine = gr.Checkbox(label="精细处理主唱", value=refine_ready, interactive=refine_ready,
                                         info="先分离主唱并去除混响，减少和声干扰。处理时间稍长。" if refine_ready
                                         else "请先安装主唱分离与去混响模型。")
                    index_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="音色检索强度",
                                           info="越高越贴近训练录音，但可能影响咬字；建议从 0.25 开始。")
                start = gr.Button("开始翻唱", variant="primary", elem_id="start-cover", size="lg")
                gr.HTML('<p class="field-note local-note">音频在本机处理 · 成品自动保存到作品库</p>', container=False)
        with gr.Column(elem_id="cover-output", elem_classes="studio-panel", scale=6, min_width=320):
            gr.HTML('<div class="panel-heading"><span class="eyebrow">02 / 制作与试听</span>'
                    '<h2>你的下一首作品</h2></div>', container=False)
            empty = gr.HTML('<div class="empty-studio"><div class="empty-disc" aria-hidden="true">'
                            '<span></span><i></i></div><div class="empty-copy"><h3>让熟悉的旋律，拥有你的声音</h3>'
                            '<p>上传歌曲，点击“开始翻唱”。<br>处理进度和成品会出现在这里。</p>'
                            '<div class="studio-flow"><span>分离</span><b>→</b><span>转换</span><b>→</b><span>混音</span></div>'
                            '</div></div>', elem_id="cover-empty", container=False)
            with gr.Column(visible=False, elem_id="cover-stage") as stage:
                progress = _progress_view("cover-progress")
            # Keep control visibility independent from the stage container;
            # a parent update must not remount its child with stale defaults.
            stop = gr.Button("停止任务", elem_id="stop-cover", size="sm", visible=False)
            result = gr.HTML("", visible=False, elem_id="cover-result", container=False)
            again = gr.Button("继续制作", elem_id="again", visible=False)
    refresh_voice.click(_refresh_voices, outputs=[model], queue=False, show_progress="hidden")
    outputs = [progress, form, stage, result, again, start, stop, empty]
    task_id = gr.BrowserState("", storage_key="aicoversong-cover-task")
    poll = gr.Timer(1.0, active=False)
    start.click(_start_cover,
                inputs=[song, model, pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate, task_id],
                outputs=outputs + [task_id, poll], queue=False, trigger_mode="multiple", show_progress="hidden",
                api_name="cover")
    poll.tick(_poll_cover, inputs=[task_id], outputs=outputs + [poll], queue=False,
              show_progress="hidden", api_name="cover_status")
    # BrowserState contains only this browser's opaque task token. It restores
    # the status after a page refresh without exposing another session's job.
    task_id.change(_poll_cover, inputs=[task_id], outputs=outputs + [poll], queue=False,
                   show_progress="hidden", api_name=False)
    stop.click(_stop_cover, inputs=[task_id], outputs=outputs + [poll], queue=False,
               show_progress="hidden", api_name="stop_cover")
    again.click(_again, inputs=[task_id], outputs=outputs + [task_id, poll], queue=False,
                show_progress="hidden", api_name=False)
    return model


# --------------------------------------------------------------------------
# Voice tab: recording and training
# --------------------------------------------------------------------------

# The browser's default microphone mode is tuned for calls: noise suppression,
# echo cancellation and automatic gain. These features can cut breaths and
# consonant tails before the voice model sees them. Gradio asks for
# {audio: true}; this wrapper turns the
# three call features off for this page. Processing done by the sound driver
# itself is out of its reach; voice_recording._inspect_audio reports that case
# when a take is saved.
RAW_MICROPHONE_JS = """
(() => {
  const devices = navigator.mediaDevices;
  if (!devices || devices.__coverRawMicrophone) return;
  const original = devices.getUserMedia.bind(devices);
  devices.getUserMedia = (constraints) => {
    if (constraints && constraints.audio) {
      const audio = constraints.audio === true ? {} : {...constraints.audio};
      constraints = {...constraints, audio: {...audio, noiseSuppression: false,
                                             echoCancellation: false, autoGainControl: false}};
    }
    return original(constraints);
  };
  devices.__coverRawMicrophone = true;
})();
"""


def _voice_summary() -> str:
    stats = recorder.recording_stats()
    minutes = stats["seconds"] / 60
    target = 10.0
    share = min(1.0, minutes / target)
    high_done, high_total = recorder.high_note_progress()
    if stats["seconds"] < 300:
        advice = f"还差 {recorder._duration(300 - stats['seconds'])} 就可以开始训练。"
    elif stats["seconds"] < 600:
        advice = "已经可以训练了。录到 10 分钟以上会更稳。"
    elif high_done < high_total:
        advice = (f"录音量足够了。想让音很高的歌也像你，把“高音专项”的 {high_total} 段录完"
                  f"（已录 {high_done} 段），再训练一次。")
    else:
        advice = "录音量足够了，高音专项也录完了。训练一次，新录的内容才会用上。"
    return (f'<div class="tally"><div class="tally-bar"><i style="width:{share * 100:.0f}%"></i></div>'
            f'<p><b>已录 {stats["count"]} 段，共 {recorder._duration(stats["seconds"])}</b>'
            f'<span>朗读 {recorder._duration(stats["reading_seconds"])}，清唱 {recorder._duration(stats["singing_seconds"])}</span></p>'
            f'<p class="advice">{html.escape(advice)}</p></div>')


def _can_train() -> bool:
    return recorder.recording_stats()["seconds"] >= 300


def _save(audio_path, index, agreed):
    try:
        _saved, next_index, prompt, _summary, feedback = recorder.save_recording(audio_path, index)
        return (None, next_index, prompt, _voice_summary(), feedback,
                gr.update(interactive=bool(agreed) and _can_train()))
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc), title="这段没有保存", print_exception=False) from exc


def _clear_take(index):
    _, prompt, feedback = recorder.clear_current(index)
    return None, prompt, feedback


def _skip(index):
    next_index = recorder.next_prompt_index(index)
    return None, next_index, recorder.prompt_markdown(next_index), "已跳到下一段。刚才没保存的录音已经清掉。"


def _train(agreed):
    if not _GPU_WORK.acquire(blocking=False):
        yield ({"state": "failed", "message": "已有翻唱或训练正在进行，请等它完成或先停止。"},
               gr.update(visible=True), gr.update(visible=False))
        return
    worker = _train_worker(agreed)
    try:
        yield from worker
    finally:
        try:
            worker.close()
        finally:
            _GPU_WORK.release()


def _train_worker(agreed):
    first = True
    for snapshot in trainer.train_voice(agreed):
        if snapshot["state"] == "running":
            snapshot = {**snapshot, "hint": "训练要一到两个小时。可以先去忙别的，别关这个网页，也别关那个黑色的窗口。"}
            yield (snapshot, gr.update(visible=False) if first else gr.skip(),
                   gr.update(visible=True) if first else gr.skip())
            first = False
        else:
            if snapshot["state"] == "done":
                snapshot = {**snapshot, "message": "训练完成，新的声音可以用了",
                            "detail": "回到“翻唱”页，新的声音已经选好了。"}
            yield snapshot, gr.update(visible=True), gr.update(visible=False)


def _voice_tab():
    initial = recorder.initial_index()
    count = gr.State(initial)
    gr.HTML('<div class="panel-heading"><span class="eyebrow">VOICE / 我的声音</span>'
            '<h2>录下真实的你</h2><p>按稿件提示朗读或清唱，试听满意后保存。可以分几次完成。</p></div>', container=False)
    summary = gr.HTML(_voice_summary(), elem_id="voice-tally", container=False)
    with gr.Row(elem_id="voice-workspace", equal_height=False):
        with gr.Column(elem_id="voice-script", scale=6, min_width=320):
            prompt = gr.HTML(recorder.prompt_markdown(initial), elem_id="cover-recording-prompt", container=False,
                             js_on_load=RAW_MICROPHONE_JS)
        with gr.Column(elem_id="voice-recorder", elem_classes="studio-panel", scale=4, min_width=300):
            gr.HTML('<div class="panel-heading"><span class="eyebrow">录音台</span>'
                    '<h3>先录制，再试听</h3><p>只读或唱稿件正文，不需要读操作说明。</p></div>', container=False)
            recording = gr.Audio(label="麦克风录音", sources=["microphone"], type="filepath", format="wav",
                                 recording=False, interactive=True, editable=False, elem_id="cover-mic-recorder", buttons=[])
            with gr.Row(elem_id="take-actions"):
                save = gr.Button("保存并继续", variant="primary")
                skip = gr.Button("跳过本条")
            feedback = gr.Markdown("停止录音后先试听，满意再保存。需要重录时，点击录音框的 × 清除本次录音。", elem_id="take-feedback")
            with gr.Accordion("录音小贴士", open=False, elem_id="record-tips"):
                gr.Markdown(
                    "- 安静房间，嘴离麦克风约一拳，保持距离。\n"
                    "- 不放伴奏，不开混响、变声或声卡降噪。\n"
                    "- 浏览器录音已请求关闭通话降噪。若保存时提示“纯数字静音”，请检查系统或声卡的降噪设置。\n"
                    "- 读完即停，避免长时间留白。\n"
                    "- 清唱保持舒适音域，高音专项也不要勉强或喊叫。\n"
                    "- 已保存的片段会保留，下次可继续补录。"
                )
    gr.HTML('<h2 class="section-title">训练</h2>'
            '<p class="section-lead">用已保存的录音训练个人音色。耗时取决于显卡与录音量，训练期间请保持电脑唤醒。'
            '已有音色会保留。</p>', container=False)
    with gr.Column(elem_id="train-form") as train_form:
        terms = gr.Checkbox(label="这些录音都是我自己的声音，我不会拿它冒充别人", value=False,
                            info="开始前请阅读并遵守 [Applio 使用条款](https://github.com/IAHispano/Applio/blob/3.7.1/TERMS_OF_USE.md)。")
        train = gr.Button("开始训练", variant="primary", interactive=False, elem_id="start-train")
    with gr.Column(visible=False, elem_id="train-stage") as train_stage:
        cancel = gr.Button("停止训练", size="sm", elem_id="stop-train")
    train_progress = _progress_view("train-progress")

    recording.clear(_clear_take, inputs=[count], outputs=[recording, prompt, feedback], queue=False)
    skip.click(_skip, inputs=[count], outputs=[recording, count, prompt, feedback], queue=False)
    terms.change(lambda agreed: gr.update(interactive=bool(agreed) and _can_train()), inputs=[terms], outputs=[train],
                 queue=False, show_progress="hidden")
    save.click(_save, inputs=[recording, count, terms],
               outputs=[recording, count, prompt, summary, feedback, train],
               concurrency_limit=1, concurrency_id="voice-recording", trigger_mode="once")
    event = train.click(_train, inputs=[terms], outputs=[train_progress, train_form, train_stage],
                        concurrency_limit=1, concurrency_id="gpu-work", trigger_mode="once", show_progress="hidden",
                        api_name="train")
    cancel.click(trainer.cancel_training, outputs=None, queue=False, show_progress="hidden", api_name="stop_training")
    return event


# --------------------------------------------------------------------------
# History tab
# --------------------------------------------------------------------------

def _song_title(source: str) -> str:
    """'渡 - 薛之谦[渡].flac' -> '渡 - 薛之谦': the file name without the album or id in square brackets."""
    title = Path(source).stem.strip()
    trimmed = title
    while True:
        shorter = re.sub(r"\s*\[[^\[\]]*\]\s*$", "", trimmed)
        if shorter == trimmed:
            break
        trimmed = shorter
    return trimmed or title


def _history() -> list[dict]:
    """Finished covers, newest first: what the page needs to list and play them."""
    rows = []
    for run in sorted(pipeline.OUTPUT.glob("run_*"), reverse=True):
        cover, job = run / "cover.wav", run / "job.json"
        if not cover.is_file() or not job.is_file():
            continue
        try:
            data = json.loads(job.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = data.get("original_song") or data.get("song") or ""
        if not source:
            continue  # a re-mix made from two loose files, not a song
        stamp = run.name.split("_")
        when = f"{int(stamp[1][4:6])} 月 {int(stamp[1][6:8])} 日 {stamp[2][:2]}:{stamp[2][2:4]}" if len(stamp) >= 3 else ""
        lowered = False
        try:
            lowered = json.loads((run / "result.json").read_text(encoding="utf-8")).get("voice_range", {}).get("octave", 0) < 0
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        rows.append({"run": run.name, "title": _song_title(source), "when": when, "cover": str(cover), "lowered": lowered})
        if len(rows) >= 60:
            break
    return rows


def _history_html(query: str = "") -> str:
    rows = _history()
    if not rows:
        return ('<div class="history-empty"><span class="eyebrow">你的作品库</span><h3>第一首作品，等你开始</h3>'
                '<p>完成翻唱后，歌曲会自动保存在这里，随时试听和下载。</p></div>')
    total = len(rows)
    rows = [row for row in rows if query.strip().casefold() in row['title'].casefold()]
    if not rows:
        return '<div class="history-empty"><h3>没有找到这首歌</h3><p>试试其他关键词，或清空搜索查看全部作品。</p></div>'
    items = []
    for row in rows:
        tag = '<span class="tag">人声降了八度</span>' if row["lowered"] else ""
        items.append(
            f'<li><div class="meta"><b>{html.escape(row["title"])}</b><span>{html.escape(row["when"])}</span>{tag}</div>'
            + _track(row["cover"], f'{row["title"]} 翻唱.wav').replace('preload="metadata"', 'preload="none"') + "</li>")
    return (f'<p class="history-note">显示 {len(rows)} / {total} 首近期作品 · 按制作时间排序 · 音频保存在本机</p>'
            '<ul class="history">' + "".join(items) + "</ul>")


def _history_tab():
    gr.HTML('<div class="panel-heading"><span class="eyebrow">LIBRARY / 作品库</span>'
            '<h2>每一首，都是你的声音</h2><p>回听、下载，留住每一次创作。</p></div>', container=False)
    with gr.Row(elem_id="history-toolbar"):
        search = gr.Textbox(label="搜索作品", placeholder="按歌曲名称搜索…", show_label=False, scale=5)
        refresh = gr.Button("刷新作品", size="sm", scale=1, min_width=100)
    view = gr.HTML(_history_html(), elem_id="history-view", container=False)
    search.input(_history_html, inputs=[search], outputs=[view], queue=False, show_progress="hidden")
    refresh.click(_history_html, inputs=[search], outputs=[view], queue=False, show_progress="hidden")
    return view, search


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def applio_plugin():
    with gr.Tabs(elem_id="main-tabs") as tabs:
        with gr.Tab("创作工作台", id="cover"):
            model = _cover_tab()
        with gr.Tab("我的声音", id="voice"):
            training = _voice_tab()
        with gr.Tab("作品库", id="history") as history_tab:
            history, search = _history_tab()
    training.then(_refresh_voices, outputs=[model], queue=False, show_progress="hidden")
    history_tab.select(_history_html, inputs=[search], outputs=[history], queue=False, show_progress="hidden")
    return tabs
