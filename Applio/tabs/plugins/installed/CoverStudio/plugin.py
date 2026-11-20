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
import cover_library as library
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
    selected = _default_voice(voices)
    return gr.update(choices=voices, value=selected)


def _refresh_voices_ui():
    voices = _voices()
    selected = _default_voice(voices)
    notice, start = _cover_voice_state(selected, voices)
    return gr.update(choices=voices, value=selected), _voice_alert_update(notice), start


def _is_demo_voice(model: str | None, voices=None) -> bool:
    if not model:
        return False
    value = str(model).casefold()
    if "demo" in value or "演示" in value or "验证" in value:
        return True
    for label, candidate in voices if voices is not None else _voices():
        if str(candidate) == str(model):
            label = str(label).casefold()
            return any(word in label for word in ("演示", "demo", "验证"))
    return False


def _default_voice(voices):
    return next((value for label, value in voices if not _is_demo_voice(value, [((label), value)])),
                voices[0][1] if voices else None)


def _cover_voice_state(model, voices=None):
    voices = _voices() if voices is None else voices
    personal = any(not _is_demo_voice(value, voices) for _label, value in voices)
    if not personal:
        return ("还没有个人音色。请先打开“我的声音”，录音并完成训练；演示音色只用于界面预览。",
                gr.update(interactive=False, value="先训练个人音色"))
    if not model:
        return ("请选择一个个人音色后开始翻唱。", gr.update(interactive=False, value="选择个人音色"))
    if _is_demo_voice(model, voices):
        return ("这个演示音色不能用于生成作品。请选择训练得到的个人音色。",
                gr.update(interactive=False, value="请选择个人音色"))
    return ("", gr.update(interactive=True, value="开始翻唱"))


def _studio_alert_html(message: str, tone: str = "warning") -> str:
    """Render plain, escaped status text; return no element for an empty notice."""
    if not message:
        return ""
    tone = tone if tone in {"warning", "info", "error", "success"} else "info"
    role = "alert" if tone in {"warning", "error"} else "status"
    return (f'<div class="studio-alert" data-tone="{tone}" role="{role}">'
            f'{html.escape(str(message))}</div>')


def _voice_alert_update(message: str):
    """Hide the empty alert container so a successful voice selection leaves no blank bar."""
    return gr.update(value=_studio_alert_html(message), visible=bool(message))


def _cover_voice_change(model):
    message, button = _cover_voice_state(model)
    return _voice_alert_update(message), button


def _settings_summary(pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate):
    pitch = dict(RANGE_CHOICES).get(pitch_mode, "自动")
    harmonies = dict(BACKING_CHOICES).get(backing, "保留原唱和声")
    parts = [pitch, harmonies, f"清晰度 {float(clarity):.0%}", f"混响 {float(reverb):.1f}"]
    if float(vocal_gain) or float(music_gain):
        parts.append(f"人声 {float(vocal_gain):+.1f} dB · 伴奏 {float(music_gain):+.1f} dB")
    if not refine:
        parts.append("未精细处理")
    if float(index_rate) != 0.25:
        parts.append(f"检索 {float(index_rate):.2f}")
    return "当前设置：" + " · ".join(parts)


def _default_cover_settings(refine_ready):
    return "auto", "original", 0.75, 1.0, 0.0, 0.0, bool(refine_ready), 0.25


# --------------------------------------------------------------------------
# Progress component (shared by covers and training)
# --------------------------------------------------------------------------

PROGRESS_TEMPLATE = '<section class="stage" data-state="idle" role="region" aria-label="任务进度"></section>'

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
  if (seconds < 20) return '剩余约 20 秒';
  if (seconds < 90) return '剩余约 1 分钟';
  if (seconds < 3600) return '剩余约 ' + Math.round(seconds / 60) + ' 分钟';
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60 / 5) * 5;
  return '剩余约 ' + h + ' 小时' + (m ? ' ' + m + ' 分' : '');
};
let shown = null;          // last snapshot from the server
let received = 0;          // performance.now() when it arrived
let display = 0;           // last server-reported fraction; CSS transitions the visual bar
let rail = 0;              // how much of the line between the dots is filled
let clockTimer = 0;
let lifecycleObserver = null;
let disposed = false;
let lastSnapshot = null;

function stopClock() {
  if (clockTimer) window.clearTimeout(clockTimer);
  clockTimer = 0;
}

function updateClock() {
  if (!shown || shown.state !== 'running' || !root.isConnected) return;
  const elapsed = (shown.elapsed || 0) + (performance.now() - received) / 1000;
  const reached = Math.min(0.99, Math.max(0, Number(shown.overall) || 0));
  const left = reached > 0.03 && elapsed > 8 ? elapsed * (1 - reached) / reached : null;
  const time = root.querySelector('.time');
  time.textContent = '已用 ' + clock(elapsed) + (left === null ? '' : ' · ' + rough(left) + '（按已报告进度估算，可能变化）');
}

function scheduleClock() {
  if (disposed || clockTimer || !shown || shown.state !== 'running') return;
  clockTimer = window.setTimeout(() => {
    clockTimer = 0;
    if (!element.isConnected) { dispose(); return; }
    updateClock();
    scheduleClock();
  }, document.hidden ? 10000 : 1000);
}

function handleVisibility() {
  if (!shown || shown.state !== 'running') return;
  stopClock();
  updateClock();
  scheduleClock();
}

function dispose() {
  disposed = true;
  stopClock();
  document.removeEventListener('visibilitychange', handleVisibility);
  if (lifecycleObserver) lifecycleObserver.disconnect();
}

document.addEventListener('visibilitychange', handleVisibility);
lifecycleObserver = new MutationObserver(() => { if (!element.isConnected) dispose(); });
if (document.body) lifecycleObserver.observe(document.body, {childList: true, subtree: true});

function structure(data) {
  const steps = (data.steps || []).map((label, i) =>
    `<li data-i="${i}"><span class="dot"></span><span class="name">${esc(label)}</span></li>`).join('');
  root.innerHTML = `
    <div class="stage-head" aria-hidden="true"><span class="stage-code">LIVE / PROCESS</span><span class="stage-badge"></span></div>
    ${data.title ? `<h2 class="what">${esc(data.title)}</h2>` : ''}
    <ol class="steps" style="--n:${(data.steps || []).length}">${steps}<span class="rail"><i></i></span></ol>
    <div class="stage-meter" data-mode="indeterminate" role="progressbar" aria-label="任务整体进度" aria-valuemin="0" aria-valuemax="100" aria-valuetext="正在准备任务进度"><i aria-hidden="true"></i></div>
    <p class="now"><b></b><span class="pct"></span></p>
    <p class="stage-announcement" role="status" aria-live="polite" aria-atomic="true"></p>
    <p class="time"></p>
    <p class="hint"></p>
    <div class="notes"></div>`;
}

function paint() {
  if (!shown) return;
  const data = shown;
  const running = data.state === 'running';
  // No measured worker progress yet means indeterminate, not a misleading 0%.
  // Once the server reports a fraction, interpolate only up to that value.
  const reported = Number(data.overall);
  const measured = Number.isFinite(reported) && reported > 0;
  const target = data.state === 'done' ? 1 : Math.min(0.99, measured ? reported : 0);
  display = target;
  // the line follows the steps: dot i is where step i starts, the stretch after it is that step
  const count = (data.steps || []).length;
  const along = count > 1 ? Math.min(1, ((data.step || 0) + (data.step_fraction || 0)) / (count - 1)) : (data.step_fraction || 0);
  const railTarget = data.state === 'done' ? 1 : running ? along : rail;
  rail = railTarget;
  root.querySelector('.rail i').style.transform = `scaleX(${rail})`;
  const badge = root.querySelector('.stage-badge');
  badge.dataset.state = data.state;
  badge.textContent = data.state === 'running' ? '处理中' : data.state === 'done' ? '已完成' : data.state === 'stopped' ? '已停止' : '未完成';
  const meter = root.querySelector('.stage-meter');
  const meterBar = meter.querySelector('i');
  const meterMode = data.state === 'done' || measured ? 'determinate' : running ? 'indeterminate' : 'unknown';
  meter.dataset.mode = meterMode;
  if (meterMode === 'determinate') {
    meter.setAttribute('aria-valuenow', String(Math.round(display * 1000) / 10));
    meter.setAttribute('aria-valuetext', data.state === 'done' ? '任务已完成' : (display > 0 && display < 0.005 ? '小于 1%' : String(Math.round(display * 100)) + '%'));
    meterBar.style.transform = 'scaleX(' + display + ')';
  } else {
    meter.removeAttribute('aria-valuenow');
    const activeLabel = (data.steps || [])[data.step] || '当前步骤';
    meter.setAttribute('aria-valuetext', running ? '正在' + activeLabel + '，具体进度暂不可测' : '任务未完成，具体进度无法测量');
    meterBar.style.transform = '';
  }
  root.querySelectorAll('.steps li').forEach((li) => {
    const i = Number(li.dataset.i);
    li.dataset.s = data.state === 'done' || i < data.step ? 'done' : i === data.step ? (data.state === 'failed' ? 'failed' : data.state === 'stopped' ? 'stopped' : 'now') : 'todo';
    if (running && i === data.step) li.setAttribute('aria-current', 'step');
    else li.removeAttribute('aria-current');
  });
  const now = root.querySelector('.now b'), pct = root.querySelector('.pct'), time = root.querySelector('.time');
  const announcement = root.querySelector('.stage-announcement');
  root.querySelector('.hint').textContent = running ? (data.hint || '') : '';
  if (running) {
    const label = (data.steps || [])[data.step] || data.message || '';
    if (announcement.dataset.message !== label) {
      announcement.textContent = '正在' + label;
      announcement.dataset.message = label;
    }
    now.textContent = '正在' + label + (data.detail ? `（${data.detail}）` : '');
    pct.textContent = measured ? (display > 0 && display < 0.005 ? '<1%' : Math.round(display * 100) + '%') : '';
    updateClock();
    scheduleClock();
  } else if (data.state === 'done') {
    stopClock();
    now.textContent = data.message || '完成';
    if (announcement.dataset.message !== now.textContent) {
      announcement.textContent = now.textContent;
      announcement.dataset.message = now.textContent;
    }
    pct.textContent = '';
    time.textContent = `共用 ${clock(data.elapsed || 0)}` + (data.detail ? `。${data.detail}` : '');
  } else {
    stopClock();
    now.textContent = data.message || data.error || (data.state === 'stopped' ? '已停止' : '没有完成');
    if (announcement.dataset.message !== now.textContent) {
      announcement.textContent = now.textContent;
      announcement.dataset.message = now.textContent;
    }
    pct.textContent = '';
    time.textContent = data.detail || '';
  }
  const notes = (data.notes || []).map((text) => `<p>${esc(text)}</p>`).join('');
  const box = root.querySelector('.notes');
  if (box.dataset.html !== notes) { box.innerHTML = notes; box.dataset.html = notes; }
}

function update() {
  if (disposed) return;
  let data = props.value;
  if (typeof data === 'string') { try { data = JSON.parse(data); } catch (e) { data = null; } }
  const signature = JSON.stringify(data);
  if (signature === lastSnapshot) return;
  lastSnapshot = signature;
  if (!data || !data.state || data.state === 'idle') {
    shown = null; stopClock();
    root.dataset.state = 'idle'; root.innerHTML = ''; display = 0; rail = 0;
    return;
  }
  const newRun = shown && (shown.run !== data.run || (shown.state !== 'running' && data.state === 'running'));
  const fresh = !shown || newRun || (shown.steps || []).join('|') !== (data.steps || []).join('|') || shown.title !== data.title || !root.firstChild;
  if (fresh) { structure(data); if (!shown || newRun || data.overall < display) { display = 0; rail = 0; } }
  shown = data; received = performance.now();
  root.dataset.state = data.state;
  paint();
  if (data.state === 'running') scheduleClock();
  else stopClock();
}
watch('value', update);
update();
"""

PROGRESS_CSS = """
.stage { --stage-violet: #b9a5ff; --stage-cyan: #79e6e2; display: block; }
.stage[data-state="idle"] { display: none; }
.stage-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin: 0 0 15px; }
.stage-code { color: var(--stage-cyan); font: 600 10px/1 var(--font-mono); letter-spacing: .16em; }
.stage-badge { display: inline-flex; align-items: center; gap: 7px; padding: 6px 9px; border: 1px solid #4b416f; border-radius: 999px; background: #211d34; color: #d5c9ff; font-size: 10px; line-height: 1; letter-spacing: .04em; }
.stage-badge::before { width: 6px; height: 6px; border-radius: 50%; background: currentColor; content: ""; }
.stage-badge[data-state="running"] { color: var(--stage-cyan); border-color: #315b63; background: #13262c; }
.stage-badge[data-state="done"] { color: #a5f0d7; border-color: #316153; background: #152720; }
.stage-badge[data-state="failed"] { color: #ffb6c8; border-color: #673d58; background: #281a2a; }
.stage-badge[data-state="stopped"] { color: #c3bbd4; border-color: #514b60; background: #211f28; }
.what { margin: 0 0 22px; font-family: var(--sans); font-weight: 650; font-size: 20px; line-height: 1.4; color: var(--ink); overflow-wrap: anywhere; }
.hint { margin: 2px 0 0; font-size: 14px; color: var(--ink-3); }
.hint:empty { display: none; }
.steps { position: relative; display: grid; grid-template-columns: repeat(var(--n, 4), 1fr); margin: 0; padding: 0 0 2px; list-style: none; }
.steps li { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center; }
.steps .dot { width: 12px; height: 12px; border-radius: 50%; background: var(--paper-3); box-shadow: 0 0 0 4px var(--paper); transition: background .35s, transform .35s, box-shadow .35s; }
.steps .name { font-size: 12px; line-height: 1.35; color: var(--ink-3); max-width: 10em; transition: color .35s; }
.steps li[data-s="done"] .dot { background: var(--done); }
.steps li[data-s="done"] .name { color: var(--ink-2); }
.steps li[data-s="now"] .dot { background: var(--stage-cyan); box-shadow: 0 0 0 4px var(--paper), 0 0 15px #79e6e266; transform: scale(1.2); }
.steps li[data-s="now"] .name { color: var(--ink); font-weight: 600; }
.steps li[data-s="failed"] .dot { background: var(--bad); }
.steps li[data-s="failed"] .name { color: var(--bad); }
.steps li[data-s="stopped"] .dot { background: var(--ink-3); }
.rail { position: absolute; left: calc(50% / var(--n, 4)); right: calc(50% / var(--n, 4)); top: 6px; height: 2px; background: var(--paper-3); border-radius: 2px; overflow: hidden; }
.rail i { display: block; height: 100%; background: linear-gradient(90deg, var(--stage-violet), var(--stage-cyan)); transform-origin: left; transform: scaleX(0); transition: transform .7s cubic-bezier(.22,.8,.25,1); }
.stage-meter { position: relative; height: 4px; margin: 20px 0 0; overflow: hidden; border-radius: 99px; background: var(--paper-3); }
.stage-meter i { display: block; width: 100%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--stage-violet), var(--stage-cyan)); transform-origin: left; transform: scaleX(0); transition: transform .7s cubic-bezier(.22,.8,.25,1); }
.stage-meter[data-mode="indeterminate"] i { width: 32%; transform: none; background: linear-gradient(90deg, transparent, var(--stage-violet), var(--stage-cyan), transparent); animation: meter-sweep 1.6s cubic-bezier(.6,0,.4,1) infinite; }
.stage-meter[data-mode="unknown"] i { width: 0; transform: none; opacity: 0; }
.stage[data-state="done"] .stage-meter i { background: var(--done); }
.stage[data-state="failed"] .stage-meter i { background: var(--bad); }
.stage[data-state="stopped"] .stage-meter i { background: var(--ink-3); }
.stage[data-state="done"] .rail i { background: var(--done); }
.now { margin: 19px 0 0; display: flex; align-items: baseline; justify-content: space-between; gap: 16px; font-size: 18px; line-height: 1.4; color: var(--ink); }
.now b { font-weight: 600; }
.now .pct { font-variant-numeric: tabular-nums; color: var(--stage-cyan); font: 600 16px/1 var(--font-mono); }
.stage[data-state="failed"] .now b { color: var(--bad); font-size: 17px; font-weight: 500; white-space: pre-line; }
.time { margin: 7px 0 0; font: 12px/1.7 var(--font-mono); color: var(--ink-3); font-variant-numeric: tabular-nums; min-height: 1.5em; }
.stage-announcement { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.notes p { margin: 14px 0 0; padding: 12px 14px; border-left: 3px solid var(--lamp); background: var(--paper-2); color: var(--ink-2); font-size: 14.5px; line-height: 1.7; border-radius: 0 8px 8px 0; }
@keyframes breathe { 0%, 100% { box-shadow: 0 0 0 5px var(--paper), 0 0 0 5px transparent; } 50% { box-shadow: 0 0 0 5px var(--paper), 0 0 0 10px color-mix(in srgb, var(--lamp) 16%, transparent); } }
@keyframes meter-sweep { from { transform: translateX(-125%); } to { transform: translateX(440%); } }
@media (prefers-reduced-motion: reduce) { .steps .dot, .rail i, .stage-meter i { transition: none; } .stage-meter[data-mode="indeterminate"] i { width: 100%; transform: scaleX(.4); opacity: .7; animation: none; } }
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
            f'<div class="result-actions result-primary"><a class="save" href="{html.escape(_file_url(item["mix"]), quote=True)}" '
            f'download="{html.escape(title, quote=True)} 翻唱.wav">下载成品</a>'
            f'<span class="result-tip">成品已保存到作品库</span></div>'
            f'{_track(item["mix"], f"{title} 翻唱.wav", "最终成品", main=True)}{fold}</section>')


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
    voices = _voices()
    if not model or not any(not _is_demo_voice(value, voices) for _label, value in voices):
        raise gr.Error("还没有个人音色。先到“我的声音”里录音并训练一次。", title="还不能开始", print_exception=False)
    if _is_demo_voice(model, voices):
        raise gr.Error("演示音色只用于界面预览，请选择训练得到的个人音色。", title="还不能开始", print_exception=False)
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
    selected_voice = _default_voice(voices)
    voice_notice, voice_button = _cover_voice_state(selected_voice, voices)
    defaults = _default_cover_settings(refine_ready)
    with gr.Row(elem_id="cover-workspace", equal_height=False):
        with gr.Column(elem_id="cover-controls", elem_classes="studio-panel", scale=5, min_width=320):
            gr.HTML('<div class="panel-heading"><span class="eyebrow">01 <i>/</i> INPUT</span>'
                    '<h2>歌曲与音色</h2></div>', container=False)
            with gr.Column(elem_id="cover-form") as form:
                song = gr.File(label="上传歌曲", file_types=["audio"], type="filepath", elem_id="song-drop",
                               height=110)
                with gr.Row(elem_id="voice-picker-row"):
                    model = gr.Dropdown(label="我的音色", choices=voices, value=selected_voice,
                                        interactive=True, elem_id="voice-pick", scale=5,
                                        info=None if voices else "还没有音色？到“我的声音”录音并训练，完成后即可选择。")
                    refresh_voice = gr.Button("刷新音色", elem_id="refresh-voices", size="sm", scale=1, min_width=90)
                voice_notice_view = gr.HTML(_studio_alert_html(voice_notice), visible=bool(voice_notice),
                                            elem_id="voice-required", container=False)
                with gr.Accordion("声音设置", open=False, elem_id="more-settings"):
                    pitch_mode = gr.Radio(choices=RANGE_CHOICES, value=defaults[0], label="音域适配",
                                          info="自动模式只在原唱明显超出你的录音音域时，将人声降低一个八度；伴奏保持原调。")
                    backing = gr.Radio(choices=BACKING_CHOICES, value=defaults[1], label="和声处理",
                                       info="保留原唱和声通常更自然，主唱仍会转换为你的声音。")
                    clarity = gr.Slider(0, 1, value=defaults[2], step=0.05, label="人声清晰度",
                                        info="歌词不清楚可调高；声音偏尖、偏薄可调低。")
                    reverb = gr.Slider(0, 2, value=defaults[3], step=0.1, label="混响", info="1 接近原曲的空间感，0 不添加混响。")
                    with gr.Row():
                        vocal_gain = gr.Slider(-12, 12, value=defaults[4], step=0.5, label="人声音量（dB）", info="0 保持原曲人声响度。")
                        music_gain = gr.Slider(-12, 12, value=defaults[5], step=0.5, label="伴奏音量（dB）", info="0 保持原曲伴奏响度。")
                    refine = gr.Checkbox(label="精细处理主唱", value=defaults[6], interactive=refine_ready,
                                         info="先分离主唱并去除混响，减少和声干扰。处理时间稍长。" if refine_ready
                                         else "请先安装主唱分离与去混响模型。")
                    index_rate = gr.Slider(0, 1, value=defaults[7], step=0.05, label="音色检索强度",
                                           info="越高越贴近训练录音，但可能影响咬字；建议从 0.25 开始。")
                settings_summary = gr.Markdown(
                    _settings_summary(*defaults), elem_id="cover-settings-summary")
                with gr.Row(elem_id="cover-settings-actions"):
                    reset_settings = gr.Button("恢复默认", elem_id="reset-cover", size="sm")
                    start = gr.Button(voice_button["value"] if isinstance(voice_button, dict) else "开始翻唱",
                                      variant="primary", elem_id="start-cover", size="lg",
                                      interactive=voice_button.get("interactive", True) if isinstance(voice_button, dict) else True)
                gr.HTML('<p class="field-note local-note">音频在本机处理 · 成品自动保存到作品库</p>', container=False)
        with gr.Column(elem_id="cover-output", elem_classes="studio-panel", scale=6, min_width=320):
            gr.HTML('<div class="panel-heading"><span class="eyebrow">02 <i>/</i> MONITOR</span>'
                    '<h2>制作与试听</h2></div>', container=False)
            empty = gr.HTML('<div class="empty-studio"><div class="record-art" aria-hidden="true">'
                            '<span class="platter"></span><span class="sleeve"><b>SIDE A</b><i>VOICE<br>EDITION</i></span></div>'
                            '<div class="empty-copy"><h3>等待第一轨</h3>'
                            '<p>选好歌曲和音色，点击「开始翻唱」。<br>这里会显示制作进度，完成后即可试听。</p></div></div>',
                            elem_id="cover-empty", container=False)
            with gr.Column(visible=False, elem_id="cover-stage") as stage:
                progress = _progress_view("cover-progress")
            # Keep control visibility independent from the stage container;
            # a parent update must not remount its child with stale defaults.
            stop = gr.Button("停止任务", elem_id="stop-cover", size="sm", visible=False)
            result = gr.HTML("", visible=False, elem_id="cover-result", container=False)
            again = gr.Button("继续制作", elem_id="again", visible=False)
    refresh_voice.click(_refresh_voices_ui, outputs=[model, voice_notice_view, start], queue=False, show_progress="hidden")
    model.change(_cover_voice_change, inputs=[model], outputs=[voice_notice_view, start],
                 queue=False, show_progress="hidden")
    settings_inputs = [pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate]
    for control in settings_inputs:
        control.change(_settings_summary, inputs=settings_inputs, outputs=[settings_summary],
                      queue=False, show_progress="hidden")
    reset_settings.click(lambda: (*_default_cover_settings(refine_ready), _settings_summary(*_default_cover_settings(refine_ready))),
                         outputs=settings_inputs + [settings_summary], queue=False, show_progress="hidden")
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
        tone = ("warning" if any(word in feedback for word in ("爆音", "偏小", "注意："))
                else "info" if "已经保存过" in feedback else "success")
        return (None, next_index, prompt, _voice_summary(), _studio_alert_html(feedback, tone),
                gr.update(interactive=bool(agreed) and _can_train()), _training_gate(agreed),
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False))
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc), title="这段没有保存", print_exception=False) from exc


def _clear_take(index):
    _, prompt, feedback = recorder.clear_current(index)
    return (None, prompt, gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False), _studio_alert_html(feedback, "info"))


def _retake_take(index):
    _, prompt, _feedback = recorder.clear_current(index)
    return (None, prompt, gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False),
            _studio_alert_html("已清除这段录音，请重新录制当前稿件。", "info"))


def _discard_take(index):
    _, prompt, _feedback = recorder.clear_current(index)
    return (None, prompt, gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False),
            _studio_alert_html("本次录音已清除，当前稿件还没有保存。", "info"))


def _take_action_state(audio_path):
    enabled = bool(audio_path)
    return (gr.update(interactive=enabled), gr.update(interactive=enabled),
            gr.update(interactive=enabled), gr.update(interactive=enabled),
            _studio_alert_html("已录好这一段。先试听，满意后保存；不满意可以重录。" if enabled
                               else "点击麦克风录音，读完后先试听，满意再保存。", "info"))


def _preview_take(audio_path):
    return _studio_alert_html("正在试听这段录音。确认声音清楚、没有爆音后再保存；不满意可以直接重录。"
                              if audio_path else "还没有录音，请先点击麦克风录音。", "info")


def _training_gate(agreed):
    stats = recorder.recording_stats()
    if stats["seconds"] < 300:
        return _studio_alert_html(f"还差 {recorder._duration(300 - stats['seconds'])} 的录音才能开始训练。", "info")
    if not agreed:
        return _studio_alert_html("勾选上面的使用声明后，才能开始训练。", "warning")
    return _studio_alert_html("录音和声明都已满足，可以开始训练。", "success")


def _skip(index):
    next_index = recorder.next_prompt_index(index)
    return (None, next_index, recorder.prompt_markdown(next_index),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False),
            _studio_alert_html("已跳到下一段。刚才没保存的录音已经清掉。", "info"))


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
                preview = gr.Button("试听", elem_id="take-preview", size="sm", interactive=False)
                retake = gr.Button("重录", elem_id="take-retake", size="sm", interactive=False)
                clear = gr.Button("清除本次录音", elem_id="take-clear", size="sm", interactive=False)
            with gr.Row(elem_id="take-next"):
                skip = gr.Button("跳过本条", elem_id="take-skip")
                save = gr.Button("保存并继续", elem_id="take-save", variant="primary", interactive=False)
            feedback = gr.HTML(_studio_alert_html("点击麦克风录音，读完后先试听，满意再保存。", "info"),
                               elem_id="take-feedback", container=False)
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
    train_gate = gr.HTML(_training_gate(False), elem_id="training-gate", container=False)

    recording.clear(_clear_take, inputs=[count],
                    outputs=[recording, prompt, preview, retake, clear, save, feedback], queue=False)
    recording.change(_take_action_state, inputs=[recording], outputs=[preview, retake, clear, save, feedback],
                     queue=False, show_progress="hidden")
    preview.click(_preview_take, inputs=[recording], outputs=[feedback], queue=False, show_progress="hidden",
                  js="""() => { const player = document.querySelector('#cover-mic-recorder audio'); if (player) { player.currentTime = 0; player.play().catch(() => {}); } }""")
    retake.click(_retake_take, inputs=[count], outputs=[recording, prompt, preview, retake, clear, save, feedback], queue=False,
                 show_progress="hidden")
    clear.click(_discard_take, inputs=[count], outputs=[recording, prompt, preview, retake, clear, save, feedback], queue=False,
                show_progress="hidden")
    skip.click(_skip, inputs=[count], outputs=[recording, count, prompt, preview, retake, clear, save, feedback], queue=False)
    terms.change(lambda agreed: (gr.update(interactive=bool(agreed) and _can_train()), _training_gate(agreed)),
                 inputs=[terms], outputs=[train, train_gate],
                 queue=False, show_progress="hidden")
    save.click(_save, inputs=[recording, count, terms],
               outputs=[recording, count, prompt, summary, feedback, train, train_gate,
                        preview, retake, clear, save],
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


def _history_rows(rows, query="", order="newest"):
    needle = (query or "").strip().casefold()
    if needle:
        rows = [row for row in rows if needle in row["title"].casefold() or needle in Path(row["source"]).name.casefold()]
    if order == "oldest":
        rows = list(reversed(rows))
    elif order == "title":
        rows = sorted(rows, key=lambda row: row["title"].casefold())
    return rows[:60]


def _history_html(rows, total, selected=None):
    if not total:
        return ('<div class="history-empty"><span class="history-empty-icon" aria-hidden="true">♫</span>'
                '<h3>你的第一首作品会出现在这里</h3><p>到「创作工作台」制作翻唱，完成后就能在这里试听和下载。</p></div>')
    if not rows:
        return '<div class="history-empty"><h3>没有匹配的作品</h3><p>换个歌名，或清空搜索查看全部作品。</p></div>'
    items = []
    for row in rows:
        created = row.get("created_at", "")
        when = created[:16].replace("T", " ") if created else "时间未知"
        tag = '<span class="tag">混音版本</span>' if row.get("parent_run") else ""
        active = row["run_id"] == selected
        active_attr = "true" if active else "false"
        name = html.escape(row["title"], quote=True)
        items.append(
            f'<li class="history-entry" data-selected="{active_attr}"><div class="history-entry-heading">'
            f'<div class="meta"><b>{name}</b><span>{html.escape(when)}</span>{tag}</div>'
            f'<button type="button" class="history-pick" data-work="{html.escape(row["run_id"], quote=True)}" '
            f'aria-label="管理：{name}" aria-controls="history-inspector" aria-expanded="{active_attr}">'
            f'{"收起" if active else "管理"}</button></div>'
            + _track(row["cover"], f'{row["title"]} 翻唱.wav').replace('preload="metadata"', 'preload="none"') + "</li>")
    return (f'<p class="history-note" role="status">显示 {len(rows)} / {total} 首作品</p>'
            '<ul class="history" aria-label="作品列表">' + "".join(items) + "</ul>")


_HISTORY_PICK_JS = """
element.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-work]');
  if (!button) return;
  const id = button.getAttribute('aria-expanded') === 'true' ? '' : button.dataset.work;
  for (const entry of element.querySelectorAll('button[data-work]')) {
    const active = entry.dataset.work === id;
    entry.setAttribute('aria-expanded', String(active));
    entry.textContent = active ? '收起' : '管理';
    entry.closest('.history-entry').dataset.selected = String(active);
  }
  trigger('click', {run_id: id});
});
"""


def _trash_choices():
    return [(f'{row["title"]} · {row.get("created_at", "")[:16].replace("T", " ") or "日期未知"} · {row["run_id"][-4:]}',
             row["run_id"]) for row in library.list_trashed(pipeline.OUTPUT)]


def _history_refresh(query="", order="newest", selected=None, trash_id=None, *, message=""):
    # One inventory per refresh: the visible list and inspector always use the same snapshot.
    all_rows = library.list_works(pipeline.OUTPUT)
    rows = _history_rows(all_rows, query, order)
    work = next((row for row in rows if row["run_id"] == selected), None)
    selected = work["run_id"] if work else None
    trashed = _trash_choices()
    trash_id = trash_id if any(value == trash_id for _, value in trashed) else None
    return (_history_html(rows, len(all_rows), selected), selected,
            gr.update(choices=trashed, value=trash_id), _studio_alert_html(message, "success") if message else "",
            gr.update(visible=bool(trashed), label=f"回收站 · {len(trashed)} 首"),
            gr.update(interactive=trash_id is not None), gr.update(visible=bool(all_rows)),
            *_history_selection(work))


def _history_selection(work):
    if not work:
        return (gr.update(visible=False), "", "", 0, 0, 0.75, 1.0, "original")
    job = work["job"]
    note = ('<div class="history-section-heading"><span class="eyebrow">作品管理</span>'
            f'<h3>{html.escape(work["title"])}</h3></div>')
    return (gr.update(visible=True), note, work["title"], float(job.get("vocal_gain", 0)),
            float(job.get("instrumental_gain", 0)), float(job.get("clarity", 0.75)),
            float(job.get("reverb", 1.0)), job.get("backing", "original"))


def _history_pick(evt: gr.EventData):
    try:
        work = library.get_work(evt.run_id, pipeline.OUTPUT) if evt.run_id else None
    except (ValueError, OSError):
        return None, _studio_alert_html("这首作品已不在原位置，刷新作品列表后再试。", "warning"), *_history_selection(None)
    return work["run_id"] if work else None, "", *_history_selection(work)


def _history_rename(run_id, title, query="", order="newest"):
    if not run_id:
        raise gr.Error("先选择要重命名的作品。", title="没有选中作品", print_exception=False)
    library.rename_work(run_id, title, pipeline.OUTPUT)
    return _history_refresh(query, order, run_id, message="作品名称已更新。")


def _history_trash(run_id, query="", order="newest"):
    if not run_id:
        raise gr.Error("先选择要移入回收站的作品。", title="没有选中作品", print_exception=False)
    library.trash_work(run_id, pipeline.OUTPUT)
    return _history_refresh(query, order, trash_id=run_id, message="作品已移入回收站，可随时恢复。")


def _history_restore(run_id, query="", order="newest"):
    if not run_id:
        raise gr.Error("先选择回收站中的作品。", title="没有选中作品", print_exception=False)
    library.restore_work(run_id, pipeline.OUTPUT)
    return _history_refresh(query, order, run_id, message="作品已恢复到作品库。")


def _history_remix(run_id, vocal_gain, music_gain, clarity, reverb, backing, query="", order="newest"):
    if not run_id:
        raise gr.Error("先选择要重新混音的作品。", title="没有选中作品", print_exception=False)
    try:
        work = library.remix_work(run_id, output_root=pipeline.OUTPUT, vocal_gain_db=vocal_gain,
                                  music_gain_db=music_gain, clarity=clarity, reverb=reverb,
                                  backing_mode=backing)
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(str(exc), title="无法生成新版本", print_exception=False) from exc
    return _history_refresh(query, order, work["run_id"], message=f'已创建新版本“{work["title"]}”，原作品仍保留。')


def _history_tab(history_tab):
    rows = library.list_works(pipeline.OUTPUT)
    trashed = _trash_choices()
    selected = gr.State(None)
    gr.HTML('<header class="panel-heading history-heading"><span class="eyebrow">作品库 / LIBRARY</span>'
            '<h2>我的作品</h2><p>试听与下载成品，或为一首歌制作新的混音版本。</p></header>', container=False)
    with gr.Row(elem_id="history-toolbar"):
        with gr.Row(elem_id="history-filter-controls", visible=bool(rows)) as filters:
            search = gr.Textbox(label="搜索作品", placeholder="输入歌名或源文件名", show_label=True,
                                scale=5, elem_id="history-search")
            order = gr.Dropdown(choices=[("最新制作", "newest"), ("最早制作", "oldest"), ("按名称", "title")],
                                value="newest", label="排序", filterable=False, scale=2, elem_id="history-order")
        refresh = gr.Button("刷新作品", size="sm", scale=0, min_width=92, elem_id="history-refresh")
    status = gr.HTML("", elem_id="history-feedback", container=False)
    # Recovery is reachable even if every work has been moved out of the main list.
    with gr.Accordion(f"回收站 · {len(trashed)} 首", visible=bool(trashed), open=False,
                      elem_id="history-recovery") as recovery:
        with gr.Row(elem_id="history-recovery-actions"):
            restore_from = gr.Dropdown(choices=trashed, value=None, label="待恢复的作品",
                                       elem_id="history-trash-selection", interactive=True, scale=4)
            restore = gr.Button("恢复到作品库", size="sm", elem_id="history-restore", interactive=False, scale=0,
                                min_width=110)
    with gr.Row(elem_id="history-workspace", equal_height=False):
        with gr.Column(scale=2, min_width=350, elem_id="history-collection"):
            view = gr.HTML(_history_html(rows[:60], len(rows)), elem_id="history-view", container=False,
                           js_on_load=_HISTORY_PICK_JS)
        with gr.Column(scale=1, min_width=280, visible=False,
                       elem_id="history-inspector", elem_classes="studio-panel") as inspector:
            selection_note = gr.HTML("", elem_id="history-selection-note", container=False)
            title = gr.Textbox(label="作品名称", value="", placeholder="输入新的作品名称", elem_id="history-title")
            with gr.Row(elem_id="history-basic-actions"):
                rename = gr.Button("保存名称", size="sm", elem_id="history-rename")
                trash = gr.Button("移入回收站", size="sm", elem_id="history-trash")
            with gr.Accordion("创建新混音版本", open=False, elem_id="history-remix-settings"):
                gr.HTML('<p class="field-note">从已保存的分轨重新混音，原作品会保留。</p>', container=False)
                remix_vocal = gr.Slider(-24, 12, value=0, step=0.5, label="人声音量（dB）")
                remix_music = gr.Slider(-24, 12, value=0, step=0.5, label="伴奏音量（dB）")
                remix_clarity = gr.Slider(0, 1, value=0.75, step=0.05, label="人声清晰度")
                remix_reverb = gr.Slider(0, 2, value=1, step=0.1, label="混响")
                remix_backing = gr.Dropdown(choices=BACKING_CHOICES, value="original", label="和声处理",
                                            filterable=False)
                remix = gr.Button("生成新版本", variant="primary", elem_id="history-remix-button")
    selection_outputs = [inspector, selection_note, title, remix_vocal, remix_music, remix_clarity, remix_reverb, remix_backing]
    refresh_outputs = [view, selected, restore_from, status, recovery, restore, filters, *selection_outputs]
    quiet = dict(queue=False, show_progress="hidden")
    for event in (search.input, order.input, refresh.click, history_tab.select):
        event(_history_refresh, inputs=[search, order, selected, restore_from], outputs=refresh_outputs, **quiet)
    view.click(_history_pick, inputs=None, outputs=[selected, status, *selection_outputs], **quiet)
    restore_from.input(lambda value: gr.update(interactive=bool(value)), inputs=[restore_from], outputs=[restore], **quiet)
    rename.click(_history_rename, inputs=[selected, title, search, order], outputs=refresh_outputs, **quiet)
    trash.click(_history_trash, inputs=[selected, search, order], outputs=refresh_outputs, **quiet)
    restore.click(_history_restore, inputs=[restore_from, search, order], outputs=refresh_outputs, **quiet)
    remix.click(_history_remix, inputs=[selected, remix_vocal, remix_music, remix_clarity, remix_reverb,
                                        remix_backing, search, order], outputs=refresh_outputs,
                queue=True, trigger_mode="once", show_progress="minimal", show_progress_on=[status])


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
            _history_tab(history_tab)
    training.then(_refresh_voices, outputs=[model], queue=False, show_progress="hidden")
    return tabs
