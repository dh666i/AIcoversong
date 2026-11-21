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
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import quote

import gradio as gr
import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))
import cover_pipeline as pipeline
import cover_library as library
import voice_recording as recorder
import voice_training as trainer

BACKING_CHOICES = [("保留原唱的和声（推荐）", "original"), ("和声也换成我的声音", "convert"), ("不要和声", "remove")]
RANGE_CHOICES = [("自动（推荐）", "auto"), ("保持原来的音高", "keep"), ("降一个八度", "down")]
DUET_MODES = [("常规翻唱", "off"), ("只换男声，保留原女声", "male"),
              ("只换女声，保留原男声", "female"), ("男女声都换", "both")]
SAME_VOICE = "__same_as_primary__"
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


def _refresh_voices(current=None):
    voices = _voices()
    selected = current if any(path == current for _, path in voices) else _default_voice(voices)
    return gr.update(choices=voices, value=selected)


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


def _cover_requirements(song, model, voices=None):
    """Return a voice notice and the reason the current inputs cannot start a cover."""
    voices = _voices() if voices is None else voices
    personal = any(not _is_demo_voice(value, voices) for _label, value in voices)
    if not personal:
        return ("还没有个人音色。请先打开“我的声音”，录音并完成训练；演示音色只用于界面预览。",
                "先训练个人音色")
    if not any(value == model for _label, value in voices):
        return "请选择一个可用的个人音色。", "请选择个人音色"
    if _is_demo_voice(model, voices):
        return ("这个演示音色不能用于生成作品。请选择训练得到的个人音色。",
                "请选择个人音色")
    return "", "" if song else "请先选择歌曲"


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


def _cover_start_button(blocked_reason, snapshot):
    if snapshot is not None and not snapshot[2]:
        return gr.update(interactive=False, value="正在停止…" if snapshot[3] else "正在制作…")
    return gr.update(interactive=not blocked_reason, value=blocked_reason or "开始翻唱")


def _duet_requirements(mode, source, second, male, female, voices):
    if mode == "off":
        return "", ""
    if mode not in {"male", "female", "both"} or source not in {"auto", "stems"}:
        return "请选择有效的对唱模式和人声来源。", "请选择对唱模式"
    if mode == "both" and second and second != SAME_VOICE:
        if not any(value == second for _, value in voices) or _is_demo_voice(second, voices):
            return "女声的目标音色不可用，可选择与男声使用同一音色。", "请选择女声音色"
    if source == "stems":
        if not male or not female:
            return "请分别上传男声和女声的完整人声分轨。", "请补齐两条分轨"
        if Path(male).resolve() == Path(female).resolve():
            return "两个声部选中了同一个文件，请重新选择。", "请检查分轨"
    else:
        from duet_separator import duet_available
        if not duet_available():
            return ("自动分离需要额外模型。在项目根目录运行 .\\env\\Scripts\\python.exe "
                    "tools/download_assets.py --duet，安装后点「刷新音色」。也可选择已有分轨。", "缺少对唱分离模型")
    return "", ""


def _cover_input_change(song, model, current_task, mode="off", source="auto", second=None, male=None, female=None):
    voices = _voices()
    message, reason = _cover_requirements(song, model, voices)
    if not message:
        extra_message, extra_reason = _duet_requirements(mode, source, second, male, female, voices)
        message, reason = extra_message, extra_reason or reason
    return _voice_alert_update(message), reason, _cover_start_button(reason, _task_snapshot(current_task))


def _settings_summary(pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate, mode="off"):
    pitch = next((label for label, value in RANGE_CHOICES if value == pitch_mode), "自动")
    harmonies = next((label for label, value in BACKING_CHOICES if value == backing), "保留原唱和声")
    pitch = pitch.removesuffix("（推荐）")
    harmonies = harmonies.removesuffix("（推荐）")
    scope = harmonies if mode == "off" else next(label for label, value in DUET_MODES if value == mode)
    parts = [pitch, scope, f"清晰度 {float(clarity):.0%}", f"混响 {float(reverb):.1f}"]
    if float(vocal_gain) or float(music_gain):
        parts.append(f"人声 {float(vocal_gain):+.1f} dB · 伴奏 {float(music_gain):+.1f} dB")
    if mode == "off" and not refine:
        parts.append("未精细处理")
    if float(index_rate) != 0.25:
        parts.append(f"检索 {float(index_rate):.2f}")
    return " · ".join(parts)


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
  const left = !shown.stopping && reached > 0.03 && elapsed > 8 ? elapsed * (1 - reached) / reached : null;
  const time = root.querySelector('.time');
  time.textContent = '已用 ' + clock(elapsed) + (left === null ? '' : ' · ' + rough(left));
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
    `<li data-i="${i}"><span class="dot" aria-hidden="true"></span><span class="name">${esc(label)}</span><span class="step-status"></span></li>`).join('');
  root.innerHTML = `
    <div class="stage-head"><div class="stage-caption"><span class="stage-badge"></span><span class="step-count"></span></div>
      <h3 class="what">${esc(data.title || '音色训练')}</h3>
      <span class="pct" aria-hidden="true"></span></div>
    <div class="stage-meter" data-mode="indeterminate" role="progressbar" aria-label="任务整体进度" aria-valuemin="0" aria-valuemax="100" aria-valuetext="正在准备任务进度"><i aria-hidden="true"></i></div>
    <p class="time"></p>
    <ol class="steps" aria-label="处理步骤">${steps}</ol>
    <p class="now"></p>
    <p class="stage-announcement" role="status" aria-live="polite" aria-atomic="true"></p>
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
  const display = data.state === 'done' ? 1 : Math.min(0.99, measured ? reported : 0);
  const count = (data.steps || []).length;
  const step = Math.min(count - 1, Math.max(0, Number(data.step) || 0));
  root.querySelector('.step-count').textContent = running && count > 1 ? `第 ${step + 1} / ${count} 步` : '';
  const badge = root.querySelector('.stage-badge');
  badge.textContent = data.stopping ? '正在停止' : ({running:'制作中', done:'已完成', stopped:'已停止', failed:'未完成'})[data.state];
  if (running && !data.title && !data.stopping) badge.textContent = '训练中';
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
    const state = data.state === 'done' || i < step ? 'done' : i === step ? (running ? 'now' : data.state) : 'todo';
    li.dataset.s = state;
    li.querySelector('.step-status').textContent = ({done:'完成', now:data.stopping ? '停止中' : '处理中', failed:'失败', stopped:'已停止', todo:'待处理'})[state];
    if (running && i === step) li.setAttribute('aria-current', 'step');
    else li.removeAttribute('aria-current');
  });
  const now = root.querySelector('.now'), pct = root.querySelector('.pct'), time = root.querySelector('.time');
  const announcement = root.querySelector('.stage-announcement');
  root.querySelector('.hint').textContent = running ? (data.hint || '') :
    data.state !== 'done' ? (data.detail || '') : '';
  if (running) {
    const label = data.stopping ? '停止任务' : (data.steps || [])[step] || data.message || '准备任务';
    if (announcement.dataset.message !== label) {
      announcement.textContent = '正在' + label;
      announcement.dataset.message = label;
    }
    now.textContent = data.stopping ? '' : data.detail || (count ? '' : data.message || '');
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
    pct.textContent = '100%';
    time.textContent = Number.isFinite(data.elapsed) ? `共用 ${clock(data.elapsed)}` : '';
    if (data.detail) now.textContent = data.detail;
  } else {
    stopClock();
    now.textContent = data.message || data.error || (data.state === 'stopped' ? '已停止' : '没有完成');
    if (announcement.dataset.message !== now.textContent) {
      announcement.textContent = now.textContent;
      announcement.dataset.message = now.textContent;
    }
    pct.textContent = measured ? Math.round(display * 100) + '%' : '';
    time.textContent = Number.isFinite(data.elapsed) ? '已用 ' + clock(data.elapsed) : '';
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
    root.dataset.state = 'idle'; root.innerHTML = '';
    return;
  }
  const newRun = shown && (shown.run !== data.run || (shown.state !== 'running' && data.state === 'running'));
  const fresh = !shown || newRun || (shown.steps || []).join('|') !== (data.steps || []).join('|') || shown.title !== data.title || !root.firstChild;
  if (fresh) structure(data);
  shown = data; received = performance.now();
  root.dataset.state = data.stopping ? 'stopping' : data.state;
  paint();
  if (data.state === 'running') scheduleClock();
  else stopClock();
}
watch('value', update);
update();
"""

PROGRESS_CSS = """
.stage {
  --stage-cyan: var(--primary, #d8ff44);
  --stage-violet: var(--cyan, #55f1ff);
  --stage-accent: var(--stage-cyan);
  --stage-line: var(--line, #464550);
  --stage-text: var(--text, #f8f5ea);
  --stage-muted: var(--muted, #b7b5c3);
  display: grid; gap: 12px; min-width: 0; width: 100%; box-sizing: border-box;
  color: var(--stage-text); font-family: var(--sans, sans-serif);
}
.stage[data-state="idle"], .stage :is(.now,.time,.hint,.notes,.steps):empty { display: none; }
.stage[data-state="done"] { --stage-accent: var(--success, #64e8b6); }
.stage:is([data-state="stopped"],[data-state="stopping"]) { --stage-accent: var(--orange, #ffac64); }
.stage[data-state="failed"] { --stage-accent: var(--danger, #ff8493); }
.stage-head { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 8px 14px; align-items: end; }
.stage-caption { grid-column: 1/-1; display: flex; align-items: center; gap: 12px; }
.stage-badge { display: inline-flex; align-items: center; gap: 8px; color: var(--stage-accent); font: 650 13px/1.5 var(--sans,sans-serif); }
.stage-badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.step-count { margin-inline-start: auto; color: var(--stage-muted); font: 12px/1.5 var(--mono,monospace); }
.what { margin: 0; color: var(--stage-text); font: 730 21px/1.35 var(--sans,sans-serif); overflow-wrap: anywhere; }
.pct { color: var(--stage-accent); font: 650 30px/1 var(--mono,monospace); font-variant-numeric: tabular-nums; }
.stage-meter { height: 6px; overflow: hidden; border-radius: 999px; background: var(--stage-line); }
.stage-meter i {
  display: block; width: 100%; height: 100%; border-radius: inherit; transform-origin: left;
  background: linear-gradient(90deg,var(--stage-violet),var(--stage-accent));
  transition: transform .45s cubic-bezier(.2,0,.25,1);
}
.stage-meter[data-mode="indeterminate"] i { width: 32%; animation: meter-sweep 1.5s ease-in-out infinite; }
.stage-meter[data-mode="unknown"] i { width: 0; opacity: 0; }
.stage:not([data-state="running"]) .stage-meter i { background: var(--stage-accent); }
.time, .hint, .now, .notes p { margin: 0; color: var(--stage-muted); font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; }
.time { font-variant-numeric: tabular-nums; }
.steps {
  list-style: none; display: grid; gap: 0; margin: 2px 0 0; padding: 4px 0;
  border-block: 1px solid var(--stage-line);
}
.steps li { display: grid; grid-template-columns: 17px minmax(0,1fr) auto; align-items: center; column-gap: 10px; min-height: 34px; padding: 4px 0; box-sizing: border-box; }
.steps .dot {
  display: grid; place-content: center; justify-self: center; width: 12px; height: 12px;
  box-sizing: border-box; border: 1px solid var(--stage-muted); border-radius: 50%;
}
.steps .name { color: var(--stage-muted); font: 500 14px/1.5 var(--sans,sans-serif); overflow-wrap: anywhere; }
.step-status { color: var(--stage-muted); font-size: 12px; white-space: nowrap; }
.steps li[data-s="now"] .name { color: var(--stage-text); font-weight: 750; }
.steps li[data-s="now"] .step-status { color: var(--stage-accent); }
.steps li[data-s="now"] .dot { border-color: var(--stage-accent); }
.steps li[data-s="now"] .dot::after { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--stage-accent); }
.steps li[data-s="done"] .dot { border-color: transparent; }
.steps li[data-s="done"] .dot::after { content: ""; width: 9px; height: 5px; border: solid var(--success,#64e8b6); border-width: 0 0 2px 2px; transform: rotate(-45deg); }
.steps li:is([data-s="failed"],[data-s="stopped"]) :is(.dot,.step-status) { color: var(--stage-accent); border-color: var(--stage-accent); }
.stage[data-state="failed"] .now { color: var(--stage-accent); white-space: pre-line; }
.now { color: var(--stage-text); }
.notes { display: grid; gap: 8px; }
.stage-announcement { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
@keyframes meter-sweep { from { transform: translateX(-110%); } to { transform: translateX(315%); } }
@media (prefers-reduced-motion: reduce) {
  .stage-meter i { transition: none; }
  .stage-meter[data-mode="indeterminate"] i { animation: none; transform: none; }
}
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
    return (f'<div class="track{" track-main" if main else ""}" data-audio-state="loading">{name}'
            f'<audio controls preload="metadata" controlslist="nodownload noplaybackrate" src="{url}"></audio>'
            f'<span class="track-status" role="status" aria-live="polite"><span class="track-message">正在读取音频信息…</span>'
            f'<button type="button" class="track-retry" hidden>重试</button></span>'
            f'<a class="save" href="{url}" download="{html.escape(download_name, quote=True)}">'
            f'{"下载成品" if main else "下载"}</a></div>')


def _result_html(item: dict, title: str) -> str:
    """What is shown when a cover is finished: the cover, and the separate stems behind a fold."""
    if not item.get("mix"):
        return ""
    parts = [("已转换的声部" if item.get("duet_male") else "只有我的人声", item.get("converted"), "翻唱人声"),
             ("原唱的人声", item.get("vocals"), "原唱人声"),
             ("伴奏", item.get("instrumental"), "伴奏")]
    for key, name in (("duet_male", "男声原分轨"), ("duet_female", "女声原分轨"),
                      ("duet_male_converted", "转换后的男声部"), ("duet_female_converted", "转换后的女声部"),
                      ("duet_preserved", "保留的原唱部分")):
        if item.get(key):
            parts.append((name, item[key], name))
    rows = "".join(_track(file, f"{title} {suffix}.wav", label) for label, file, suffix in parts if file)
    fold = f'<details class="parts"><summary>分开听</summary>{rows}</details>' if rows else ""
    return (f'<section class="result"><div class="result-heading"><span class="eyebrow">制作完成</span>'
            f'<h3>{html.escape(title)}</h3><p>成品已保存到作品库</p></div>'
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
        # Failure and cancellation may arrive with only a message. Keep the
        # last measured position so a refreshed page shows where the run ended.
        if finished and item.get("progress", {}).get("state") in {"stopped", "failed"}:
            previous = task.item.get("progress", {})
            position = {key: previous[key] for key in ("steps", "step", "overall", "elapsed") if key in previous}
            item = {**item, "progress": {**position, **item["progress"]}}
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


def _cover_view(snapshot, blocked_reason=""):
    """Progress/form/stage/result/again/start/stop/empty/timer updates."""
    # Gradio consumes update dictionaries in place during postprocessing.
    # Never share an update object between components or successive responses.
    def hidden():
        return gr.update(visible=False)

    def shown():
        return gr.update(visible=True)

    start_button = _cover_start_button(blocked_reason, snapshot)
    if snapshot is None:
        return (dict(IDLE), gr.skip(), hidden(), gr.update(value="", visible=False), hidden(),
                start_button, hidden(), shown(), gr.update(active=False))
    title, item, finished, cancelling = snapshot
    progress = {**item.get("progress", {}), "title": title, "run": item.get("run")}
    if not finished:
        progress["state"] = "running"
        progress["stopping"] = cancelling
        progress["hint"] = "正在释放资源…" if cancelling else "可切换页面，成品将保存至作品库。"
        if cancelling:
            progress["detail"] = "正在停止"
        return (progress, gr.skip(), shown(), gr.update(value="", visible=False), hidden(),
                start_button,
                gr.update(visible=True, interactive=not cancelling), hidden(), gr.update(active=True))
    state = progress.get("state")
    if state == "done":
        progress["message"] = "翻唱好了"
        return (progress, gr.skip(), shown(), gr.update(value=_result_html(item, title), visible=True),
                gr.update(visible=True, value="继续制作"), start_button, hidden(), hidden(), gr.update(active=False))
    if state == "stopped":
        return (progress, gr.skip(), shown(), gr.update(value="", visible=False), hidden(),
                start_button, hidden(), hidden(), gr.update(active=False))
    progress["message"] = progress.get("error") or progress.get("message") or "没有完成，请重试。"
    return (progress, gr.skip(), shown(), gr.update(value="", visible=False),
            gr.update(visible=True, value="继续制作"), start_button, hidden(), hidden(), gr.update(active=False))


def _start_cover(song, model, pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate,
                 current_task="", duet_mode="off", duet_source="auto", duet_second_model=None,
                 duet_male_stem=None, duet_female_stem=None, duet_swap=False, duet_dereverb=False):
    """Start a task immediately; /cover returns a task token, not a long stream."""
    if not song:
        raise gr.Error("先选一首歌。", title="还不能开始", print_exception=False)
    voices = _voices()
    voice_notice, reason = _cover_requirements(song, model, voices)
    if not reason:
        voice_notice, reason = _duet_requirements(duet_mode, duet_source, duet_second_model,
                                                 duet_male_stem, duet_female_stem, voices)
    if reason:
        raise gr.Error(voice_notice, title="还不能开始", print_exception=False)
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
        if duet_mode != "off":
            second = duet_second_model if duet_mode == "both" and duet_second_model != SAME_VOICE else None
            manual = duet_source == "stems"
            arguments += (duet_mode, bool(duet_swap) and not manual, second,
                          duet_male_stem if manual else None, duet_female_stem if manual else None, duet_dereverb)
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


def _poll_cover(current_task="", blocked_reason=""):
    return _cover_view(_task_snapshot(current_task), blocked_reason)


def _stop_cover(current_task="", blocked_reason=""):
    with _COVER_TASKS_LOCK:
        task = _COVER_TASKS.get(current_task) if isinstance(current_task, str) else None
        if task is not None and not task.finished:
            task.cancel.set()
    return _poll_cover(current_task, blocked_reason)


def _again(current_task="", blocked_reason=""):
    """Back to the form, with the song and the settings as they were."""
    snapshot = _task_snapshot(current_task)
    if snapshot is not None and not snapshot[2]:
        view = _cover_view(snapshot, blocked_reason)
        return (*view[:-1], current_task, view[-1])
    view = _cover_view(None, blocked_reason)
    return (*view[:-1], "", view[-1])


def _cover_preparation(song, model):
    voice_ready = bool(model) and not _is_demo_voice(model, [])
    if not song:
        status = "等待原曲"
    elif not voice_ready:
        status = "等待个人音色"
    else:
        status = "可以开始制作"
    ready = "true" if song and voice_ready else "false"
    return (f'<div class="monitor-readiness" role="status" data-ready="{ready}">'
            f'<span>本次任务</span><b>{status}</b></div>')


def _duet_controls(mode, source):
    duet = mode != "off"
    manual = source == "stems"
    first_label = {"male": "男声使用的音色", "female": "女声使用的音色", "both": "男声使用的音色"}
    return (gr.update(visible=duet), gr.update(visible=duet and manual),
            gr.update(visible=duet and not manual), gr.update(visible=mode == "both"),
            gr.update(label=first_label.get(mode, "我的音色")),
            gr.update(visible=not duet), gr.update(visible=not duet),
            gr.update(info="只调整转换声部的空间感；0 为干声。" if duet else "1 接近原曲的空间感，0 不添加混响。"),
            gr.update(visible=duet))


def _duet_second_voices(current):
    choices = [("与男声使用同一音色", SAME_VOICE), *[pair for pair in _voices() if not _is_demo_voice(pair[1], [pair])]]
    selected = current if any(value == current for _, value in choices) else SAME_VOICE
    return gr.update(choices=choices, value=selected)


@lru_cache(maxsize=8)
def _recent_waveform(path: str, mtime_ns: int, file_size: int):
    """Cache by file revision and sample at most 72 short windows across the mix."""
    with sf.SoundFile(path) as audio:
        if not audio.frames or audio.channels > 8:
            return ""
        duration = audio.frames / audio.samplerate
        bins = min(72, audio.frames)
        levels = []
        for index in range(bins):
            start = index * audio.frames // bins
            end = (index + 1) * audio.frames // bins
            count = min(end - start, 1024)
            audio.seek(start + (end - start - count) // 2)
            samples = audio.read(count, dtype="int16").astype("float64")
            levels.append(float(np.sqrt(np.square(samples).mean())))

    loudest = max(levels) or 1
    width = bins * 4
    segments = []
    for index, level in enumerate(levels):
        height = 21 * level / loudest
        if height >= 0.5:
            segments.append(f"M{4 * index + 2},{24 - height:.1f}v{2 * height:.1f}")
    seconds = int(duration)
    end_time = f"{seconds // 60}:{seconds % 60:02d}" if seconds else f"{duration:.2f} 秒"
    return (f'<figure class="monitor-waveform"><div class="wave-scrubber">'
            f'<svg viewBox="0 0 {width} 48" preserveAspectRatio="none" '
            f'aria-hidden="true"><path class="wave-base" d="M0,24H{width}"/>'
            f'<path class="wave-levels" d="{" ".join(segments)}"/></svg>'
            f'<input class="wave-seek" type="range" min="0" max="{duration:.3f}" step="0.1" '
            f'value="0" disabled aria-label="最近作品试听位置"></div>'
            f'<figcaption><span class="wave-time">0:00</span><span>时长 {end_time}</span></figcaption></figure>')


def _recent_cover_preview():
    works = library.list_works(pipeline.OUTPUT)
    if not works:
        return ('<section class="monitor-recent"><h3>最近完成</h3>'
                '<p class="monitor-no-work">暂无作品可试听</p></section>')
    latest = works[0]
    title = latest["title"]
    date = str(latest.get("created_at") or "")[:16].replace("T", " ")
    player = _track(latest["cover"], f"{title} 翻唱.wav")
    try:
        stat = Path(latest["cover"]).stat()
        waveform = _recent_waveform(latest["cover"], stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, sf.LibsndfileError):
        waveform = ""
    return ('<section class="monitor-recent"><div class="monitor-meta"><h3>最近完成</h3>'
            f'<span>{html.escape(date)}</span></div><p class="monitor-title">{html.escape(title)}</p>'
            f'{waveform}{player}</section>')


def _cover_tab(cover_tab=None):
    voices = _voices()
    refine_ready = pipeline.refine_available()
    selected_voice = _default_voice(voices)
    voice_notice, initial_reason = _cover_requirements(None, selected_voice, voices)
    blocked_reason = gr.State(initial_reason)
    defaults = _default_cover_settings(refine_ready)
    with gr.Row(elem_id="cover-workspace", equal_height=False):
        with gr.Column(elem_id="cover-controls", elem_classes=["studio-panel", "tool-card", "input-panel"], scale=5, min_width=320):
            gr.HTML('<div class="panel-head"><h2 class="panel-title">歌曲与音色</h2></div>', container=False)
            with gr.Column(elem_id="cover-form") as form:
                with gr.Column(elem_id="song-field"):
                    gr.HTML('<div class="song-field-label"><span>原曲文件</span>'
                            '<small>MP3 / FLAC / WAV / M4A</small></div>', container=False)
                    song = gr.File(label="原曲文件", file_types=["audio"], type="filepath",
                                   show_label=False, elem_id="song-drop", height=84)
                mode = gr.Dropdown(label="翻唱范围", choices=DUET_MODES, value="off",
                                   filterable=False, interactive=True, elem_id="duet-mode")
                with gr.Column(visible=False, min_width=0, elem_id="duet-routing") as routing:
                    source = gr.Radio(label="对唱人声来源", choices=[("从歌曲自动分离", "auto"), ("使用已有分轨", "stems")],
                                      value="auto", elem_id="duet-source")
                swap = gr.Checkbox(label="交换自动分出的男、女声部", value=False, visible=False, elem_id="duet-swap",
                                   info="如果试听发现分反了，可交换后重新制作。重叠、同音或音色相近时可能串音。")
                with gr.Column(visible=False, min_width=0, elem_id="duet-uploads") as uploads:
                    gr.HTML('<p class="field-note">两轨需与原曲从同一时间起点导出，保留休止处的静音和完整曲长。</p>',
                            container=False)
                    with gr.Row():
                        male_stem = gr.File(label="原男声分轨", file_types=[".wav", ".flac"], type="filepath", height=95)
                        female_stem = gr.File(label="原女声分轨", file_types=[".wav", ".flac"], type="filepath", height=95)
                with gr.Row(elem_id="voice-picker-row"):
                    model = gr.Dropdown(label="我的音色", choices=voices, value=selected_voice,
                                        filterable=False, interactive=True, elem_id="voice-pick", scale=5,
                                        info=None if voices else "还没有音色？到“我的声音”录音并训练，完成后即可选择。")
                    refresh_voice = gr.Button("刷新音色", elem_id="refresh-voices", size="sm", scale=0, min_width=84)
                second = gr.Dropdown(label="女声使用的音色", choices=_duet_second_voices(SAME_VOICE)["choices"],
                                     value=SAME_VOICE, filterable=False, visible=False, interactive=True,
                                     elem_id="duet-second-voice")
                voice_notice_view = gr.HTML(_studio_alert_html(voice_notice), visible=bool(voice_notice),
                                            elem_id="voice-required", container=False)
                with gr.Row(elem_id="cover-settings-actions"):
                    start = gr.Button(initial_reason, variant="primary", elem_id="start-cover",
                                      size="lg", scale=0, min_width=136, interactive=False)
                    gr.HTML('<p class="local-note">成品自动保存到作品库</p>', container=False)
                with gr.Accordion("声音设置", open=False, elem_id="more-settings"):
                    with gr.Column(elem_id="basic-voice-settings", min_width=0):
                        gr.HTML('<h3 class="settings-group-title">人声适配</h3>', container=False)
                        pitch_mode = gr.Radio(choices=RANGE_CHOICES, value=defaults[0], label="音域适配",
                                              info="自动模式只在原唱明显超出你的录音音域时，将人声降低一个八度；伴奏保持原调。")
                        backing = gr.Radio(choices=BACKING_CHOICES, value=defaults[1], label="和声处理",
                                           info="保留原唱和声通常更自然，主唱仍会转换为你的声音。")
                        clarity = gr.Slider(0, 1, value=defaults[2], step=0.05, label="人声清晰度",
                                            info="调高可补强较弱的字头和气声；齿音偏刺时调低。")
                    with gr.Accordion("混音与精细调节", open=False, elem_id="advanced-voice-settings"):
                        gr.HTML('<p class="field-note">只在需要时调整。人声和伴奏响度分别控制，0 表示保持原曲响度。</p>',
                                container=False)
                        reverb = gr.Slider(0, 2, value=defaults[3], step=0.1, label="混响", info="1 接近原曲的空间感，0 不添加混响。")
                        with gr.Row():
                            vocal_gain = gr.Slider(-12, 12, value=defaults[4], step=0.5, label="人声音量（dB）", info="0 保持原曲人声响度。")
                            music_gain = gr.Slider(-12, 12, value=defaults[5], step=0.5, label="伴奏音量（dB）", info="0 保持原曲伴奏响度。")
                        refine = gr.Checkbox(label="精细处理主唱", value=defaults[6], interactive=refine_ready,
                                             info="先分离主唱并去除混响，减少和声干扰。处理时间稍长。" if refine_ready
                                             else "请先安装主唱分离与去混响模型。")
                        duet_dry = gr.Checkbox(label="转声前去除对唱声部的混响", value=False, visible=False,
                                               info="字尾混在一起时可试用。仅处理要翻唱的声部，需要重新制作。")
                        index_rate = gr.Slider(0, 1, value=defaults[7], step=0.05, label="音色检索强度",
                                               info="默认 0.25。字音含混时可试 0.10 并重新翻唱；音色相似度可能变化。")
                    reset_settings = gr.Button("恢复默认设置", elem_id="reset-cover", size="sm")
                settings_summary = gr.Markdown(
                    _settings_summary(*defaults), elem_id="cover-settings-summary")
        with gr.Column(elem_id="cover-output", elem_classes=["studio-panel", "tool-card", "side-panel"], scale=6, min_width=320):
            gr.HTML('<h2 class="panel-title">制作与试听</h2>', elem_id="monitor-heading", container=False)
            with gr.Column(elem_id="cover-empty") as empty:
                preparation = gr.HTML(_cover_preparation(None, selected_voice), elem_id="monitor-preparation",
                                      container=False)
                recent = gr.HTML(_recent_cover_preview, elem_id="monitor-audition", container=False)
            with gr.Column(visible=False, elem_id="cover-stage") as stage:
                progress = _progress_view("cover-progress")
            # Keep control visibility independent from the stage container;
            # a parent update must not remount its child with stale defaults.
            stop = gr.Button("停止任务", elem_id="stop-cover", size="sm", visible=False)
            result = gr.HTML("", visible=False, elem_id="cover-result", container=False)
            again = gr.Button("继续制作", elem_id="again", visible=False)
    quiet = dict(queue=False, show_progress="hidden", api_name=False)
    gr.on([mode.change, source.change], _duet_controls, inputs=[mode, source],
          outputs=[routing, uploads, swap, second, model, backing, refine, reverb, duet_dry], **quiet)
    if cover_tab is not None:
        cover_tab.select(_recent_cover_preview, outputs=[recent], queue=False, show_progress="hidden", api_name=False)
    task_id = gr.BrowserState("", storage_key="aicoversong-cover-task")
    def input_status(song, model, task_id, mode, source, second, male, female):
        updates = _cover_input_change(song, model, task_id, mode, source, second, male, female)
        return (*updates, _cover_preparation(song, model) if not updates[1] else
                f'<div class="monitor-readiness" role="status"><span>本次任务</span><b>{html.escape(updates[1])}</b></div>')

    ready_inputs = [song, model, task_id, mode, source, second, male_stem, female_stem]
    ready_outputs = [voice_notice_view, blocked_reason, start, preparation]
    refresh_voice.click(_refresh_voices, inputs=[model], outputs=[model], queue=False, show_progress="hidden").then(
        _duet_second_voices, inputs=[second], outputs=[second], **quiet).then(
        input_status, inputs=ready_inputs, outputs=ready_outputs, **quiet)
    gr.on([control.change for control in (song, model, mode, source, second, male_stem, female_stem)],
          input_status, inputs=ready_inputs, outputs=ready_outputs, **quiet)
    settings_inputs = [pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate]
    for control in [*settings_inputs, mode]:
        control.change(_settings_summary, inputs=[*settings_inputs, mode], outputs=[settings_summary],
                      queue=False, show_progress="hidden")
    reset_settings.click(lambda mode: (*defaults, _settings_summary(*defaults, mode), False),
                         inputs=[mode], outputs=settings_inputs + [settings_summary, duet_dry],
                         queue=False, show_progress="hidden")
    outputs = [progress, form, stage, result, again, start, stop, empty]
    poll = gr.Timer(1.0, active=False)
    start.click(_start_cover,
                inputs=[song, model, pitch_mode, backing, clarity, reverb, vocal_gain, music_gain, refine, index_rate, task_id,
                        mode, source, second, male_stem, female_stem, swap, duet_dry],
                outputs=outputs + [task_id, poll], queue=False, trigger_mode="multiple", show_progress="hidden",
                api_name="cover")
    poll.tick(_poll_cover, inputs=[task_id, blocked_reason], outputs=outputs + [poll], queue=False,
              show_progress="hidden", api_name="cover_status")
    # BrowserState contains only this browser's opaque task token. It restores
    # the status after a page refresh without exposing another session's job.
    task_id.change(_poll_cover, inputs=[task_id, blocked_reason], outputs=outputs + [poll], queue=False,
                   show_progress="hidden", api_name=False)
    stop.click(_stop_cover, inputs=[task_id, blocked_reason], outputs=outputs + [poll], queue=False,
               show_progress="hidden", api_name="stop_cover")
    again.click(_again, inputs=[task_id, blocked_reason], outputs=outputs + [task_id, poll], queue=False,
                show_progress="hidden", api_name=False).then(
                    _recent_cover_preview, outputs=[recent], queue=False, show_progress="hidden", api_name=False)
    return model, second


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
const notice = element.querySelector('.mic-device-warning');
const message = notice.querySelector('.mic-device-message');
const retry = notice.querySelector('button');
const devices = navigator.mediaDevices;
const reasons = {
  NotFoundError: '没有可用的麦克风。请连接设备，并在 Windows 声音设置中启用输入设备。',
  NotAllowedError: '麦克风访问受限。请检查浏览器站点权限和 Windows 麦克风隐私设置。',
  SecurityError: '请用本机地址或 HTTPS 打开页面，并检查浏览器的麦克风权限。',
  NotReadableError: '麦克风无法读取。请检查设备连接，或关闭占用麦克风的程序后重试。',
  AbortError: '录音启动中断。请检查麦克风连接后重试。',
  UnknownError: '暂时无法检查麦克风。请检查设备和浏览器权限后重试。'
};
function show(reason) {
  message.textContent = reason ? (reasons[reason] || reasons.UnknownError) : '';
  notice.hidden = !reason;
  retry.removeAttribute('aria-busy');
  retry.disabled = false;
}
if (!devices || !devices.getUserMedia) {
  show('SecurityError');
  retry.hidden = true;
  return;
}

// Stop subscriptions from an earlier mount if Gradio reconstructs this tab.
window.__coverMicSubscriptions?.abort();
const controller = new AbortController();
window.__coverMicSubscriptions = controller;
const options = {signal: controller.signal};
const recorder = document.getElementById('cover-mic-recorder');
let preferredInput = '';
function labelRecorder() {
  const select = recorder?.querySelector('select.mic-select');
  if (select) {
    if (preferredInput && Array.from(select.options).some(item => item.value === preferredInput)) {
      select.value = preferredInput;
    }
    Array.from(select.options).forEach((item, index) => {
      if (!item.label.trim()) item.label = '输入设备 ' + (index + 1);
    });
    const name = select.selectedOptions[0]?.label || '选择麦克风';
    select.title = name;
    select.setAttribute('aria-label', '输入设备');
    select.dataset.selectedMic = select.value === preferredInput ? preferredInput : '';
  }
  const pause = recorder?.querySelector('.pause-button');
  if (pause) {
    pause.title = '暂停录音';
    pause.setAttribute('aria-label', pause.title);
  }
}
if (recorder) {
  labelRecorder();
  recorder.addEventListener('change', event => {
    if (event.target.matches('select.mic-select')) {
      preferredInput = event.target.value;
      labelRecorder();
    }
  }, options);
  const observer = new MutationObserver(changes => {
    if (changes.some(change => Array.from(change.addedNodes).some(node =>
      node.nodeType === 1 && (node.matches('select.mic-select, option, .pause-button') ||
      node.querySelector('select.mic-select, .pause-button'))))) labelRecorder();
  });
  observer.observe(recorder, {childList: true, subtree: true});
  controller.signal.addEventListener('abort', () => observer.disconnect(), {once: true});
}
let revision = 0;
let permission;
async function inspect() {
  const check = ++revision;
  retry.disabled = true;
  retry.setAttribute('aria-busy', 'true');
  let reason = '';
  try {
    if (permission?.state === 'denied') reason = 'NotAllowedError';
    else if (!(await devices.enumerateDevices()).some(device => device.kind === 'audioinput')) reason = 'NotFoundError';
  } catch (error) {
    reason = error.name || 'UnknownError';
  }
  if (check === revision && !controller.signal.aborted) show(reason);
}

if (!devices.__coverRawMicrophone) {
  const original = devices.getUserMedia.bind(devices);
  devices.getUserMedia = async (constraints) => {
    if (!constraints?.audio) return original(constraints);
    const audio = constraints.audio === true ? {} : {...constraints.audio};
    // In Gradio 6.20 the selector lists inputs but Wavesurfer starts with {audio:true}.
    const selected = document.querySelector('#cover-mic-recorder select.mic-select')?.dataset.selectedMic;
    if (selected && !audio.deviceId) audio.deviceId = {exact: selected};
    try {
      const stream = await original({...constraints, audio: {...audio,
        noiseSuppression: false, echoCancellation: false, autoGainControl: false}});
      window.dispatchEvent(new CustomEvent('cover-mic-capture', {detail: ''}));
      return stream;
    } catch (error) {
      window.dispatchEvent(new CustomEvent('cover-mic-capture', {detail: error.name || 'UnknownError'}));
      throw error;
    }
  };
  devices.__coverRawMicrophone = true;
}
window.addEventListener('cover-mic-capture', event => {
  revision++;
  show(event.detail);
}, options);
devices.addEventListener('devicechange', inspect, options);
retry.addEventListener('click', inspect, options);
(async () => {
  try { permission = await navigator.permissions?.query({name: 'microphone'}); }
  catch { /* Some browsers support capture but omit microphone from the Permissions API. */ }
  if (controller.signal.aborted) return;
  permission?.addEventListener('change', inspect, options);
  if (!revision) await inspect();
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
    return (gr.update(interactive=enabled, value="试听"), gr.update(interactive=enabled),
            gr.update(interactive=enabled), gr.update(interactive=enabled),
            _studio_alert_html("录音已就绪，可以试听、保存或重录。", "info") if enabled else "")


def _recording_started():
    return _studio_alert_html("正在录音。读完或唱完后点停止，再试听确认。", "info")


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
            snapshot = {**snapshot, "hint": "训练期间请保持网页和服务窗口开启。"}
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
    gr.HTML('<header class="page-hero"><h1>录音与<span>训练</span></h1></header>',
            container=False, apply_default_css=False)
    summary = gr.HTML(_voice_summary(), elem_id="voice-tally", container=False)
    with gr.Row(elem_id="voice-workspace", equal_height=False):
        with gr.Column(elem_id="voice-script", scale=6, min_width=320):
            prompt = gr.HTML(recorder.prompt_markdown(initial), elem_id="cover-recording-prompt", container=False)
        with gr.Column(elem_id="voice-recorder", elem_classes=["studio-panel", "tool-card", "recorder-card"], scale=4, min_width=300):
            gr.HTML('<div class="panel-head"><h3 class="panel-title">录音控制</h3></div>', container=False)
            recording = gr.Audio(label="麦克风录音", sources=["microphone"], type="filepath", format="wav",
                                 recording=False, interactive=True, editable=False, elem_id="cover-mic-recorder", buttons=[])
            gr.HTML('<div class="studio-alert mic-device-warning" data-tone="warning" hidden>'
                    '<span class="mic-device-message" role="status"></span>'
                    '<button class="mic-retry" type="button">重新检测</button></div>',
                    elem_id="mic-device-status", container=False, apply_default_css=False, js_on_load=RAW_MICROPHONE_JS)
            with gr.Row(elem_id="take-actions"):
                preview = gr.Button("试听", elem_id="take-preview", size="sm", interactive=False)
                retake = gr.Button("重录", elem_id="take-retake", size="sm", interactive=False)
                clear = gr.Button("清除本次录音", elem_id="take-clear", size="sm", interactive=False)
            with gr.Row(elem_id="take-next"):
                skip = gr.Button("跳过本条", elem_id="take-skip")
                save = gr.Button("保存并继续", elem_id="take-save", variant="primary", interactive=False)
            feedback = gr.HTML("", elem_id="take-feedback", container=False)
            with gr.Accordion("录音小贴士", open=False, elem_id="record-tips"):
                gr.Markdown(
                    "- 安静房间，嘴离麦克风约一拳，保持距离。\n"
                    "- 不放伴奏，不开混响、变声或声卡降噪。\n"
                    "- 浏览器录音已请求关闭通话降噪。若保存时提示“纯数字静音”，请检查系统或声卡的降噪设置。\n"
                    "- 读完即停，避免长时间留白。\n"
                    "- 清唱保持舒适音域，高音专项也不要勉强或喊叫。\n"
                    "- 已保存的片段会保留，下次可继续补录。"
                )
    gr.HTML('<h2 class="section-title">训练音色</h2>'
            '<p class="section-lead">用已保存的录音训练个人音色。耗时取决于显卡与录音量，训练期间请保持电脑唤醒。'
            '已有音色会保留。</p>', container=False)
    with gr.Column(elem_id="train-form") as train_form:
        terms = gr.Checkbox(label="这些录音都是我自己的声音，我不会拿它冒充别人", value=False,
                            info="开始前请阅读并遵守 [Applio 使用条款](https://github.com/IAHispano/Applio/blob/3.7.1/TERMS_OF_USE.md)。")
        train = gr.Button("开始训练", variant="primary", interactive=False, elem_id="start-train")
    train_progress = _progress_view("train-progress")
    with gr.Column(visible=False, elem_id="train-stage") as train_stage:
        cancel = gr.Button("停止训练", size="sm", elem_id="stop-train", scale=0)
    train_gate = gr.HTML(_training_gate(False), elem_id="training-gate", container=False)

    recording.clear(_clear_take, inputs=[count],
                    outputs=[recording, prompt, preview, retake, clear, save, feedback], queue=False)
    recording.start_recording(_recording_started, outputs=[feedback], queue=False, show_progress="hidden")
    recording.change(_take_action_state, inputs=[recording], outputs=[preview, retake, clear, save, feedback],
                     queue=False, show_progress="hidden")
    preview.click(fn=None, js="""() => document.querySelector('#cover-mic-recorder .play-pause-button')?.click()""",
                  queue=False, show_progress="hidden")
    recording.play(lambda: gr.update(value="暂停试听"), outputs=[preview], queue=False, show_progress="hidden")
    gr.on([recording.pause, recording.stop], lambda: gr.update(value="试听"),
          outputs=[preview], queue=False, show_progress="hidden")
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


def _history_rows(rows, query="", order="newest", *, limit=60):
    needle = (query or "").strip().casefold()
    if needle:
        rows = [row for row in rows if needle in row["title"].casefold() or needle in Path(row["source"]).name.casefold()]
    if order == "oldest":
        rows = list(reversed(rows))
    elif order == "title":
        rows = sorted(rows, key=lambda row: row["title"].casefold())
    return rows[:limit]


@lru_cache(maxsize=256)
def _cover_duration(path: str, mtime_ns: int, size: int) -> str:
    """Read just the WAV header; replace cached durations when the file changes."""
    with sf.SoundFile(path) as audio:
        seconds = int(audio.frames / audio.samplerate)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _history_duration(path):
    try:
        stat = Path(path).stat()
        return _cover_duration(path, stat.st_mtime_ns, stat.st_size)
    except (ValueError, OSError, sf.LibsndfileError):
        return ""


# Lucide static v0.468.0 (ISC), see licenses/Lucide-ISC.txt.
_WORK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" width="16" height="16">'
_PLAY_ICON = _WORK_ICON + '<polygon points="6 3 20 12 6 21 6 3"/></svg>'
_PAUSE_ICON = _WORK_ICON + '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>'
_DOWNLOAD_ICON = _WORK_ICON + '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>'


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
        run_id = html.escape(row["run_id"], quote=True)
        url = html.escape(_file_url(row["cover"]), quote=True)
        duration = _history_duration(row["cover"])
        duration_markup = f'<span class="work-duration" aria-label="时长 {duration}">{duration}</span>' if duration else ""
        items.append(
            f'<li class="history-entry" data-selected="{active_attr}"><div class="history-entry-heading">'
            f'<div class="meta"><b title="{name}">{name}</b><span>{html.escape(when)}</span>{duration_markup}{tag}</div>'
            f'<button type="button" class="history-pick" data-work="{run_id}" '
            f'aria-label="管理：{name}" aria-controls="history-inspector" aria-expanded="{active_attr}">'
            f'{"收起" if active else "管理"}</button></div>'
            f'<div class="work-actions"><button type="button" class="work-play" data-preview="{run_id}" '
            f'aria-controls="library-audio" aria-label="试听：{name}">{_PLAY_ICON}{_PAUSE_ICON}<span>试听</span></button>'
            f'<a class="save" href="{url}" download="{name} 翻唱.wav" aria-label="下载：{name}">{_DOWNLOAD_ICON}下载</a></div></li>')
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


# This component lives outside the regenerated card list so sorting and searching
# don't interrupt playback. Only an explicit user action assigns a media URL.
_LIBRARY_PLAYER = """
<section class="library-deck" hidden aria-label="作品试听">
  <div class="deck-info"><span class="deck-state" role="status"></span><strong class="deck-title"></strong></div>
  <audio id="library-audio" controls preload="none" aria-label="作品试听"></audio>
  <button type="button" class="deck-retry" hidden>重试</button>
  <button type="button" class="deck-close" aria-label="关闭试听" title="关闭试听">关闭</button>
</section>
"""

_LIBRARY_PLAYER_JS = """
const deck = element.querySelector('.library-deck');
const library = element.closest('#history-collection');
const view = library.querySelector('#history-view');
const player = deck.querySelector('audio');
const name = deck.querySelector('.deck-title');
const status = deck.querySelector('.deck-state');
const retry = deck.querySelector('.deck-retry');
let active = '';

function updateCards() {
  view.querySelectorAll('.work-play').forEach((button) => {
    const card = button.closest('.history-entry');
    const selected = button.dataset.preview === active;
    const playing = selected && !player.paused && !player.ended;
    const label = playing ? '暂停' : '试听';
    const title = card.querySelector('.meta b').textContent;
    if (selected) name.textContent = title;
    card.dataset.previewActive = String(selected);
    button.dataset.playing = String(playing);
    button.setAttribute('aria-label', label + '：' + title);
    const caption = button.querySelector('span');
    if (caption.textContent !== label) caption.textContent = label;
  });
}
function announce(next) {
  if (!active) return;
  status.textContent = next;
  retry.hidden = next !== '读取失败';
  updateCards();
}
function start() {
  const requested = active;
  player.play().catch((error) => {
    if (requested === active && error.name !== 'AbortError') announce('读取失败');
  });
}
view.addEventListener('click', (event) => {
  const button = event.target.closest('.work-play');
  if (!button) return;
  if (active === button.dataset.preview && !player.paused) { player.pause(); return; }
  if (active !== button.dataset.preview) {
    active = button.dataset.preview;
    const card = button.closest('.history-entry');
    player.src = card.querySelector('.work-actions a').href;
    deck.hidden = false;
    announce('正在载入');
  } else if (player.error) {
    player.load();
    announce('正在载入');
  }
  start();
});
player.addEventListener('playing', () => announce('正在试听'));
player.addEventListener('waiting', () => { if (!player.paused) announce('正在载入'); });
player.addEventListener('pause', () => { if (!player.ended && !player.error && player.paused) announce('已暂停'); });
player.addEventListener('ended', () => announce('试听结束'));
player.addEventListener('error', () => { if (player.hasAttribute('src')) announce('读取失败'); });
retry.addEventListener('click', () => { announce('正在载入'); player.load(); start(); });
function closePlayer(returnFocus) {
  const sourceButton = Array.from(view.querySelectorAll('.work-play')).find((b) => b.dataset.preview === active);
  active = '';
  player.pause();
  player.removeAttribute('src');
  player.load();
  deck.hidden = true;
  updateCards();
  if (returnFocus) {
    (sourceButton || document.querySelector('#history-search textarea') || document.querySelector('#history-refresh'))
      ?.focus({preventScroll: true});
  }
}
deck.querySelector('.deck-close').addEventListener('click', () => closePlayer(true));
deck.addEventListener('studio-delete-works', (event) => {
  if (event.detail.includes(active)) closePlayer(false);
});
// Gradio replaces the list's HTML when a filter, name or sort order changes.
const observer = new MutationObserver((changes) => {
  if (!active || !changes.some(change => Array.from(change.addedNodes).some(n => n.nodeType === 1))) return;
  updateCards();
});
observer.observe(view, {childList:true, subtree:true});
"""


def _work_choices(rows):
    return [(f'{row["title"]} · {row.get("created_at", "")[:16].replace("T", " ") or "日期未知"} · {row["run_id"][-4:]}',
             row["run_id"]) for row in rows]


def _history_refresh(query="", order="newest", selected=None, bulk_ids=None, *, message="", tone="success"):
    # One inventory per refresh: the visible list and inspector always use the same snapshot.
    all_rows = library.list_works(pipeline.OUTPUT)
    matching = _history_rows(all_rows, query, order, limit=None)
    rows = matching[:60]
    work = next((row for row in rows if row["run_id"] == selected), None)
    selected = work["run_id"] if work else None
    requested = set(bulk_ids or [])
    bulk_ids = [row["run_id"] for row in matching if row["run_id"] in requested]
    return (_history_html(rows, len(all_rows), selected), selected,
            _studio_alert_html(message, tone) if message else "", gr.update(visible=bool(all_rows)),
            *_history_selection(work),
            gr.update(choices=_work_choices(matching), value=bulk_ids, label=f"全选 / 清空当前筛选 · {len(matching)} 首"),
            gr.update(visible=bool(matching)),
            *_batch_actions(bulk_ids), gr.update(value=None, visible=False))


def _batch_actions(ids):
    count = len(ids or [])
    suffix = f"（{count} 首）" if count else ""
    return (gr.update(interactive=bool(count), value="打包下载" + suffix),
            gr.update(interactive=bool(count), value="删除所选" + suffix))


def _history_delete(ids, query="", order="newest", selected=None, bulk_ids=None):
    if not ids:
        return gr.skip()
    deleted, failed = library.delete_selected(ids, output_root=pipeline.OUTPUT)
    message = f"已删除 {len(deleted)} 首作品。"
    if failed:
        details = "；".join(f"{run_id}：{reason}" for run_id, reason in list(failed.items())[:3])
        message += f"有 {len(failed)} 首未能删除。{details}"
    return _history_refresh(query, order, selected, bulk_ids, message=message, tone="warning" if failed else "success")


def _confirm_delete_js(bulk):
    # Gradio State values are server-side. Use the identity rendered with the
    # inspector title or the checkbox IDs captured by this click.
    source = "bulkIds.slice()" if bulk else "(heading ? [heading.dataset.work] : [])"
    return """(_unused, query, order, selected, bulkIds) => {
  const heading = document.querySelector('#history-selection-note [data-work]');
  const ids = SOURCE;
  if (ids.length) {
    const title = ids.length === 1 && heading?.dataset.work === ids[0] ? heading.querySelector('h3').textContent : '';
    const target = title ? '《' + title + '》' : '所选的 ' + ids.length + ' 首作品';
    if (window.confirm('永久删除' + target + '及其分轨？此操作无法撤销。')) {
      document.querySelector('#history-player .library-deck')?.dispatchEvent(new CustomEvent('studio-delete-works', {detail: ids}));
      return [ids, query, order, selected, bulkIds];
    }
  }
  return [[], query, order, selected, bulkIds];
}""".replace("SOURCE", source)


def _history_export(ids):
    from gradio.processing_utils import save_file_to_cache
    from gradio.utils import get_upload_folder

    try:
        with tempfile.TemporaryDirectory(prefix="aicoversong-export-") as directory:
            archive = Path(directory) / "AIcoversong-works.zip"
            library.export_selected(ids, archive, output_root=pipeline.OUTPUT)
            cached = save_file_to_cache(archive, get_upload_folder())
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc), title="未能打包作品", print_exception=False) from exc
    return gr.update(value=cached, visible=True)


def _history_selection(work):
    if not work:
        return (gr.update(visible=False), "", "", 0, 0, 0.75, 1.0, "original")
    job = work["job"]
    duet = job.get("duet_mode", "off") != "off"
    note = (f'<div class="history-section-heading" data-work="{html.escape(work["run_id"], quote=True)}">'
            '<span class="eyebrow">作品管理</span>'
            f'<h3>{html.escape(work["title"])}</h3></div>')
    return (gr.update(visible=True), note, work["title"],
            gr.update(value=float(job.get("vocal_gain", 0)), label="翻唱声部音量（dB）" if duet else "人声音量（dB）"),
            float(job.get("instrumental_gain", 0)), float(job.get("clarity", 0.75)),
            gr.update(value=float(job.get("reverb", 1.0)), info="只给转换后的声部添加混响。" if duet else None),
            gr.update(value=job.get("backing", "original"), visible=not duet))


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
    selected = gr.State(None)
    confirmed_ids = gr.JSON(value=[], visible=False)
    gr.HTML('<header class="page-hero"><h1>我的<span>作品库</span></h1></header>',
            container=False, apply_default_css=False)
    with gr.Row(elem_id="history-toolbar"):
        with gr.Row(elem_id="history-filter-controls", visible=bool(rows)) as filters:
            search = gr.Textbox(label="搜索作品", placeholder="输入歌名或源文件名", show_label=True,
                                scale=5, elem_id="history-search")
            order = gr.Dropdown(choices=[("最新制作", "newest"), ("最早制作", "oldest"), ("按名称", "title")],
                                value="newest", label="排序", filterable=False, scale=2, elem_id="history-order")
        refresh = gr.Button("刷新作品", size="sm", scale=0, min_width=92, elem_id="history-refresh")
    status = gr.HTML("", elem_id="history-feedback", container=False)
    with gr.Accordion("批量管理", open=False, visible=bool(rows), elem_id="history-bulk") as bulk_panel:
        bulk_ids = gr.CheckboxGroup(choices=_work_choices(rows), value=[], label=f"全选 / 清空当前筛选 · {len(rows)} 首",
                                    show_select_all=True, interactive=True, elem_id="history-bulk-selection")
        with gr.Row(elem_id="history-bulk-actions"):
            export = gr.Button("打包下载", size="sm", interactive=False, scale=0, min_width=120)
            bulk_delete = gr.Button("删除所选", elem_id="history-bulk-delete", size="sm",
                                    interactive=False, scale=0, min_width=126)
        archive = gr.File(label="下载成品 ZIP", type="filepath", interactive=False, visible=False,
                          elem_id="history-bulk-download")
    with gr.Row(elem_id="history-workspace", equal_height=False):
        with gr.Column(scale=2, min_width=350, elem_id="history-collection"):
            view = gr.HTML(_history_html(rows[:60], len(rows)), elem_id="history-view", container=False,
                           js_on_load=_HISTORY_PICK_JS)
            gr.HTML(_LIBRARY_PLAYER, elem_id="history-player", container=False, apply_default_css=False,
                    js_on_load=_LIBRARY_PLAYER_JS)
        with gr.Column(scale=1, min_width=280, visible=False,
                       elem_id="history-inspector", elem_classes=["studio-panel", "tool-card", "inspector-card"]) as inspector:
            selection_note = gr.HTML("", elem_id="history-selection-note", container=False)
            title = gr.Textbox(label="作品名称", value="", placeholder="输入新的作品名称", elem_id="history-title")
            with gr.Row(elem_id="history-basic-actions"):
                rename = gr.Button("保存名称", size="sm", elem_id="history-rename")
                delete = gr.Button("删除作品", size="sm", elem_id="history-delete")
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
    refresh_outputs = [view, selected, status, filters, *selection_outputs,
                       bulk_ids, bulk_panel, export, bulk_delete, archive]
    quiet = dict(queue=False, show_progress="hidden")
    for event in (search.input, order.input, refresh.click, history_tab.select):
        event(_history_refresh, inputs=[search, order, selected, bulk_ids], outputs=refresh_outputs, **quiet)
    view.click(_history_pick, inputs=None, outputs=[selected, status, *selection_outputs], **quiet)
    bulk_ids.input(lambda ids: (*_batch_actions(ids), gr.update(visible=False, value=None)),
                   inputs=[bulk_ids], outputs=[export, bulk_delete, archive], **quiet)
    export.click(_history_export, inputs=[bulk_ids], outputs=[archive],
                 queue=True, trigger_mode="once", show_progress="minimal", show_progress_on=[export])
    for button, bulk in ((delete, False), (bulk_delete, True)):
        button.click(_history_delete, js=_confirm_delete_js(bulk), inputs=[confirmed_ids, search, order, selected, bulk_ids],
                     outputs=refresh_outputs, queue=True, trigger_mode="once", show_progress="hidden", api_name=False)
    rename.click(_history_rename, inputs=[selected, title, search, order], outputs=refresh_outputs, **quiet)
    remix.click(_history_remix, inputs=[selected, remix_vocal, remix_music, remix_clarity, remix_reverb,
                                        remix_backing, search, order], outputs=refresh_outputs,
                queue=True, trigger_mode="once", show_progress="minimal", show_progress_on=[status])


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def applio_plugin():
    with gr.Tabs(elem_id="main-tabs") as tabs:
        with gr.Tab("创作工作台", id="cover", elem_id="studio-cover") as cover_tab:
            gr.HTML('<header class="page-hero"><h1 class="cover-title">用自己的声音  <span>唱喜欢的歌</span></h1></header>',
                    container=False, apply_default_css=False)
            model, second = _cover_tab(cover_tab)
        with gr.Tab("我的声音", id="voice", elem_id="studio-voice"):
            training = _voice_tab()
        with gr.Tab("作品库", id="history", elem_id="studio-history") as history_tab:
            _history_tab(history_tab)
    training.then(_refresh_voices, outputs=[model], queue=False, show_progress="hidden").then(
        _duet_second_voices, inputs=[second], outputs=[second], queue=False, show_progress="hidden", api_name=False)
    return tabs
