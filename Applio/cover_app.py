"""AIcoversong — the local voice studio web page.

Only the three Chinese tabs of tabs/plugins/installed/CoverStudio are shown.
The full English Applio interface is untouched and still starts with
``python app.py`` (see 使用说明.md); nothing here imports its pages, so this
page opens faster and never downloads anything.

Start it with 启动翻唱.bat (tools/start.ps1), which sets the cache folders.
"""
import argparse
import os
import pathlib
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
os.environ.setdefault("APPLIO_CONFIG_FILE", CONFIG_PATH)
if not os.path.exists(CONFIG_PATH):
    import shutil

    shutil.copy(os.path.join(now_dir, "assets", "config_template.json"), CONFIG_PATH)

import logging

import gradio as gr

from tabs.plugins.installed.CoverStudio.plugin import applio_plugin

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Windows: a browser tab closed in the middle of a download raises this during shutdown.
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _original_lost = _pe._ProactorBasePipeTransport._call_connection_lost

    def _quiet_lost(self, exc):
        try:
            _original_lost(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _quiet_lost


def _keep_existing_upload(files, destinations) -> None:
    """Stand-in for gradio.route_utils.move_uploaded_files_to_cache.

    Gradio stores an upload under the hash of its content, and the microphone recorder uploads
    every take twice (once for "change", once for "stop_recording"). On Windows the second copy
    cannot be renamed over the first, so Gradio rewrites the file in place, at the very moment
    the browser is fetching it to draw the waveform. The fetch then fails and the take cannot
    be played (seen here: "0:00 / 0:00" and ERR_CONTENT_LENGTH_MISMATCH). Same hash and same
    size means the file is already there, so the second copy is simply dropped.
    """
    import shutil

    for file, destination in zip(files, destinations):
        try:
            if os.path.isfile(destination) and os.path.getsize(destination) == os.path.getsize(file):
                os.remove(file)
                continue
        except OSError:
            pass
        shutil.move(file, destination)


try:
    import gradio.routes as _gradio_routes

    if hasattr(_gradio_routes, "move_uploaded_files_to_cache"):
        _gradio_routes.move_uploaded_files_to_cache = _keep_existing_upload
except Exception:   # a different Gradio: the page works as before, only without this safeguard
    pass

ROOT = pathlib.Path(now_dir).parent
UI_ASSETS = pathlib.Path(now_dir) / "assets" / "ui"
STYLESHEETS = ("cover.css", "cover-controls.css", "cover-media.css", "cover-library.css")
CSS = "\n".join((UI_ASSETS / name).read_text(encoding="utf-8") for name in STYLESHEETS)
THEME = gr.themes.Base(
    primary_hue="lime", secondary_hue="cyan", neutral_hue="zinc",
    font=["system-ui", "Microsoft YaHei", "sans-serif"], font_mono=["Consolas", "monospace"],
)
# Gradio words its own buttons ("Record", "Stop", "Download" ...) in the browser's language.
# This page is Chinese whatever the browser is set to, so tell Gradio the browser is Chinese.
HEAD = """<meta name="color-scheme" content="dark">
<script>
try {
  Object.defineProperty(navigator, 'language', {get: () => 'zh-CN', configurable: true});
  Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh'], configurable: true});
} catch (e) {}
document.documentElement.lang = 'zh-CN';
// one player at a time: starting one pauses the others
document.addEventListener('play', (event) => {
  document.querySelectorAll('audio').forEach((audio) => { if (audio !== event.target) audio.pause(); });
}, true);
const setAudioStatus = (audio, state, message) => {
  const track = audio.closest('.track');
  if (!track) return;
  track.dataset.audioState = state;
  const status = track.querySelector('.track-message');
  if (status) status.textContent = message;
};
const clearAudioTimer = (audio) => {
  if (audio.__coverLoadTimer) window.clearTimeout(audio.__coverLoadTimer);
  audio.__coverLoadTimer = 0;
};
const watchAudioLoad = (audio) => {
  clearAudioTimer(audio);
  audio.__coverLoadTimer = window.setTimeout(() => {
    if (audio.readyState < 1 && !audio.error) {
      setAudioStatus(audio, 'idle', '点击播放后继续读取，或下载音频文件');
    }
  }, 6000);
};
const bindAudioStates = () => {
  document.querySelectorAll('.track audio:not([data-state-bound])').forEach((audio) => {
    if (audio.dataset.stateBound === 'true') return;
    audio.dataset.stateBound = 'true';
    if (audio.readyState >= 1) setAudioStatus(audio, 'ready', '');
    else watchAudioLoad(audio);
    audio.addEventListener('loadstart', () => {
      setAudioStatus(audio, 'loading', '正在读取音频信息…');
      watchAudioLoad(audio);
    });
    audio.addEventListener('loadedmetadata', () => {
      clearAudioTimer(audio);
      setAudioStatus(audio, 'ready', '');
    });
    audio.addEventListener('canplay', () => {
      clearAudioTimer(audio);
      setAudioStatus(audio, 'ready', '');
    });
    audio.addEventListener('error', () => {
      clearAudioTimer(audio);
      setAudioStatus(audio, 'error', '音频暂时无法读取。请重试，或用右侧按钮下载文件。');
      const retry = audio.closest('.track')?.querySelector('.track-retry');
      if (retry) retry.hidden = false;
    });
  });
  document.querySelectorAll('.track-retry').forEach((button) => {
    if (button.dataset.bound === 'true') return;
    button.dataset.bound = 'true';
    button.addEventListener('click', () => {
      const audio = button.closest('.track')?.querySelector('audio');
      if (!audio) return;
      button.hidden = true;
      audio.preload = 'metadata';
      setAudioStatus(audio, 'loading', '正在重新读取音频信息…');
      watchAudioLoad(audio);
      audio.load();
    });
  });
};
const audioObserver = new MutationObserver(bindAudioStates);
const observeAudio = () => {
  if (document.body) audioObserver.observe(document.body, {childList: true, subtree: true});
  bindAudioStates();
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', observeAudio, {once: true});
} else {
  observeAudio();
}
const controlButtons = '.gr-accordion > button.label-wrap, .tab-like-container > .reset-button, #song-drop button.center';
const titleFields = '#page input[role="combobox"], #history-search textarea, #history-title textarea';
function disableTitleCorrections(field) {
  field.spellcheck = false;
  field.setAttribute('autocorrect', 'off');
  field.setAttribute('autocapitalize', 'off');
}
// Gradio opens a readonly choice on focus. Escape can leave it focused and closed.
document.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  const trigger = event.target.closest?.('#page .wrap-inner');
  const field = trigger?.querySelector('input[role="combobox"][readonly]:enabled');
  if (!field) return;
  event.preventDefault();
  if (field.getAttribute('aria-expanded') === 'true') {
    field.blur();
  } else {
    if (field === document.activeElement) field.blur();
    field.focus({preventScroll: true});
  }
});
function labelControl(button) {
  if (button.classList.contains('label-wrap')) {
    button.setAttribute('aria-expanded', String(button.classList.contains('open')));
    const panel = button.nextElementSibling;
    if (panel?.matches('[data-testid="accordion-content"]') && button.parentElement.id) {
      panel.id = button.parentElement.id + '-content';
      button.setAttribute('aria-controls', panel.id);
    }
    button.querySelector('.icon')?.setAttribute('aria-hidden', 'true');
  } else if (button.classList.contains('reset-button')) {
    const slider = button.closest('.wrap');
    const label = slider?.querySelector('.head label [data-testid="block-info"]')?.textContent.trim();
    if (!label) return;
    button.title = '恢复' + label + '默认值';
    button.setAttribute('aria-label', button.title);
    slider.querySelector('input[type="number"]')?.setAttribute('aria-label', label + '数值');
    slider.querySelector('input[type="range"]')?.setAttribute('aria-label', label);
  } else {
    button.setAttribute('aria-label', '选择音频文件，也可以拖放文件');
  }
}
function labelNewControls(element) {
  if (!(element instanceof Element)) return;
  if (element.matches(controlButtons)) labelControl(element);
  if (element.matches(titleFields)) disableTitleCorrections(element);
  element.querySelectorAll(controlButtons).forEach(labelControl);
  element.querySelectorAll(titleFields).forEach(disableTitleCorrections);
}
const observeControlLabels = () => {
  labelNewControls(document.body);
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'childList') record.addedNodes.forEach(labelNewControls);
      else if (record.target.matches('.gr-accordion > button.label-wrap')) labelControl(record.target);
    }
  }).observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['class']});
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', observeControlLabels, {once: true});
} else {
  observeControlLabels();
}
</script>""" + "<script>" + (UI_ASSETS / "cover-motion.js").read_text(encoding="utf-8") + "</script>"

_WAVE_MARK = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 10v4m4-8v12m5-15v18m5-15v12m4-8v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
MASTHEAD = f'''<header class="studio-bar">
  <span class="studio-brand"><span class="studio-symbol">{_WAVE_MARK}</span>AIcoversong</span>
  <a class="engine-location" href="https://github.com/dh666i/AIcoversong" target="_blank" rel="noopener noreferrer">
    <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6m0-6L10 14"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>
    GitHub 仓库
  </a>
</header>'''
FOOTER = '''<footer class="studio-footer"><span>声音引擎 · <a href="https://github.com/IAHispano/Applio" target="_blank" rel="noopener noreferrer">Applio / RVC</a></span><nav class="footer-links" aria-label="项目资源"><a href="https://github.com/dh666i/AIcoversong#readme" target="_blank" rel="noopener noreferrer">使用文档</a><a href="https://github.com/dh666i/AIcoversong/issues" target="_blank" rel="noopener noreferrer">反馈问题</a><a href="https://github.com/dh666i/AIcoversong" target="_blank" rel="noopener noreferrer">GitHub</a></nav></footer>'''

# delete_cache: the page copies every result it shows into cache/gradio. The originals stay in
# output/, so copies older than a day are removed once an hour (they had grown to 7 GB).
with gr.Blocks(title="AIcoversong · 个人声音工作室", delete_cache=(3600, 86400), analytics_enabled=False) as page:
    with gr.Column(elem_id="page"):
        gr.HTML(MASTHEAD, elem_id="studio-masthead", container=False, apply_default_css=False)
        applio_plugin()
        gr.HTML(FOOTER, elem_id="studio-resources", container=False, apply_default_css=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIcoversong · 个人声音工作室")
    parser.add_argument("--port", type=int, default=6969)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="open the browser when the page is ready")
    args, _ = parser.parse_known_args()
    page.queue(default_concurrency_limit=4)
    print(f"AIcoversong 已启动：http://{args.server_name}:{args.port}", flush=True)
    print("用的时候请保留这个窗口；关掉它，或双击“关闭翻唱.bat”，就会停止。", flush=True)
    page.launch(
        favicon_path="assets/ui/aicoversong.svg", inbrowser=args.open, server_name=args.server_name, server_port=args.port,
        allowed_paths=[str(ROOT / "output"), str(ROOT / "recordings" / "MyVoice")],
        theme=THEME, css=CSS, head=HEAD, footer_links=[], show_error=True, quiet=True,
    )


if __name__ == "__main__":
    main()
