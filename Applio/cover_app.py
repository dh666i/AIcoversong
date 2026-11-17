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
CSS = (pathlib.Path(now_dir) / "assets" / "ui" / "cover.css").read_text(encoding="utf-8")
THEME = gr.themes.Base(
    primary_hue="emerald", secondary_hue="slate", neutral_hue="slate",
    font=["Microsoft YaHei UI", "PingFang SC", "sans-serif"], font_mono=["Consolas", "monospace"],
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
</script>"""

_WAVE_MARK = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 10v4m4-8v12m5-15v18m5-15v12m4-8v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
_GITHUB_MARK = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .8a11.4 11.4 0 0 0-3.6 22.2c.6.1.8-.2.8-.6v-2.2c-3.2.7-3.9-1.3-3.9-1.3-.5-1.3-1.3-1.7-1.3-1.7-1.1-.8.1-.8.1-.8 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.4 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.4-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.1 1.2a10.8 10.8 0 0 1 5.8 0c2.1-1.5 3.1-1.2 3.1-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.1v3.3c0 .4.2.7.8.6A11.4 11.4 0 0 0 12 .8Z"/></svg>'
_SIGNAL_HEIGHTS = [10, 18, 30, 16, 39, 49, 25, 39, 55, 31, 46, 23, 13, 35, 50, 58, 34, 21, 42, 28, 46, 57, 38, 23, 31, 15, 24, 13, 6]
_SIGNAL = ''.join(f'<i style="--h:{height}px;--o:{.38 + height / 100:.2f}"></i>' for height in _SIGNAL_HEIGHTS)
MASTHEAD = f'''<header id="masthead">
  <div class="brand" aria-label="AIcoversong"><span class="brand-mark">{_WAVE_MARK}</span><span class="brand-name"><span>AI</span>coversong</span><span class="brand-label">个人声音工作室</span></div>
  <nav class="head-links" aria-label="项目链接"><span class="local-badge"><i aria-hidden="true"></i>本地创作</span><a class="repo-link" href="https://github.com/dh666i/AIcoversong" target="_blank" rel="noopener noreferrer">{_GITHUB_MARK}<span>GitHub</span></a></nav>
</header>
<section class="studio-hero" aria-label="欢迎使用">
  <div class="hero-copy"><p class="studio-eyebrow">YOUR VOICE. YOUR SONG.</p><h1>把喜欢的歌，唱成<em>自己的声音。</em></h1><p>选一首歌，留住熟悉的旋律。让下一次播放，属于你的音色。</p></div>
  <div class="hero-signal" aria-hidden="true"><div class="signal-bars">{_SIGNAL}</div><div class="signal-caption"><span>AIcoversong</span><span>VOICE STUDIO</span></div></div>
</section>'''
FOOTER = '''<footer class="studio-footer"><span>AIcoversong · 用自己的声音，唱喜欢的歌。</span><div class="footer-links"><span>基于 <a href="https://github.com/IAHispano/Applio" target="_blank" rel="noopener noreferrer">Applio / RVC</a></span><a href="https://github.com/dh666i/AIcoversong#readme" target="_blank" rel="noopener noreferrer">使用文档</a><a href="https://github.com/dh666i/AIcoversong/issues" target="_blank" rel="noopener noreferrer">反馈问题</a></div></footer>'''

# delete_cache: the page copies every result it shows into cache/gradio. The originals stay in
# output/, so copies older than a day are removed once an hour (they had grown to 7 GB).
with gr.Blocks(title="AIcoversong · 个人声音工作室", delete_cache=(3600, 86400), analytics_enabled=False) as page:
    with gr.Column(elem_id="page"):
        gr.HTML(MASTHEAD, container=False)
        applio_plugin()
        gr.HTML(FOOTER, container=False)


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
