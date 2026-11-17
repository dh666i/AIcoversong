"""Local UI-only harness: synthetic audio and fake jobs, never audio models.

Run with the project's environment. Uses port 6971 and isolated temporary data.
"""
from pathlib import Path
import json
import os
import sys
import time
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'Applio'
DATA = ROOT / 'cache' / 'ui-smoke'
DATA.mkdir(parents=True, exist_ok=True)
os.chdir(APP)
sys.path.insert(0, str(APP))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = str(DATA / 'gradio')

from tabs.plugins.installed.CoverStudio import plugin as ui

ui.pipeline.OUTPUT = DATA / 'output'
ui.recorder.DATASET = DATA / 'recordings'
ui.recorder.RECORD_INFO = DATA / 'metadata'
ui._voices = lambda: [('演示音色 · 仅用于界面验证', 'demo.pth')]
ui._index_for = lambda model: ''
ui.pipeline.refine_available = lambda: True

tone = DATA / 'UI-test.wav'
if not tone.exists():
    import math
    import struct
    with wave.open(str(tone), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b''.join(struct.pack('<h', round(800 * math.sin(2 * math.pi * 220 * n / 16000))) for n in range(32000)))


def fake_cover(song, *args):
    run = ui.pipeline.OUTPUT / ('run_20261117_120000_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    steps = ['分离人声与伴奏', '分出主唱、去掉混响', '换成你的声音', '处理人声并合成']
    began = time.monotonic()
    for n in range(16):
        if 'fail' in Path(song).name.lower():
            raise RuntimeError('演示错误：用于验证失败后可重新制作。')
        yield {'run': str(run), 'progress': {'state': 'running', 'steps': steps, 'step': n // 4,
                 'step_fraction': (n % 4) / 4, 'overall': n / 16, 'elapsed': time.monotonic()-began}}
        time.sleep(.4)
    import shutil
    cover = run / 'cover.wav'
    shutil.copyfile(tone, cover)
    item = {'run': str(run), 'mix': str(cover), 'converted': str(cover), 'vocals': str(cover), 'instrumental': str(cover),
            'progress': {'state': 'done', 'steps': steps, 'step': 4, 'overall': 1, 'elapsed': time.monotonic()-began}}
    (run/'job.json').write_text(json.dumps({'original_song': 'UI-test.wav'}), encoding='utf-8')
    (run/'result.json').write_text(json.dumps(item), encoding='utf-8')
    yield item


ui.pipeline.run_cover = fake_cover

def no_training(*args):
    raise RuntimeError('UI harness never trains a model.')

ui.trainer.train_voice = no_training
import cover_app

if __name__ == '__main__':
    cover_app.page.queue(default_concurrency_limit=4).launch(server_name='127.0.0.1', server_port=6971,
        allowed_paths=[str(DATA)], theme=cover_app.THEME, css=cover_app.CSS, head=cover_app.HEAD,
        footer_links=[], show_error=True, quiet=True, inbrowser=False)
