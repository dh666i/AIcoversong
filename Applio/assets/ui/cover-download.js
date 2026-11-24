const dialog = element.querySelector('dialog');
const format = dialog.querySelector('select');
const start = dialog.querySelector('.export-start');
const status = dialog.querySelector('.export-status');
let link;
let revision = 0;

function reset() {
  revision++;
  status.textContent = '';
  start.disabled = false;
  start.textContent = '下载 ' + format.value.toUpperCase();
}

// Delegate to the page since switching tabs, filtering, or completing a cover replaces its players.
const page = element.closest('#page') || element.closest('.gradio-container');
page.addEventListener('click', event => {
  const target = event.target.closest('a[data-export-run]');
  if (!target || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  link = target;
  dialog.querySelector('.export-title').textContent = link.download.replace(/\.wav$/i, '');
  reset();
  dialog.showModal();
  format.focus();
});
format.addEventListener('change', reset);
dialog.querySelector('.export-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => {
  revision++;
  link?.focus({preventScroll: true});
});

start.addEventListener('click', async () => {
  const current = ++revision;
  const chosen = format.value;
  const item = link;
  start.disabled = true;
  status.textContent = '正在准备 ' + chosen.toUpperCase() + '…';
  try {
    let url = item.href;
    if (chosen !== 'wav') {
      const result = await server.prepare_audio({
        run: item.dataset.exportRun, track: item.dataset.exportTrack, format: chosen,
      });
      if (current !== revision || !dialog.open) return;
      if (!result?.url) throw new Error(result?.error || '未能生成下载文件，请重试。');
      url = result.url;
    }
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = item.download.replace(/\.wav$/i, '.' + chosen);
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    status.textContent = '已开始下载 ' + chosen.toUpperCase() + '。';
  } catch (err) {
    if (current === revision) status.textContent = err.message || '下载失败，请重试。';
  } finally {
    if (current === revision) start.disabled = false;
  }
});
