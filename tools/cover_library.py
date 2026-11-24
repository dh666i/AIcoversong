"""Safe local works-library helpers for AICover.

The library treats finished ``output/run_*`` folders as works, stores titles in
``work.json``, and creates a separate folder for each new mix. Deleting a work
removes its entire run folder.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any
import uuid
from zipfile import ZIP_STORED, ZipFile

if __package__:
    from .audio_export import EXPORT_FORMATS, check_format, write_audio
else:
    from audio_export import EXPORT_FORMATS, check_format, write_audio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "output"
_RUN_NAME = re.compile(r"^run_[A-Za-z0-9][A-Za-z0-9_-]*$")
_TITLE_LIMIT = 200
_DELETE_LOCK = Lock()
_EXPORT_TRACKS = {
    "mix": ("cover.wav", "翻唱"),
    "converted": ("my_vocals.wav", "翻唱人声"),
    "vocals": ("vocals.wav", "原唱人声"),
    "instrumental": ("instrumental.wav", "伴奏"),
    "lead": ("lead_vocals.wav", "主唱"),
    "backing": ("backing_vocals.wav", "和声"),
    "duet_male": ("duet_male.wav", "男声原分轨"),
    "duet_female": ("duet_female.wav", "女声原分轨"),
    "duet_male_converted": ("duet_male_converted.wav", "转换后的男声部"),
    "duet_female_converted": ("duet_female_converted.wav", "转换后的女声部"),
    "duet_preserved": ("duet_preserved.wav", "保留的原唱部分"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _root(output_root: str | Path | None) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    root = root.expanduser()
    _assert_root(root)
    root = root.resolve(strict=False)
    if root.name.casefold() == ".trash":
        raise ValueError("只能操作当前作品库。")
    return root


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _assert_root(root: Path) -> None:
    if root.is_symlink() or root.is_junction():
        raise ValueError("作品库根目录不能是链接目录。")
    if root.exists() and not root.is_dir():
        raise ValueError(f"作品库根目录不是文件夹：{root}")


def _run_from(value: Any, root: Path) -> Path:
    """Accept only an immediate child run directory, rejecting links before resolving."""
    _assert_root(root)
    if isinstance(value, dict):
        value = value.get("path") or value.get("run") or value.get("run_id")
    if value is None:
        raise ValueError("没有指定作品。")
    raw = Path(str(value))
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.parent != root or not _RUN_NAME.fullmatch(candidate.name):
        raise ValueError("只能操作作品库中的 run_* 作品目录。")
    if candidate.is_symlink() or candidate.is_junction():
        raise ValueError("不能操作链接目录。")
    if not candidate.is_dir() or candidate.resolve().parent != root:
        raise ValueError("作品目录不存在或不是普通文件夹。")
    return candidate


def _safe_file(run: Path, name: str, *, required: bool = False) -> Path | None:
    path = run / name
    if path.is_symlink() or not _inside(path, run):
        if required:
            raise ValueError(f"作品文件不安全：{name}")
        return None
    if not path.is_file() or path.stat().st_size <= 0:
        if required:
            raise ValueError(f"作品缺少必要文件：{name}")
        return None
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取作品元数据：{path.name}") from exc
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_title(source: str | Path | None) -> str:
    title = Path(str(source or "")).stem.strip()
    while True:
        trimmed = re.sub(r"\s*\[[^\[\]]*\]\s*$", "", title)
        if trimmed == title:
            break
        title = trimmed
    return title or "未命名作品"


def _metadata(run: Path) -> dict[str, Any]:
    job = _read_json(_safe_file(run, "job.json", required=True))
    custom_path = _safe_file(run, "work.json")
    custom = _read_json(custom_path) if custom_path else {}
    source = job.get("original_song") or job.get("song") or ""
    created = custom.get("created_at") or _created_from_name(run.name) or ""
    return {
        "run_id": run.name,
        "path": str(run),
        "title": custom.get("title") or _source_title(source),
        "source": str(source),
        "created_at": created,
        "updated_at": custom.get("updated_at") or created,
        "parent_run": custom.get("parent_run") or job.get("parent_run"),
        "cover": str(_safe_file(run, "cover.wav")) if _safe_file(run, "cover.wav") else None,
        "job": job,
        "metadata": custom,
    }


def _created_from_name(name: str) -> str:
    match = re.match(r"^run_(\d{8})_(\d{6})_", name)
    if not match:
        return ""
    try:
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return ""
    return dt.isoformat(timespec="seconds")


def list_works(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return finished works, newest directory first."""
    root = _root(output_root)
    _assert_root(root)
    if not root.is_dir():
        return []
    works: list[dict[str, Any]] = []
    for run in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not run.is_dir() or run.is_symlink() or run.is_junction() or not _RUN_NAME.fullmatch(run.name):
            continue
        try:
            _safe_file(run, "cover.wav", required=True)
            works.append(_metadata(run))
        except (OSError, ValueError):
            continue
    return works


def get_work(run_id_or_path: str | Path | dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(output_root)
    run = _run_from(run_id_or_path, root)
    _safe_file(run, "cover.wav", required=True)
    return _metadata(run)


def rename_work(work: str | Path | dict[str, Any], title: str, output_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(output_root)
    run = _run_from(work, root)
    title = " ".join(str(title or "").split())
    if not title:
        raise ValueError("作品名称不能为空。")
    if len(title) > _TITLE_LIMIT:
        raise ValueError(f"作品名称不能超过 {_TITLE_LIMIT} 个字符。")
    existing = _safe_file(run, "work.json")
    custom = _read_json(existing) if existing else {}
    custom.update({"title": title, "updated_at": _now()})
    _write_json(run / "work.json", custom)
    return _metadata(run)


def delete_work(work: str | Path | dict[str, Any], output_root: str | Path | None = None) -> None:
    """Remove one finished run, leaving its catalog entries intact on a busy stem."""
    with _DELETE_LOCK:
        _delete_finished_run(_run_from(work, _root(output_root)))


def _delete_finished_run(run: Path) -> None:
    cover = _safe_file(run, "cover.wav", required=True)
    job = _safe_file(run, "job.json", required=True)
    custom = _safe_file(run, "work.json")
    catalog = [job, custom] if custom else [job]
    for child in run.iterdir():
        if child == cover or child in catalog:
            continue
        if child.is_symlink():
            child.unlink()
        elif child.is_junction():
            child.rmdir()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # cover.wav and job.json are both needed for a failed deletion to stay
    # selectable. In particular Windows media playback may hold cover.wav open.
    removed_metadata = {}
    try:
        for path in catalog:
            contents = path.read_bytes()
            path.unlink()
            removed_metadata[path] = contents
        cover.unlink()
    except OSError:
        for path, contents in removed_metadata.items():
            path.write_bytes(contents)
        raise
    run.rmdir()


def _selected_ids(run_ids: list[str]) -> list[str]:
    if not isinstance(run_ids, list) or not run_ids:
        raise ValueError("请先勾选作品。")
    if any(not isinstance(item, str) or not _RUN_NAME.fullmatch(item) for item in run_ids):
        raise ValueError("选择中含有无效的作品编号，请刷新后重试。")
    return list(dict.fromkeys(run_ids))


def delete_selected(run_ids: list[str], *, output_root=None) -> tuple[list[str], dict[str, str]]:
    """Report each work separately if a file is busy or another session deleted it."""
    deleted = []
    failed = {}
    for run_id in _selected_ids(run_ids):
        try:
            delete_work(run_id, output_root)
        except (OSError, ValueError) as exc:
            failed[run_id] = str(exc)
        else:
            deleted.append(run_id)
    return deleted, failed


def _export_title(title: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")
    # Stay within a filesystem component even for titles containing CJK or emoji.
    clean = clean[:120].encode("utf-8")[:180].decode("utf-8", errors="ignore").rstrip(" .") or "翻唱"
    return "_" + clean if PureWindowsPath(clean).is_reserved() else clean


def _publish_download(staged: Path, destination: Path) -> None:
    # Both operations fail if another request already published this filename.
    if os.name == "nt":
        staged.rename(destination)
    else:
        os.link(staged, destination)


def list_export_tracks(work, *, output_root=None) -> dict[str, tuple[Path, str]]:
    """Return the mix and named stems that still exist for a saved work."""
    run = _run_from(work, _root(output_root))
    available = {}
    for key, (filename, label) in _EXPORT_TRACKS.items():
        path = _safe_file(run, filename, required=(key == "mix"))
        if path is not None:
            available[key] = (path, label)
    return available


def export_track(work, track: str, destination_dir: Path, *, audio_format: str = "wav", output_root=None) -> Path:
    """Export one named track from a finished work into a caller-owned directory."""
    check_format(audio_format)
    if not isinstance(track, str) or track not in _EXPORT_TRACKS:
        raise ValueError("请选择作品中的有效音轨。")
    metadata = get_work(work, output_root)
    filename, label = _EXPORT_TRACKS[track]
    source = _safe_file(Path(metadata["path"]), filename, required=True)
    target = Path(destination_dir) / f"{_export_title(metadata['title'])} {label}.{audio_format}"
    if target.exists():
        raise FileExistsError(target)
    with TemporaryDirectory(dir=destination_dir) as temp:
        staged = Path(temp) / ("track." + audio_format)
        write_audio(source, staged)
        _publish_download(staged, target)
    return target


def export_selected(run_ids: list[str], destination: Path, *, output_root=None, audio_format: str = "wav") -> None:
    """Write only finished mixes, with portable, unique filenames inside the ZIP."""
    check_format(audio_format)
    works = [get_work(run_id, output_root) for run_id in _selected_ids(run_ids)]
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "works.zip"
        converted = Path(temp) / ("track." + audio_format)
        with ZipFile(staged, "x", compression=ZIP_STORED, allowZip64=True) as archive:
            for index, work in enumerate(works, 1):
                name = f"{index:03d} - {_export_title(work['title'])}.{audio_format}"
                if audio_format == "wav":
                    archive.write(work["cover"], name)
                else:
                    write_audio(Path(work["cover"]), converted)
                    archive.write(converted, name)
                    converted.unlink()
        _publish_download(staged, destination)


def _new_run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        candidate = root / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("无法创建新的作品版本目录。")


def _numeric(value: Any, default: float, low: float, high: float, label: str) -> float:
    result = default if value is None else float(value)
    if not low <= result <= high:
        raise ValueError(f"{label}必须在 {low:g} 到 {high:g} 之间。")
    return result


def remix_work(work: str | Path | dict[str, Any], *, output_root: str | Path | None = None,
               vocal_gain_db: float | None = None, music_gain_db: float | None = None,
               backing_gain_db: float | None = None, clarity: float | None = None,
               dynamics: float | None = None, reverb: float | None = None,
               backing_mode: str | None = None) -> dict[str, Any]:
    """Create a new, fully treated mix from an existing work's stems."""
    root = _root(output_root)
    parent = _run_from(work, root)
    parent_meta = _metadata(parent)
    job = parent_meta["job"]
    duet_mode = job.get("duet_mode", "off")
    mode = backing_mode or job.get("backing", "original")
    if mode not in {"original", "convert", "remove"}:
        raise ValueError("和声处理方式无效。")
    output = _new_run(root)
    try:
        params = {
            "vocal_gain_db": _numeric(vocal_gain_db, float(job.get("vocal_gain", 0)), -24, 12, "人声音量"),
            "music_gain_db": _numeric(music_gain_db, float(job.get("instrumental_gain", 0)), -24, 12, "伴奏音量"),
            "backing_gain_db": _numeric(backing_gain_db, 0.0, -24, 12, "和声音量"),
            "clarity": _numeric(clarity, float(job.get("clarity", 0.75)), 0, 1, "人声清晰度"),
            "dynamics": _numeric(dynamics, 0.6, 0, 1, "人声动态"),
            "reverb": _numeric(reverb, float(job.get("reverb", 1.0)), 0, 2, "混响量"),
        }
        if duet_mode != "off":
            if mode != "original":
                raise ValueError("对唱作品重混会保留原声部，不能修改原曲和声模式。")
            if __package__:
                from . import duet_mix
            else:
                import duet_mix
            sources = {key: _safe_file(parent, filename, required=True)
                       for key, filename in duet_mix.remix_inputs(duet_mode).items()}
            files = duet_mix.retain_stems(sources, output)
            source_info = _safe_file(parent, "duet_result.json") or _safe_file(parent, "result.json")
            separator = _read_json(source_info).get("duet_separator") if source_info else None
            report = duet_mix.mix_duet(files, duet_mode, output,
                                       auto_separated=duet_mix.automatically_separated(job, separator),
                                       **{key: value for key, value in params.items() if key != "backing_gain_db"})
        else:
            if __package__:
                from .vocal_mix import mix_cover
            else:
                from vocal_mix import mix_cover
            instrumental = _safe_file(parent, "instrumental.wav", required=True)
            converted = _safe_file(parent, "converted_vocals.wav") or _safe_file(parent, "my_vocals.wav", required=True)
            reference_dry = _safe_file(parent, "lead_dry.wav") or _safe_file(parent, "vocals.wav", required=True)
            backing = _safe_file(parent, "backing_vocals.wav")
            backing_converted = _safe_file(parent, "converted_backing.wav")
            if mode == "convert" and (backing is None or backing_converted is None):
                raise ValueError("当前作品没有可用于转换的和声 stem，请保留原作品重新生成。")
            report = mix_cover(
                instrumental=instrumental, converted=converted, reference_dry=reference_dry,
                output_mix=output / "cover.wav", output_vocal=output / "my_vocals.wav",
                reference_wet=_safe_file(parent, "lead_reverb.wav"),
                backing=backing if mode != "remove" else None,
                backing_converted=backing_converted if mode == "convert" else None,
                backing_solo=backing_converted if mode == "original" else None, **params)
        new_job = dict(job)
        new_job.update({"run_dir": str(output), "parent_run": parent.name,
                        "vocal_gain": params["vocal_gain_db"], "instrumental_gain": params["music_gain_db"],
                        "backing": mode, "clarity": params["clarity"], "dynamics": params["dynamics"],
                        "reverb": params["reverb"]})
        _write_json(output / "job.json", new_job)
        _write_json(output / "mix_report.json", report if isinstance(report, dict) else {})
        result = {"status": "重混音完成", "run": str(output), "mix": str(output / "cover.wav"),
                  "converted": str(output / "my_vocals.wav"), "parent_run": parent.name}
        if duet_mode != "off":
            result.update(files, duet_mode=duet_mode)
            if separator is not None:
                result["duet_separator"] = separator
        _write_json(output / "result.json", result)
        custom = dict(parent_meta.get("metadata") or {})
        custom.update({"title": parent_meta["title"], "parent_run": parent.name, "created_at": _now(),
                       "updated_at": _now(), "remix": params})
        _write_json(output / "work.json", custom)
        return _metadata(output)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
