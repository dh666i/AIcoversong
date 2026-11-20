"""Safe local works-library helpers for AICover.

The library is deliberately independent from Gradio.  It treats a finished
``output/run_*`` directory as an immutable work, stores user-facing metadata
in ``work.json``, and puts deleted works in ``output/.trash`` so they can be
restored.  Re-mixes are new run directories; an earlier version is never
overwritten.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "output"
_RUN_NAME = re.compile(r"^run_[A-Za-z0-9][A-Za-z0-9_-]*$")
_TITLE_LIMIT = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _root(output_root: str | Path | None) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    root = root.expanduser().resolve(strict=False)
    if root.name == ".trash":
        raise ValueError("作品库根目录不能是回收站目录。")
    return root


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _assert_root(root: Path) -> None:
    if root.exists() and root.is_symlink():
        raise ValueError("作品库根目录不能是符号链接。")
    if root.exists() and not root.is_dir():
        raise ValueError(f"作品库根目录不是文件夹：{root}")


def _run_from(value: Any, root: Path, *, allow_trash: bool = False) -> Path:
    """Resolve a work argument and reject path traversal or symlink escape."""
    _assert_root(root)
    if isinstance(value, dict):
        value = value.get("path") or value.get("run") or value.get("run_id")
    if value is None:
        raise ValueError("没有指定作品。")
    raw = Path(str(value))
    if not raw.is_absolute():
        raw = root / raw
    candidate = raw.expanduser().resolve(strict=False)
    trash = (root / ".trash").resolve(strict=False)
    parent = trash if allow_trash and _inside(candidate, trash) else root
    if not _inside(candidate, parent) or candidate.parent != parent:
        raise ValueError("只能操作作品库中的 run_* 作品目录。")
    if not _RUN_NAME.fullmatch(candidate.name):
        raise ValueError("作品目录名称无效。")
    if candidate.is_symlink() or not candidate.is_dir():
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
        if not run.is_dir() or run.is_symlink() or not _RUN_NAME.fullmatch(run.name):
            continue
        try:
            _safe_file(run, "cover.wav", required=True)
            works.append(_metadata(run))
        except (OSError, ValueError):
            continue
    return works


def list_trashed(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return recoverable works in the private trash folder, newest first."""
    root = _root(output_root)
    trash = root / ".trash"
    if not trash.exists():
        return []
    if trash.is_symlink() or not trash.is_dir():
        raise ValueError("作品回收站不是安全的普通文件夹。")
    works: list[dict[str, Any]] = []
    for run in sorted(trash.iterdir(), key=lambda path: path.name, reverse=True):
        if not run.is_dir() or run.is_symlink() or not _RUN_NAME.fullmatch(run.name):
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


def trash_work(work: str | Path | dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(output_root)
    run = _run_from(work, root)
    trash = root / ".trash"
    if trash.exists() and trash.is_symlink():
        raise ValueError("作品回收站不能是符号链接。")
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / run.name
    if destination.exists() or destination.is_symlink():
        raise ValueError("回收站中已经有同名作品。")
    shutil.move(str(run), str(destination))
    return {"run_id": run.name, "path": str(destination), "trashed_at": _now()}


def restore_work(work: str | Path | dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(output_root)
    raw = work.get("path") if isinstance(work, dict) else work
    if raw is None:
        raise ValueError("没有指定要恢复的作品。")
    trash = root / ".trash"
    raw_path = Path(str(raw))
    if not raw_path.is_absolute() and raw_path.parent == Path("."):
        # A bare run id addresses the recoverable copy, never the live work.
        candidate = _run_from(trash / raw_path.name, root, allow_trash=True)
    else:
        candidate = _run_from(raw, root, allow_trash=True)
        if candidate.parent != trash.resolve(strict=False):
            raise ValueError("只能从作品回收站恢复作品。")
    destination = root / candidate.name
    if destination.exists() or destination.is_symlink():
        raise ValueError("作品库中已经有同名作品，无法恢复。")
    shutil.move(str(candidate), str(destination))
    return _metadata(destination)


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
    instrumental = _safe_file(parent, "instrumental.wav", required=True)
    converted = _safe_file(parent, "converted_vocals.wav") or _safe_file(parent, "my_vocals.wav", required=True)
    reference_dry = _safe_file(parent, "lead_dry.wav") or _safe_file(parent, "vocals.wav", required=True)
    reference_wet = _safe_file(parent, "lead_reverb.wav")
    backing = _safe_file(parent, "backing_vocals.wav")
    backing_converted = _safe_file(parent, "converted_backing.wav")
    mode = backing_mode or job.get("backing", "original")
    if mode not in {"original", "convert", "remove"}:
        raise ValueError("和声处理方式无效。")
    if mode == "convert" and (backing is None or backing_converted is None):
        raise ValueError("当前作品没有可用于转换的和声 stem，请保留原作品重新生成。")
    if mode == "original" and backing is None:
        backing = None
    try:
        from vocal_mix import mix_cover
    except ImportError:  # package import when called outside the UI process
        from .vocal_mix import mix_cover

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
        mix_args = {
            "instrumental": instrumental, "converted": converted, "reference_dry": reference_dry,
            "output_mix": output / "cover.wav", "output_vocal": output / "my_vocals.wav",
            "reference_wet": reference_wet,
            "backing": backing if mode != "remove" else None,
            "backing_converted": backing_converted if mode == "convert" else None,
            "backing_solo": backing_converted if mode == "original" else None,
            **params,
        }
        report = mix_cover(**mix_args)
        new_job = dict(job)
        new_job.update({"run_dir": str(output), "parent_run": parent.name,
                        "vocal_gain": params["vocal_gain_db"], "instrumental_gain": params["music_gain_db"],
                        "backing": mode, "clarity": params["clarity"], "dynamics": params["dynamics"],
                        "reverb": params["reverb"]})
        _write_json(output / "job.json", new_job)
        _write_json(output / "mix_report.json", report if isinstance(report, dict) else {})
        result = {"status": "重混音完成", "run": str(output), "mix": str(output / "cover.wav"),
                  "converted": str(output / "my_vocals.wav"), "parent_run": parent.name}
        _write_json(output / "result.json", result)
        custom = dict(parent_meta.get("metadata") or {})
        custom.update({"title": parent_meta["title"], "parent_run": parent.name, "created_at": _now(),
                       "updated_at": _now(), "remix": params})
        _write_json(output / "work.json", custom)
        return _metadata(output)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
