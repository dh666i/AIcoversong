"""Fail on private/generated files or likely secrets in the source publication.

This is a focused guard, not a guarantee that arbitrary content is safe to share.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {"recordings", "songs", "output", "logs", "cache", "downloads", "runtime", "env", "separator-env", ".setup-env", ".venv", ".claude", ".codex"}
FORBIDDEN_EXTENSIONS = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".index", ".bin", ".exe", ".dll", ".whl", ".mp3", ".flac", ".m4a", ".ogg", ".wav", ".npy", ".npz", ".part", ".pid", ".pem", ".key", ".p12", ".pfx"}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".ps1", ".bat", ".js", ".css", ".html", ".toml", ".cfg", ".ini"}
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+(?:Administrator|ADMINI~|[^\\/\s]+\.DESKTOP)"),
    re.compile(r"(?i)[A-Z]:[\\/]+SSHyuankong[\\/]+codex[\\/]+xxx"),
)


def inspect(root: Path, paths: list[Path], fixtures: dict) -> list[str]:
    errors = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if path.is_symlink():
            errors.append(f"Symlink requires explicit review: {relative}")
            continue
        if not path.is_file():
            continue
        if relative in fixtures:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != fixtures[relative]:
                errors.append(f"Upstream fixture differs from pinned commit: {relative}")
            continue
        private = (relative.split('/')[0].lower() in PRIVATE_ROOTS
                   or lower.startswith('applio/logs/') and lower != 'applio/logs/reference/create_reference.py'
                   or lower == 'applio/assets/config.json'
                   or '/__pycache__/' in lower
                   or path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example')
        if private:
            errors.append(f"Private or generated path: {relative}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"Runtime/model/audio artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Unexpected large source artifact: {relative}")
        if path.suffix.lower() in TEXT_EXTENSIONS:
            value = path.read_text(encoding='utf-8-sig', errors='replace')
            if any(pattern.search(value) for pattern in SECRET_PATTERNS):
                errors.append(f"Possible credential or personal machine path: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked', action='store_true', help='Inspect the Git index (recommended before push)')
    args = parser.parse_args()
    fixtures = json.loads((ROOT/'assets/upstream-fixtures.json').read_text(encoding='utf-8'))
    if args.tracked:
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True)
        paths = [ROOT / name for name in result.stdout.decode('utf-8').split('\0') if name]
    else:
        paths = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.relative_to(ROOT).parts]
    errors = inspect(ROOT, paths, fixtures)
    for error in errors:
        print('FAIL', error)
    if errors:
        return 1
    print(f'Publication check passed: {len(paths)} files. No inference or downloads were performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
