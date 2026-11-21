"""Download the public asset inventory; no model loading and no private proxy.

Use --check for read-only verification or --list to see the transfer size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "model-manifest.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def destination(root: Path, item: dict) -> Path:
    relative = Path(item["dest"])
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError("Asset destination must stay within the installation folder")
    if not isinstance(item.get("size"), int) or item["size"] <= 0:
        raise ValueError("Asset size is missing or invalid")
    value = item.get("sha256", "")
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("Every asset requires a SHA-256 digest")
    if urlparse(item["url"]).scheme != "https":
        raise ValueError("Only HTTPS download sources are accepted")
    return target


def valid(path: Path, item: dict) -> bool:
    return (path.is_file() and path.stat().st_size == item["size"]
            and digest(path) == item["sha256"])


def download(item: dict, root: Path, opener, hf_endpoint: str | None) -> None:
    target = destination(root, item)
    if valid(target, item):
        print("READY", item["dest"], flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = item["url"]
    if hf_endpoint and url.startswith("https://huggingface.co/"):
        url = hf_endpoint.rstrip("/") + url[len("https://huggingface.co"):]
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= item["size"]:
                if valid(partial, item):
                    partial.replace(target)
                    print("VERIFIED", item["dest"], flush=True)
                    return
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "AIcoversong-asset-setup/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"DOWNLOAD {item['dest']} ({item['size'] / 1e6:.1f} MB), attempt {attempt}", flush=True)
            with opener.open(Request(url, headers=headers), timeout=45) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent byte range")
                with partial.open("ab" if resume else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if not valid(partial, item):
                raise ValueError("Downloaded size or SHA-256 did not match the manifest")
            partial.replace(target)
            print("VERIFIED", item["dest"], flush=True)
            return
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(f"Could not download {item['dest']}: {exc}") from exc
            print(f"Retrying {item['dest']}: {exc}", flush=True)
            time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Installation folder (also useful with --check)")
    parser.add_argument("--check", action="store_true", help="Read-only file and hash verification")
    parser.add_argument("--list", action="store_true", help="Print inventory without downloading")
    parser.add_argument("--duet", action="store_true", help="Select only the optional male/female duet separator assets")
    parser.add_argument("--direct", action="store_true", help="Ignore system/environment proxies for this download")
    parser.add_argument("--hf-endpoint", help="Explicit HTTPS mirror for Hugging Face downloads")
    args = parser.parse_args()
    if args.hf_endpoint:
        parsed = urlparse(args.hf_endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            parser.error("--hf-endpoint must be an HTTPS origin without credentials, query or path")
    inventory = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = inventory["duet_files" if args.duet else "files"]
    for item in items:
        destination(args.root, item)
    print(f"{len(items)} assets, {sum(item['size'] for item in items) / 1e9:.2f} GB. Terms: docs/MODELS.md")
    if args.list:
        for item in items:
            print(f"{item['size'] / 1e6:8.2f} MB  {item['dest']}")
        return 0
    if args.check:
        missing = []
        for item in items:
            ok = valid(destination(args.root, item), item)
            print("OK" if ok else "MISSING/INVALID", item["dest"])
            if not ok:
                missing.append(item["dest"])
        return 1 if missing else 0
    opener = build_opener(ProxyHandler({})) if args.direct else build_opener()
    for item in items:
        download(item, args.root, opener, args.hf_endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
