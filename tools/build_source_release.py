"""Create a source-only ZIP from committed files, after publication checks.

Run from the repository after committing the intended release. User datasets,
downloaded weights and Python environments are never recursively packaged.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run([sys.executable, str(ROOT/'tools/verify_package.py'), '--tracked'], cwd=ROOT, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT).strip():
        raise SystemExit('Commit intended tracked changes before creating a release archive.')
    output = ROOT/'dist'/'AIcoversong-source.zip'
    output.parent.mkdir(exist_ok=True)
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=AIcoversong/', '-o', str(output), 'HEAD'], cwd=ROOT, check=True)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
