"""Encode the studio's WAV masters for download without modifying the source."""
from pathlib import Path
import os
import shutil
import subprocess


FFMPEG = Path(__file__).resolve().parents[1] / "Applio" / "ffmpeg.exe"
_CODECS = {
    "wav": (),
    "flac": ("-c:a", "flac"),
    "mp3": ("-c:a", "libmp3lame", "-q:a", "2"),
    "m4a": ("-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"),
    "ogg": ("-c:a", "libvorbis", "-q:a", "6"),
    "opus": ("-c:a", "libopus", "-b:a", "160k"),
    "aiff": ("-c:a", "pcm_s24be"),
}
EXPORT_FORMATS = tuple(_CODECS)


def check_format(audio_format: str) -> None:
    if audio_format not in EXPORT_FORMATS:
        raise ValueError("下载格式无效，请重新选择。")


def write_audio(source: Path, destination: Path) -> None:
    """Write into the exporter's private temporary directory."""
    audio_format = destination.suffix[1:]
    check_format(audio_format)
    if audio_format == "wav":
        shutil.copyfile(source, destination)
        return
    if not FFMPEG.is_file():
        raise ValueError("缺少 FFmpeg，请运行基础模型和工具安装，或选择 WAV 下载。")

    process = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-f", "wav", "-i", str(source), "-map", "0:a:0", "-map_metadata", "-1",
         *_CODECS[audio_format], str(destination)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if process.returncode or not destination.is_file() or not destination.stat().st_size:
        raise ValueError(f"未能生成 {audio_format.upper()} 文件，请检查源音频和可用磁盘空间。")
