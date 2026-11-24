"""Audio inputs shared by song separation and uploaded duet stems."""
from pathlib import Path
import os
import subprocess


AUDIO_EXTENSIONS = (
    ".wav", ".wave", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".aifc", ".ape", ".wv", ".tta", ".tak", ".caf",
    ".alac", ".m4b", ".mp4", ".mka", ".webm", ".weba", ".ac3", ".eac3",
    ".dts", ".amr", ".au", ".mp2", ".w64",
)
FFMPEG = Path(__file__).resolve().parents[1] / "Applio" / "ffmpeg.exe"


def decode_audio(source: Path, target: Path, *, separation_input: bool = False) -> None:
    """Decode the first audio stream, keeping native channel count/rate for uploaded stems."""
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"缺少 FFmpeg：{FFMPEG}")
    pcm = ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le"] if separation_input else ["-c:a", "pcm_f32le"]
    completed = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-protocol_whitelist", "file", "-format_whitelist",
         "wav,w64,mp3,flac,mov,aac,ogg,asf,aiff,ape,wv,tta,tak,caf,matroska,ac3,eac3,dts,amr,au",
         "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn", *pcm, str(target)],
        capture_output=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode:
        detail = completed.stderr.strip()[-1600:]
        raise ValueError(f"无法解码 {source.name}。请确认文件包含可读取的音轨并重新上传。\n{detail}")
