"""
Subtitle generation service: FFmpeg → Whisper → SRT/VTT
"""
import os
import subprocess
from pathlib import Path
import whisper
from whisper.utils import get_writer


def extract_audio(video_path: str, output_path: str, log_callback=None) -> str:
    """Extract audio from video using FFmpeg."""
    def _log(msg: str):
        if log_callback:
            log_callback(f"[FFmpeg] {msg}")

    _log("Memulai ekstraksi audio...")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            output_path,
            "-loglevel", "warning"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            err = result.stderr or result.stdout or "Unknown error"
            _log(f"Error: {err}")
            raise RuntimeError(f"FFmpeg gagal: {err}")
        _log("Ekstraksi audio selesai.")
        return output_path
    except subprocess.TimeoutExpired:
        _log("Error: Timeout (max 5 menit)")
        raise
    except FileNotFoundError:
        _log("Error: FFmpeg tidak ditemukan. Install: apt install ffmpeg")
        raise


def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "base",
    log_callback=None
) -> dict:
    """Transcribe audio using Whisper ASR."""
    def _log(msg: str):
        if log_callback:
            log_callback(f"[Whisper] {msg}")

    _log(f"Memuat model Whisper ({model_size})...")
    try:
        model = whisper.load_model(model_size)
        _log("Model dimuat. Memulai transkripsi...")
        result = model.transcribe(audio_path, language=None, fp16=False)
        _log("Transkripsi selesai.")
        return result
    except Exception as e:
        _log(f"Error: {str(e)}")
        raise


def format_subtitle(
    result: dict,
    output_dir: str,
    base_name: str,
    formats: list[str] = ["srt", "vtt"],
    log_callback=None
) -> dict[str, str]:
    """Format Whisper result to SRT and/or VTT."""
    def _log(msg: str):
        if log_callback:
            log_callback(f"[Formatter] {msg}")

    _log("Memformat subtitle...")
    paths = {}
    for fmt in formats:
        writer = get_writer(fmt, output_dir)
        writer(result, base_name, {})
        path = os.path.join(output_dir, f"{base_name}.{fmt}")
        if os.path.exists(path):
            paths[fmt] = path
            _log(f"  → {fmt.upper()} tersedia")
    _log("Formatting selesai.")
    return paths


AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def process_media_to_subtitle(
    input_path: str,
    output_dir: str,
    model_size: str = "base",
    subtitle_formats: list[str] = None,
    log_callback=None
) -> dict[str, str]:
    """
    Full pipeline: extract audio (jika video) → transcribe → format subtitle.
    Returns dict of format -> file path.
    """
    if subtitle_formats is None:
        subtitle_formats = ["srt", "vtt"]

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    base_name = Path(input_path).stem
    ext = Path(input_path).suffix.lower()
    temp_audio = None
    audio_path = input_path

    try:
        if ext in AUDIO_EXT:
            _log("=== Step 1: File audio, skip ekstraksi ===")
        else:
            _log("=== Step 1: Ekstraksi Audio ===")
            temp_audio = os.path.join(output_dir, f"{base_name}_temp.mp3")
            extract_audio(input_path, temp_audio, log_callback)
            audio_path = temp_audio

        _log("=== Step 2: Transkripsi Whisper ===")
        result = transcribe_with_whisper(audio_path, model_size, log_callback)

        _log("=== Step 3: Format Subtitle ===")
        paths = format_subtitle(
            result, output_dir, base_name, subtitle_formats, log_callback
        )
        return paths
    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass
