"""
Autocaption Web - FastAPI backend
Video/Audio → FFmpeg → Whisper → SRT/VTT
"""
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.services import process_media_to_subtitle

app = FastAPI(title="Autocaption Web", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "autocaption_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed extensions: video + audio
ALLOWED_EXT = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma",
}


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    format: str = Form("srt"),
):
    """
    Upload video/audio, process, return subtitle download info.
    format: srt | vtt | both
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400,
            detail=f"Format tidak didukung. Gunakan: {', '.join(sorted(ALLOWED_EXT))}"
        )

    logs: list[str] = []

    def log_cb(msg: str):
        logs.append(msg)
        print(msg)

    try:
        # Save upload
        safe_name = "".join(c for c in (file.filename or "upload") if c.isalnum() or c in "._- ")
        if not safe_name:
            safe_name = "upload"
        input_path = UPLOAD_DIR / safe_name
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        log_cb(f"[Upload] File disimpan: {file.filename} ({len(content)} bytes)")

        # Determine subtitle formats
        formats = ["srt", "vtt"] if format == "both" else [format]
        if format not in ("srt", "vtt", "both"):
            formats = ["srt"]

        # Process
        output_dir = str(UPLOAD_DIR)
        paths = process_media_to_subtitle(
            str(input_path),
            output_dir,
            model_size=model_size,
            subtitle_formats=formats,
            log_callback=log_cb,
        )

        # Cleanup input
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass

        # Return download URLs (only subtitle files)
        downloads = []
        for fmt, p in paths.items():
            name = Path(p).name
            downloads.append({"format": fmt, "filename": name, "url": f"/api/download/{name}"})

        return JSONResponse({
            "success": True,
            "logs": logs,
            "downloads": downloads,
        })
    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()
        log_cb(f"[ERROR] {err_msg}")
        log_cb(f"[Traceback]\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "logs": logs,
                "error": err_msg,
                "traceback": tb,
            },
        )


@app.get("/api/download/{filename}")
async def download_subtitle(filename: str):
    """Download subtitle file only (SRT/VTT)."""
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File tidak ditemukan")
    ext = path.suffix.lower()
    if ext not in (".srt", ".vtt"):
        raise HTTPException(400, "Hanya file subtitle (SRT/VTT) yang dapat diunduh")
    return FileResponse(
        path,
        filename=filename,
        media_type="text/plain" if ext == ".srt" else "text/vtt",
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
