# Autocaption Web

Generate subtitle (SRT/VTT) dari video atau audio menggunakan FFmpeg + Whisper ASR.

## Alur

```
Video/Audio (mp4, mp3, dll)
    ↓
FFmpeg (extract audio)
    ↓
Whisper ASR (transkripsi)
    ↓
Subtitle (SRT / VTT)
```

## Fitur

- **Upload**: Video (MP4, MKV, AVI, MOV, WebM, dll) atau Audio (MP3, WAV, M4A, FLAC, dll)
- **Download**: Hanya file subtitle (SRT/VTT), bukan video
- **Log**: Tampilan log proses dan error teknis
- **Model**: Tiny, Base, Small, Medium, Large

## Setup Lokal

```bash
# Install FFmpeg (Linux)
sudo apt install ffmpeg

# Install Python deps
pip install -r requirements.txt

# Jalankan
python run.py
# atau: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Buka http://localhost:8000

## Colab / Kaggle

Lihat `notebooks/colab_autocaption.ipynb` untuk menjalankan di Google Colab atau Kaggle dengan ngrok.
