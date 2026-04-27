"""
CorpBrain — Transcriber
========================
Loads Whisper once and transcribes audio files to plain text.
Runs fully locally on your GPU (RTX 4070) — no API call, no cost.
"""

import whisper
from pathlib import Path

from config import WHISPER_MODEL

# Load the model once when this module is first imported.
# Takes ~5 seconds on first run, then stays in memory.
print(f"[transcriber] Loading Whisper '{WHISPER_MODEL}' model ...")
_model = whisper.load_model(WHISPER_MODEL)
print(f"[transcriber] Whisper ready.")


def transcribe(audio_path: str | Path) -> str:
    """
    Run Whisper on the audio file at *audio_path*.

    Returns the full transcript as a plain string.
    Whisper auto-detects the language — works for Arabic and English.
    fp16=False forces CPU-safe float32 (GPU will still be used automatically).
    """
    result = _model.transcribe(str(audio_path), fp16=False)
    transcript = result["text"].strip()
    print(f"[transcriber] Done — {len(transcript)} characters.")
    return transcript