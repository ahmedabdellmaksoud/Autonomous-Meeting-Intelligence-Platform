"""
CorpBrain — Transcriber
========================
Uses faster-whisper for local, GPU-accelerated transcription.
No API key needed. Supports Arabic and English natively.
Runs on GPU (float16) if available, CPU (int8) otherwise.
"""

from faster_whisper import WhisperModel
from pathlib import Path

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE

print(f"[transcriber] Loading faster-whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print("[transcriber] Ready.")


def transcribe(audio_path: str | Path) -> dict:
    """
    Transcribe audio file using faster-whisper.

    Returns dict with:
      - text: full concatenated transcript
      - language: detected language code (e.g. 'ar', 'en')
      - language_probability: confidence score 0.0–1.0
      - segments: list of timed segments [{start, end, text}]
    """
    segments_gen, info = _model.transcribe(str(audio_path), beam_size=5)

    segment_list = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_gen
    ]
    full_text = " ".join(s["text"] for s in segment_list).strip()

    print(f"[transcriber] Done — {len(full_text)} chars, language={info.language} ({info.language_probability:.2f})")
    return {
        "text":                 full_text,
        "language":             info.language,
        "language_probability": round(info.language_probability, 3),
        "segments":             segment_list,
    }