"""
CorpBrain — Cognitive Service Configuration
============================================
No external AI API required.
Transcription: faster-whisper (local)
Classification: BERT checkpoint (local, from Kaggle training)
Extraction: rule-based + T5 model checkpoint (local, from Kaggle training)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROOT_DIR    = BASE_DIR.parent
CHECKPOINTS = ROOT_DIR / "models" / "checkpoints"

# ── Service identity ───────────────────────────────────────────────────────
SERVICE_NAME = "cognitive"
SERVICE_HOST = os.getenv("COGNITIVE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("COGNITIVE_PORT", "8001"))

# ── Whisper (faster-whisper) ───────────────────────────────────────────────
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE  = "cuda" if __import__("torch").cuda.is_available() else "cpu"
WHISPER_COMPUTE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

# ── BERT Classifier checkpoint (trained on Kaggle, downloaded here) ────────
BERT_CHECKPOINT     = CHECKPOINTS / "meeting_classifier"
CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))

# ── T5 Extractor checkpoint (trained on Kaggle, downloaded here) ──────────
EXTRACTOR_CHECKPOINT = CHECKPOINTS / "extractor"