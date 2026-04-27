"""
CorpBrain — Cognitive Service Configuration
============================================
Central settings for the transcription + extraction microservice.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly by path so it works from any working directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Service identity ──────────────────────────────────────
# Each service owns its own port:
#   ingestion  → 8000
#   cognitive  → 8001
#   agentic    → 8002
#   knowledge  → 8003
SERVICE_NAME = "cognitive"
SERVICE_HOST = os.getenv("COGNITIVE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("COGNITIVE_PORT", "8001"))

# ── Whisper ───────────────────────────────────────────────
# "base" is fast and good enough for MVP.
# Upgrade to "small" or "medium" later for better Arabic accuracy.
WHISPER_MODEL = "base"

# ── Gemini ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-1.5-flash"