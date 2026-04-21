"""
CorpBrain — Ingestion Service Configuration
=============================================
Central settings for the file-upload microservice.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The upload directory lives *inside* the ingestion_service folder.
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"

# Ensure the folder exists on import so we never hit a missing-dir error.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File constraints
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".mp4"}
MAX_FILE_SIZE_MB: int = 500  # Maximum upload size in megabytes
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVICE_NAME: str = "ingestion"
SERVICE_HOST: str = os.getenv("INGESTION_HOST", "0.0.0.0")
SERVICE_PORT: int = int(os.getenv("INGESTION_PORT", "8000"))
