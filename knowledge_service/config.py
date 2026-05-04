"""
CorpBrain — Knowledge Service Configuration
============================================
No external AI API required.
Embeddings: sentence-transformers all-MiniLM-L6-v2 (local)
Vector store: ChromaDB (local persistent)
Answers: extractive (sentence overlap) — or plug in your T5 model
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

BASE_DIR   = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_NAME = "knowledge"
SERVICE_HOST = os.getenv("KNOWLEDGE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("KNOWLEDGE_PORT", "8003"))

EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "meeting_transcripts")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "50"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "10"))
