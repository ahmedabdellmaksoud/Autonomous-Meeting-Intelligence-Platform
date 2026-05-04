"""
CorpBrain — Cognitive Service Utils
=====================================
Saves and loads cognitive result JSON from disk.
Filename convention: {meeting_id}_cognitive.json
"""

import json
from pathlib import Path

from config import RESULTS_DIR


def save_result(meeting_id: str, data: dict) -> Path:
    dest = RESULTS_DIR / f"{meeting_id}_cognitive.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[utils] Saved → {dest}")
    return dest


def load_result(meeting_id: str) -> dict | None:
    dest = RESULTS_DIR / f"{meeting_id}_cognitive.json"
    if not dest.exists():
        return None
    with open(dest, "r", encoding="utf-8") as f:
        return json.load(f)