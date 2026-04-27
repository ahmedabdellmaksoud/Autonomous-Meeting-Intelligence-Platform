"""
CorpBrain — Cognitive Service Utils
=====================================
Saves and loads task JSON results from disk.
Every other service finds the result using the meeting_id.
"""

import json
from pathlib import Path

from config import RESULTS_DIR


def save_result(meeting_id: str, data: dict) -> Path:
    """
    Save *data* as results/{meeting_id}_tasks.json.
    Returns the full path of the saved file.
    """
    dest = RESULTS_DIR / f"{meeting_id}_tasks.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[utils] Saved → {dest}")
    return dest


def load_result(meeting_id: str) -> dict | None:
    """
    Load and return the task JSON for *meeting_id*.
    Returns None if the file does not exist yet.
    """
    dest = RESULTS_DIR / f"{meeting_id}_tasks.json"
    if not dest.exists():
        return None
    with open(dest, "r", encoding="utf-8") as f:
        return json.load(f)