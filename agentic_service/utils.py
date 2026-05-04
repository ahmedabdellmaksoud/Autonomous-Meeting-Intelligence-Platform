"""
CorpBrain — Agentic Service Utilities
"""

import json
from pathlib import Path

from config import COGNITIVE_RESULTS_DIR, RESULTS_DIR


def load_cognitive_result(meeting_id: str) -> dict | None:
    """Load the cognitive service output for a given meeting."""
    path = COGNITIVE_RESULTS_DIR / f"{meeting_id}_cognitive.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_agentic_result(meeting_id: str, data: dict) -> Path:
    """Persist the final agentic pipeline output."""
    path = RESULTS_DIR / f"{meeting_id}_agentic.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[utils] Saved → {path}")
    return path
