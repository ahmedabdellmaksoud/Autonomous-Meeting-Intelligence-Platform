"""
CorpBrain — Knowledge Service Utilities
"""
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    """Read JSON file, return None if not found."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
