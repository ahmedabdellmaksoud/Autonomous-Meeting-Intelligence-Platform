"""
CorpBrain — Extractor
======================
Extracts structured context from meeting transcripts using local models only.

Priority:
  1. T5 checkpoint (if models/checkpoints/extractor/ exists — train on Kaggle)
  2. Rule-based regex extraction (always available, no model needed)
"""

import json
import re
from pathlib import Path

from config import EXTRACTOR_CHECKPOINT

_FENCE_RE = re.compile(r"^```[a-z]*\n?|```$", re.MULTILINE)

# ── Action/blocker/decision patterns ─────────────────────────────────────
_ACTION_PATTERNS = [
    re.compile(r'\b([A-Z][a-z]+)\b\s+will\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'\b([A-Z][a-z]+)\b\s+(?:needs?|has)\s+to\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'\b([A-Z][a-z]+)\b\s+(?:should|must)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'(?:assign|assigned)\s+(?:to\s+)?([A-Z][a-z]+)\b[:\s]+(.+?)(?:[.!?]|$)', re.IGNORECASE),
]
_BLOCKER_PATTERNS = [
    re.compile(r'\b(?:blocked?\s+by|blocker[:\s]+|impediment[:\s]+|stuck\s+(?:on|at)[:\s]+)\s*(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'\b(?:issue|problem)\s+(?:with|in|on)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'\bmain\s+blocker\s*[:\-]\s*(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'\bwaiting\s+(?:for|on)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
]
_DECISION_PATTERNS = [
    re.compile(r'(?:we\s+)?decided?\s+(?:to\s+)?(.+?)(?:[.!?]|$)', re.IGNORECASE),
    re.compile(r'(?:we\s+)?agreed?\s+(?:to\s+)?(.+?)(?:[.!?]|$)', re.IGNORECASE),
]
_SPRINT_RE = re.compile(r'sprint\s*(\d+|[a-z]+)', re.IGNORECASE)
_COMMON_WORDS = {
    "The","This","That","We","Our","For","But","And","Or","So","If","In","On",
    "At","To","By","Ok","Yes","No","All","Any","Can","Let","Good","New","Jira",
    "Sprint","Team","API","Dev","QA","UI","PR","Git","Tech","Code","Task",
}
_MEETING_TYPES = {
    "standup":      ["standup","stand-up","daily","yesterday","today","blocker"],
    "planning":     ["planning","backlog","sprint plan","story point","estimate"],
    "review":       ["review","demo","showcase","completed","delivered"],
    "retrospective":["retrospective","retro","what went well","improve"],
}


def _rule_based_extract(transcript: str) -> dict:
    """Rule-based extraction using regex. No model or API needed."""
    words_lower = transcript.lower()
    meeting_type = "other"
    for mtype, keywords in _MEETING_TYPES.items():
        if any(kw in words_lower for kw in keywords):
            meeting_type = mtype
            break

    sprint_match = _SPRINT_RE.search(transcript)
    sprint = f"Sprint {sprint_match.group(1)}" if sprint_match else "unknown"

    action_items, seen = [], set()
    name_candidates = set()
    for pattern in _ACTION_PATTERNS:
        for m in pattern.finditer(transcript):
            assignee = m.group(1).strip()
            task     = m.group(2).strip()
            if assignee in _COMMON_WORDS or len(task) < 5 or task in seen:
                continue
            seen.add(task)
            name_candidates.add(assignee)
            action_items.append({
                "task":     task[:150],
                "assignee": assignee,
                "deadline": "not specified",
                "priority": "medium",
                "type":     "feature",
                "context":  "",
            })

    blockers, decisions = [], []
    for p in _BLOCKER_PATTERNS:
        for m in p.finditer(transcript):
            b = m.group(1).strip()
            if len(b) > 5 and b not in blockers:
                blockers.append(b[:100])
    for p in _DECISION_PATTERNS:
        for m in p.finditer(transcript):
            d = m.group(1).strip()
            if len(d) > 5 and d not in decisions:
                decisions.append(d[:100])

    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    summary   = " ".join(sentences[:3]) if sentences else transcript[:300]

    print(f"[extractor] Rule-based: {len(action_items)} action items.")
    return {
        "summary":      summary[:400],
        "meeting_type": meeting_type,
        "project":      "unknown",
        "sprint":       sprint,
        "participants": list(name_candidates)[:10],
        "decisions":    decisions[:5],
        "blockers":     blockers[:5],
        "action_items": action_items[:8],
    }


def _t5_extract(transcript: str) -> dict:
    """
    Use the fine-tuned T5/Flan-T5 checkpoint from Kaggle.
    Returns the same dict format as _rule_based_extract.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(EXTRACTOR_CHECKPOINT))
    model     = AutoModelForSeq2SeqLM.from_pretrained(str(EXTRACTOR_CHECKPOINT))
    model.eval()

    input_text = "extract meeting context: " + transcript
    inputs     = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)

    raw    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = json.loads(raw)
    print(f"[extractor] T5 model: {len(result.get('action_items', []))} items.")
    return result


def extract(transcript: str) -> dict:
    """
    Extract structured context from transcript.
    Priority: T5 Kaggle checkpoint → rule-based regex.
    """
    if EXTRACTOR_CHECKPOINT.exists():
        try:
            return _t5_extract(transcript)
        except Exception as e:
            print(f"[extractor] T5 failed ({e}), falling back to rule-based.")
    return _rule_based_extract(transcript)