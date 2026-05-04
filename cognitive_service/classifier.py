"""
CorpBrain — Meeting Classifier
================================
Priority:
  1. BERT checkpoint (models/checkpoints/meeting_classifier/ — train on Kaggle)
  2. Keyword-based fallback (always available, no model needed)
"""

import re
from pathlib import Path

from config import BERT_CHECKPOINT, CLASSIFIER_THRESHOLD

_MEETING_KEYWORDS = [
    "standup", "stand-up", "sprint", "backlog", "scrum", "retrospective", "retro",
    "planning", "action item", "blocker", "blocked", "assigned", "assignee", "deadline",
    "team", "review", "demo", "discussion", "agenda", "meeting", "participants", "decision",
    "milestone", "ticket", "jira", "pull request", "pr", "deployment", "deploy", "release",
    "estimate", "story point", "velocity", "capacity", "stakeholder", "will fix", "will work",
    "will deploy", "will complete", "will finish", "will add", "will implement", "will write",
    "finish", "completed", "yesterday", "today", "tomorrow", "by friday", "by monday",
    "staging", "production", "bug fix", "integrate", "integration", "feature", "epic",
]
_NOT_MEETING_KEYWORDS = [
    "warranty", "insurance", "automated message", "press 1", "press 2",
    "weather forecast", "cooking show", "documentary", "podcast", "lecture",
    "flight", "fasten seatbelt", "your account balance", "recipe", "novel",
    "news report", "breaking news", "advertisement", "commercial",
]


def _keyword_classify(transcript: str) -> dict:
    """Keyword-based classifier. Always available, zero dependencies."""
    text_lower = transcript.lower()
    m_score  = sum(1 for kw in _MEETING_KEYWORDS if kw in text_lower)
    nm_score = sum(1 for kw in _NOT_MEETING_KEYWORDS if kw in text_lower)

    if nm_score >= 2 and m_score <= 1:
        label, confidence = "not_meeting", min(0.5 + nm_score * 0.08, 0.92)
    elif m_score >= 3:
        label, confidence = "meeting", min(0.5 + m_score * 0.07, 0.92)
    else:
        label      = "meeting" if m_score > nm_score else "not_meeting"
        confidence = 0.60

    print(f"[classifier] Keyword: {label} (meeting={m_score}, not_meeting={nm_score})")
    return {
        "label":        label,
        "confidence":   round(confidence, 3),
        "reason":       f"keyword-based (meeting={m_score}, not_meeting={nm_score})",
        "meeting_type": "unknown",
    }


def _bert_classify(transcript: str) -> dict:
    """Use the BERT checkpoint trained on Kaggle."""
    from transformers import pipeline
    clf    = pipeline("text-classification", model=str(BERT_CHECKPOINT))
    result = clf(transcript[:512])[0]
    label  = result["label"].lower()
    score  = round(result["score"], 3)
    print(f"[classifier] BERT: {label} ({score:.1%})")
    return {
        "label":        label,
        "confidence":   score,
        "reason":       "BERT fine-tuned classifier",
        "meeting_type": "unknown",
    }


def classify(transcript: str) -> dict:
    """
    Classify transcript. Priority: BERT checkpoint → keyword fallback.
    Never raises — always returns a valid dict.
    """
    if BERT_CHECKPOINT.exists():
        try:
            return _bert_classify(transcript)
        except Exception as e:
            print(f"[classifier] BERT failed ({e}), using keywords.")
    return _keyword_classify(transcript)
