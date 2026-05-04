"""
CorpBrain — Agentic Service Configuration
==========================================
No external AI API required for core pipeline.
Jira and Slack integrations are optional (configured via .env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COGNITIVE_RESULTS_DIR = Path(
    os.getenv("COGNITIVE_RESULTS_DIR", str(BASE_DIR.parent / "cognitive_service" / "results"))
)

SERVICE_NAME = "agentic"
SERVICE_HOST = os.getenv("AGENTIC_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("AGENTIC_PORT", "8002"))

# ── Jira (optional) ───────────────────────────────────────────────────────
JIRA_BASE_URL   = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL      = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN  = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "CORP")

# ── Slack (optional) ──────────────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
