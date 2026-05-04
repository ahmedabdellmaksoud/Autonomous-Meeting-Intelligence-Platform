"""
CorpBrain — Jira Integration
==============================
Creates Jira issues via the Jira REST API v3.
Requires: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY in .env

If Jira credentials are not configured, creation is skipped
and a warning is printed — safe for local development.
"""

import requests
from requests.auth import HTTPBasicAuth

from config import JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY

PRIORITY_MAP = {"high": "High", "medium": "Medium", "low": "Low"}
TYPE_MAP     = {
    "bug":      "Bug",
    "feature":  "Story",
    "research": "Task",
    "admin":    "Task",
    "other":    "Task",
}


def _is_configured() -> bool:
    return bool(JIRA_BASE_URL and JIRA_EMAIL and JIRA_API_TOKEN)


def create_ticket(task: dict) -> dict:
    """
    Create a single Jira issue from an enriched task dict.
    Returns {"key": ..., "url": ..., "id": ...} or {"key": "SKIPPED", ...}.
    """
    if not _is_configured():
        print(f"[jira] Jira not configured — skipping ticket for: {task['task'][:50]}")
        return {"key": "SKIPPED", "url": "", "id": "", "reason": "Jira credentials not set"}

    auth    = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    base    = f"{JIRA_BASE_URL}/rest/api/3"

    payload = task.get("jira_payload", {})
    body    = {
        "fields": {
            "project":   {"key": JIRA_PROJECT_KEY},
            "summary":   payload.get("summary", task["task"])[:255],
            "issuetype": {"name": TYPE_MAP.get(task.get("type", "feature"), "Story")},
            "priority":  {"name": PRIORITY_MAP.get(task.get("priority", "medium"), "Medium")},
            "description": {
                "type":    "doc",
                "version": 1,
                "content": [{
                    "type":    "paragraph",
                    "content": [{"type": "text", "text": payload.get("description", task.get("context", ""))}],
                }],
            },
        }
    }

    if task.get("assignee") and task["assignee"] != "unassigned":
        body["fields"]["assignee"] = {"name": task["assignee"]}

    response = requests.post(f"{base}/issue", json=body, auth=auth, headers=headers, timeout=10)
    response.raise_for_status()

    result = response.json()
    url    = f"{JIRA_BASE_URL}/browse/{result['key']}"
    print(f"[jira] Created: {result['key']} → {task['task'][:50]}")
    return {"key": result["key"], "url": url, "id": result["id"]}


def create_all_tickets(tasks: list[dict]) -> list[dict]:
    """Create Jira tickets for all given tasks. Returns the enriched task list."""
    for task in tasks:
        try:
            ticket         = create_ticket(task)
            task["jira_ticket"] = ticket
            task["status"]      = "created"
        except Exception as e:
            print(f"[jira] Failed for '{task['task'][:50]}': {e}")
            task["status"] = "failed"
    return tasks
