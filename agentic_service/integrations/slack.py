"""
CorpBrain — Slack Integration
================================
Sends rich formatted notifications via Slack Incoming Webhook.
Skips gracefully if SLACK_WEBHOOK_URL is not configured.
"""

import requests

from config import SLACK_WEBHOOK_URL

PRIORITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def send_meeting_summary(meeting_id: str, summary: str, tasks: list[dict]) -> bool:
    """
    Send a Slack notification after tickets are created.
    Returns True if message was sent, False otherwise.
    """
    if not SLACK_WEBHOOK_URL:
        print("[slack] SLACK_WEBHOOK_URL not set — skipping notification.")
        return False

    task_lines = "\n".join([
        f"• {PRIORITY_EMOJI.get(t.get('priority', 'medium'), '⚪')} "
        f"*{t.get('jira_ticket', {}).get('key', 'N/A')}* — "
        f"{t.get('task', '')[:60]} "
        f"→ _{t.get('assignee', 'unassigned')}_ "
        f"({t.get('story_points', '?')} pts)"
        for t in tasks
        if t.get("status") == "created"
    ]) or "_No tickets created_"

    message = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "CorpBrain — Meeting Processed ✅"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Summary:*\n{summary}"},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Jira Tickets Created:*\n{task_lines}"},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Meeting ID: `{meeting_id}` | Powered by CorpBrain 🤖"}
                ],
            },
        ]
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=10)
        success  = response.status_code == 200
        if success:
            print("[slack] Notification sent.")
        else:
            print(f"[slack] Failed: {response.status_code} — {response.text}")
        return success
    except Exception as e:
        print(f"[slack] Exception: {e}")
        return False
