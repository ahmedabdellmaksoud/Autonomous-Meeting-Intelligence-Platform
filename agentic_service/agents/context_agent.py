"""
CorpBrain — Context Agent (LangGraph Node)
===========================================
Builds project context from cognitive output.
Pure local — no external API needed.
"""

import re
from typing import TypedDict


class AgentState(TypedDict):
    meeting_id:     str
    cognitive_data: dict
    context:        dict
    tasks:          list
    approved_tasks: list
    jira_tickets:   list
    error:          str | None


def _derive_project_key(project_name: str) -> str:
    """Derive a short Jira-style project key from the project name."""
    words = re.findall(r'[A-Za-z]+', project_name)
    if not words:
        return "CORP"
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[0] for w in words[:4]).upper()


def context_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: extract project context directly from cognitive output.
    No API call — uses what extractor.py already found.
    """
    data     = state["cognitive_data"]
    project  = data.get("project", "unknown")
    sprint   = data.get("sprint", "unknown")
    team     = data.get("participants", [])
    blockers = data.get("blockers", [])
    decisions = data.get("decisions", [])

    project_key   = _derive_project_key(project)
    priority_ctx  = (
        f"Blockers: {'; '.join(blockers[:3])}" if blockers else "No blockers reported"
    )

    state["context"] = {
        "project_key":         project_key,
        "sprint_name":         sprint,
        "team_members":        team,
        "priority_context":    priority_ctx,
        "decisions_this_sprint": decisions[:5],
        "tech_stack_hints":    [],
        "definition_of_done":  "not specified",
    }

    print(f"[context_agent] project_key={project_key} | sprint={sprint} | team={team}")
    return state
