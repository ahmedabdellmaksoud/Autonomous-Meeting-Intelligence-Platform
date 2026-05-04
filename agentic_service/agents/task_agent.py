"""
CorpBrain — Task Agent (LangGraph Node)
=========================================
Processes tasks in parallel. Each task gets:
  1. Story points estimated (rule-based)
  2. Jira payload built (professional markdown template)

No external API needed.
"""

import concurrent.futures

try:
    from agents.context_agent import AgentState  # running as service
    from agents.estimator import estimate_story_points
except ImportError:
    from context_agent import AgentState  # running from notebook
    from estimator import estimate_story_points


def _build_jira_payload(task: dict) -> dict:
    """
    Build a professional Jira ticket payload from task fields.
    No API — uses a rich markdown template.
    """
    task_text = task.get("task", "")
    context   = task.get("context", "")
    assignee  = task.get("assignee", "unassigned")
    priority  = task.get("priority", "medium").capitalize()
    t_type    = task.get("type", "feature").capitalize()
    sprint    = task.get("sprint_name", "Current Sprint")
    pts       = task.get("story_points", 3)
    deadline  = task.get("deadline", "not specified")

    summary = task_text[:99] if len(task_text) <= 99 else task_text[:96] + "..."

    description = f"""## Task Description
{task_text}

## Context
{context if context else "No additional context provided."}

## Acceptance Criteria
- [ ] Implementation complete and peer-reviewed
- [ ] Unit tests written and passing
- [ ] Documentation updated if applicable
- [ ] Deployed to staging and verified

## Details
| Field | Value |
|---|---|
| Assignee | {assignee} |
| Priority | {priority} |
| Type | {t_type} |
| Story Points | {pts} |
| Sprint | {sprint} |
| Deadline | {deadline} |
"""

    labels = list({t_type.lower(), priority.lower()})
    if "bug" in task_text.lower():
        labels.append("bug")
    if any(w in task_text.lower() for w in ["api", "endpoint", "rest", "graphql"]):
        labels.append("api")

    return {
        "summary":     summary,
        "description": description,
        "labels":      labels,
        "components":  [],
    }


def _process_single_task(task: dict) -> dict:
    """Estimate + build Jira payload for one task."""
    idx = task["task_index"]
    print(f"[task_agent] Task {idx}: {task['task'][:60]}...")

    task["story_points"] = estimate_story_points(task)
    task["jira_payload"] = _build_jira_payload(task)
    task["status"]       = "ready_for_approval"

    print(f"[task_agent] Task {idx} done — {task['story_points']} pts")
    return task


def task_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: process all tasks in parallel using ThreadPoolExecutor.
    Pure local — no API calls.
    """
    tasks       = state["tasks"]
    max_workers = min(len(tasks), 5) if tasks else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        processed = list(pool.map(_process_single_task, tasks))

    state["tasks"] = processed
    print(f"[task_agent] All {len(processed)} tasks processed.")
    return state
