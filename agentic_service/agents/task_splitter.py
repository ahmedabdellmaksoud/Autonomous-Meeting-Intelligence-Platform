"""
CorpBrain — Task Splitter Agent (LangGraph Node)
=================================================
Takes raw action items from the cognitive service output and
enriches each one with project context before passing them
to the parallel task agents.

This is the second node in the LangGraph pipeline.
"""

try:
    from agents.context_agent import AgentState  # running as service
except ImportError:
    from context_agent import AgentState  # running from notebook


def task_splitter(state: AgentState) -> AgentState:
    """
    LangGraph node: split meeting action items into enriched task objects.

    Input state keys used:  cognitive_data, context
    Output state keys set:  tasks
    """
    raw_items = state["cognitive_data"].get("action_items", [])
    context   = state["context"]

    tasks = []
    for i, item in enumerate(raw_items):
        task = {
            "task_index":   i,
            "task":         item.get("task", ""),
            "assignee":     item.get("assignee", "unassigned"),
            "deadline":     item.get("deadline", "not specified"),
            "priority":     item.get("priority", "medium"),
            "type":         item.get("type", "feature"),
            "context":      item.get("context", ""),
            "project_key":  context.get("project_key", "CORP"),
            "sprint_name":  context.get("sprint_name", ""),
            # Fields to be filled by task_agent:
            "story_points": None,
            "duplicate_of": None,
            "jira_payload": None,
            "jira_ticket":  None,
            "status":       "pending",
        }
        tasks.append(task)

    state["tasks"] = tasks
    print(f"[task_splitter] Split into {len(tasks)} tasks")
    return state
