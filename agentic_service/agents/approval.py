"""
CorpBrain — Approval Gate (LangGraph Node)
===========================================
Marks all tasks as 'pending_approval'.
The API returns tasks to the caller who must explicitly
call POST /approve with selected task indices.

No Jira tickets are created until a human approves.
This is the fourth (final) node in the LangGraph pipeline.
"""

try:
    from agents.context_agent import AgentState  # running as service
except ImportError:
    from context_agent import AgentState  # running from notebook


def approval_gate(state: AgentState) -> AgentState:
    """
    LangGraph node: mark all tasks as pending human approval.

    Input state keys used:  tasks
    Output state keys set:  tasks (status updated), approved_tasks (empty)
    """
    for task in state["tasks"]:
        task["status"] = "pending_approval"

    state["approved_tasks"] = []
    print(f"[approval] {len(state['tasks'])} tasks waiting for human approval.")
    return state
