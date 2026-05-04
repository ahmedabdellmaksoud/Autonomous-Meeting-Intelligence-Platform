"""
CorpBrain — Agentic Service API
=================================
LangGraph multi-agent pipeline for autonomous Scrum Master operations.

Pipeline: context_agent → task_splitter → task_agent (parallel) → approval_gate

Endpoints:
  GET  /health
  POST /run          → run pipeline, returns tasks pending approval
  GET  /pending/{id} → get pending tasks for a meeting
  POST /approve      → approve tasks, create Jira tickets, notify Slack
  GET  /results/{id} → retrieve completed agentic result
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from config import SERVICE_NAME, SERVICE_HOST, SERVICE_PORT
from agents.context_agent import context_agent, AgentState
from agents.task_splitter import task_splitter
from agents.task_agent import task_agent
from agents.approval import approval_gate
from integrations.jira import create_all_tickets
from integrations.slack import send_meeting_summary
from utils import load_cognitive_result, save_agentic_result


# ── Build the LangGraph pipeline ──────────────────────────────────────────

_graph = StateGraph(AgentState)
_graph.add_node("context_agent", context_agent)
_graph.add_node("task_splitter", task_splitter)
_graph.add_node("task_agent",    task_agent)
_graph.add_node("approval_gate", approval_gate)

_graph.set_entry_point("context_agent")
_graph.add_edge("context_agent", "task_splitter")
_graph.add_edge("task_splitter", "task_agent")
_graph.add_edge("task_agent",    "approval_gate")
_graph.add_edge("approval_gate", END)

pipeline = _graph.compile()


# ── In-memory store for pending approvals ────────────────────────────────
# In production this would be Redis or a database.
_pending: dict[str, dict] = {}


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="CorpBrain — Agentic Service", version="1.0.0")


class RunRequest(BaseModel):
    meeting_id: str


class ApproveRequest(BaseModel):
    meeting_id:       str
    approved_indices: list[int]  # task_index values to approve


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/run", status_code=201)
async def run_pipeline(req: RunRequest):
    """
    Run the full LangGraph agent pipeline for a meeting.
    Reads cognitive service output, runs all agents, returns tasks
    in 'pending_approval' state for human review.
    """
    cognitive_data = load_cognitive_result(req.meeting_id)
    if cognitive_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cognitive result found for meeting: {req.meeting_id}",
        )
    if not cognitive_data.get("is_meeting"):
        raise HTTPException(
            status_code=400,
            detail="This recording was not classified as a meeting — no tasks to generate.",
        )

    initial_state: AgentState = {
        "meeting_id":     req.meeting_id,
        "cognitive_data": cognitive_data,
        "context":        {},
        "tasks":          [],
        "approved_tasks": [],
        "jira_tickets":   [],
        "error":          None,
    }

    try:
        final_state = pipeline.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    _pending[req.meeting_id] = final_state

    return JSONResponse(
        status_code=201,
        content={
            "meeting_id": req.meeting_id,
            "task_count": len(final_state["tasks"]),
            "tasks":      final_state["tasks"],
            "status":     "pending_approval",
            "next_step":  "POST /approve with {meeting_id, approved_indices: [0,1,2,...]}",
        },
    )


@app.get("/pending/{meeting_id}")
async def get_pending(meeting_id: str):
    """Get the tasks currently awaiting human approval for a meeting."""
    if meeting_id not in _pending:
        raise HTTPException(status_code=404, detail=f"No pending tasks for: {meeting_id}")
    state = _pending[meeting_id]
    return {"meeting_id": meeting_id, "tasks": state["tasks"]}


@app.post("/approve")
async def approve_tasks(req: ApproveRequest):
    """
    Approve specific tasks by index, then:
      1. Create Jira tickets for approved tasks
      2. Send Slack notification
      3. Persist the result
    """
    if req.meeting_id not in _pending:
        raise HTTPException(
            status_code=404,
            detail=f"No pending pipeline for meeting: {req.meeting_id}",
        )

    state     = _pending[req.meeting_id]
    all_tasks = state["tasks"]

    approved = [t for t in all_tasks if t["task_index"] in req.approved_indices]
    if not approved:
        raise HTTPException(status_code=400, detail="No valid task indices provided.")

    # Create Jira tickets (skipped gracefully if not configured)
    created = create_all_tickets(approved)

    # Notify Slack (skipped gracefully if not configured)
    summary = state["cognitive_data"].get("summary", "")
    send_meeting_summary(req.meeting_id, summary, created)

    # Persist result
    result = {
        "meeting_id":      req.meeting_id,
        "approved_count":  len(created),
        "tickets_created": len([t for t in created if t["status"] == "created"]),
        "tickets":         created,
        "status":          "executed",
    }
    save_agentic_result(req.meeting_id, result)
    del _pending[req.meeting_id]

    return result


@app.get("/results/{meeting_id}")
async def get_result(meeting_id: str):
    """Retrieve a completed agentic pipeline result."""
    from pathlib import Path
    from config import RESULTS_DIR
    import json

    path = RESULTS_DIR / f"{meeting_id}_agentic.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No result for: {meeting_id}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=True)
