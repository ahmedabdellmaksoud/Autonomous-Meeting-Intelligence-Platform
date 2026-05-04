"""
CorpBrain — Knowledge Service API
===================================
Store transcripts, search, answer questions, detect duplicates.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SERVICE_NAME, SERVICE_HOST, SERVICE_PORT
from embedder import store_transcript, search_similar
from retriever import answer, check_duplicate


class StoreRequest(BaseModel):
    meeting_id: str
    transcript: str
    metadata:   dict = {}


class DuplicateRequest(BaseModel):
    task_description: str
    threshold:        float = 0.15


app = FastAPI(title="CorpBrain — Knowledge Service", version="1.0.0")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/store", status_code=201)
async def store(req: StoreRequest):
    """Chunk, embed, and store a meeting transcript in ChromaDB."""
    try:
        count = store_transcript(req.meeting_id, req.transcript, req.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {e}")
    return JSONResponse(
        status_code=201,
        content={"meeting_id": req.meeting_id, "chunks_stored": count},
    )


@app.get("/ask")
async def ask(q: str):
    """Answer a natural language question using RAG over stored transcripts."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return answer(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")


@app.post("/check_duplicate")
async def check_dup(req: DuplicateRequest):
    """Check if a task already exists in previous meetings (POST)."""
    try:
        return check_duplicate(req.task_description, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate check failed: {e}")


@app.get("/check_duplicate")
async def check_dup_get(task: str, threshold: float = 0.15):
    """Check if a task already exists in previous meetings (GET convenience)."""
    if not task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty.")
    try:
        return check_duplicate(task, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate check failed: {e}")


@app.get("/search")
async def search(q: str, top_k: int = 5):
    """Raw semantic search — returns matching transcript chunks."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return {"results": search_similar(q, top_k)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=True)
