"""
CorpBrain — Cognitive Service (Phase 2)
=========================================
FastAPI microservice that transcribes a meeting audio file
and extracts structured action items using an LLM.

Endpoints
---------
GET  /health                → health check
POST /transcribe            → transcribe audio + extract tasks
GET  /results/{meeting_id} → retrieve a saved result
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SERVICE_NAME, SERVICE_HOST, SERVICE_PORT
from transcriber import transcribe
from extractor import extract_tasks
from utils import save_result, load_result


# ── Request body model ────────────────────────────────────
class TranscribeRequest(BaseModel):
    meeting_id: str  # UUID returned by the ingestion service
    file_path:  str  # path to the saved audio file on disk


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="CorpBrain — Cognitive Service",
    description="Transcribes meeting audio and extracts structured action items.",
    version="0.1.0",
)


# ── GET /health ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": SERVICE_NAME}


# ── POST /transcribe ──────────────────────────────────────
@app.post("/transcribe", status_code=201)
async def transcribe_meeting(req: TranscribeRequest):
    """
    Full pipeline:
    1. Confirm the audio file exists on disk
    2. Run Whisper → raw transcript
    3. Send transcript to Gemini → structured JSON tasks
    4. Save result to disk as {meeting_id}_tasks.json
    5. Return the full result
    """
    # 1. File must exist
    audio_path = Path(req.file_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {req.file_path}"
        )

    # 2. Transcribe with Whisper (runs on your RTX 4070)
    try:
        transcript = transcribe(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # 3. Extract tasks with Gemini
    try:
        tasks = extract_tasks(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # 4. Build the full result object
    result = {
        "meeting_id":   req.meeting_id,
        "transcript":   transcript,
        "summary":      tasks.get("summary", ""),
        "action_items": tasks.get("action_items", []),
        "status":       "processed",
    }

    # 5. Save to disk
    save_result(req.meeting_id, result)

    # 6. Return
    return JSONResponse(status_code=201, content=result)


# ── GET /results/{meeting_id} ─────────────────────────────
@app.get("/results/{meeting_id}")
async def get_result(meeting_id: str):
    """Return the saved result for an already-processed meeting."""
    data = load_result(meeting_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for meeting: {meeting_id}"
        )
    return data


# ── Dev runner ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=True)