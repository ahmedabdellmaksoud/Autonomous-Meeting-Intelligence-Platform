"""
CorpBrain — Cognitive Service API
===================================
3-stage pipeline: Transcribe → Classify → Extract
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SERVICE_NAME, SERVICE_HOST, SERVICE_PORT
from transcriber import transcribe
from classifier import classify
from extractor import extract
from utils import save_result, load_result


class ProcessRequest(BaseModel):
    meeting_id: str
    file_path:  str


app = FastAPI(title="CorpBrain — Cognitive Service", version="1.0.0")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/process", status_code=201)
async def process_meeting(req: ProcessRequest):
    """
    Full 3-stage cognitive pipeline:
      1. faster-whisper transcribes the audio file
      2. BERT/Gemini classifies: is this a meeting?
      3. If yes → Gemini extracts project context + action items
    """
    audio_path = Path(req.file_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    # ── Stage 1: Transcribe ───────────────────────────────
    try:
        transcription = transcribe(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    transcript = transcription["text"]

    # ── Stage 2: Classify ─────────────────────────────────
    classification = classify(transcript)

    if classification["label"] == "not_meeting":
        result = {
            "meeting_id":     req.meeting_id,
            "is_meeting":     False,
            "classification": classification,
            "transcript":     transcript,
            "language":       transcription["language"],
            "status":         "rejected",
        }
        save_result(req.meeting_id, result)
        return JSONResponse(status_code=200, content=result)

    # ── Stage 3: Extract (only for confirmed meetings) ────
    try:
        context = extract(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    result = {
        "meeting_id":     req.meeting_id,
        "is_meeting":     True,
        "classification": classification,
        "transcript":     transcript,
        "language":       transcription["language"],
        "language_probability": transcription["language_probability"],
        "segments":       transcription["segments"],
        "summary":        context.get("summary", ""),
        "meeting_type":   context.get("meeting_type", ""),
        "project":        context.get("project", "unknown"),
        "sprint":         context.get("sprint", "unknown"),
        "participants":   context.get("participants", []),
        "decisions":      context.get("decisions", []),
        "blockers":       context.get("blockers", []),
        "action_items":   context.get("action_items", []),
        "status":         "processed",
    }

    save_result(req.meeting_id, result)
    return JSONResponse(status_code=201, content=result)


@app.get("/results/{meeting_id}")
async def get_result(meeting_id: str):
    """Retrieve previously processed cognitive result."""
    data = load_result(meeting_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No result for: {meeting_id}")
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=True)