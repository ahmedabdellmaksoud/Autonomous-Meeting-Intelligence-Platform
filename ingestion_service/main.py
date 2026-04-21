"""
CorpBrain — Ingestion Service (Phase 1)
=========================================
FastAPI microservice that accepts meeting audio/video uploads,
stores them locally, and returns a unique Meeting ID.

Endpoints
---------
GET  /health                 → health-check
POST /upload                 → upload an .mp3 / .wav / .mp4 file
GET  /meetings               → list all uploaded meetings
GET  /meetings/{meeting_id}  → look up a meeting by its ID
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from config import SERVICE_NAME, SERVICE_HOST, SERVICE_PORT, UPLOAD_DIR, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from utils import generate_meeting_id, validate_file_extension, save_upload

# ── In-memory registry of uploaded meetings ───────────────
# Maps meeting_id → metadata dict. Good enough for MVP;
# a database will replace this once we need persistence.
meetings: dict[str, dict] = {}

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="CorpBrain — Ingestion Service",
    description="Accepts meeting recordings and stores them for downstream processing.",
    version="0.1.0",
)


# ── GET /health ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Return service health status."""
    return {"status": "ok", "service": SERVICE_NAME}


# ── POST /upload ──────────────────────────────────────────
@app.post("/upload", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    """
    Accept an audio/video file upload.

    Validates the file extension, checks size, saves the file to
    temp_uploads/, and returns a unique Meeting ID.
    """
    # 1. Validate extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.filename}'. "
                f"Allowed types: .mp3, .wav, .mp4"
            ),
        )

    # 2. Check file size
    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    # 3. Generate a unique Meeting ID
    meeting_id = generate_meeting_id()

    # 4. Save to disk
    saved_path = await save_upload(file, meeting_id)

    # 5. The stored filename (relative name on disk)
    stored_as = saved_path.name

    # 6. Register in our in-memory store
    meetings[meeting_id] = {
        "meeting_id": meeting_id,
        "filename":   file.filename,
        "stored_as":  stored_as,
        "status":     "uploaded",
    }

    # 7. Respond with explicit 201
    return JSONResponse(
        status_code=201,
        content={
            "meeting_id": meeting_id,
            "filename":   file.filename,
            "stored_as":  stored_as,
            "status":     "uploaded",
            "message":    "File uploaded successfully. Ready for processing.",
        },
    )


# ── GET /meetings ─────────────────────────────────────────
@app.get("/meetings")
async def list_meetings():
    """Return a list of all uploaded meetings."""
    return {
        "total":    len(meetings),
        "meetings": list(meetings.values()),
    }


# ── GET /meetings/{meeting_id} ────────────────────────────
@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Return metadata for a previously uploaded meeting."""
    if meeting_id not in meetings:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return meetings[meeting_id]


# ── Dev entry-point ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=True)