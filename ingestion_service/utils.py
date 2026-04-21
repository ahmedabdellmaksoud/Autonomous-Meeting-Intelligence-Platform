"""
CorpBrain — Ingestion Service Utilities
========================================
Helper functions for meeting-ID generation, file validation, and saving.
"""

import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile

from config import ALLOWED_EXTENSIONS, UPLOAD_DIR


def generate_meeting_id() -> str:
    """Return a new UUID-4 string to uniquely identify a meeting upload."""
    return str(uuid.uuid4())


def validate_file_extension(filename: str) -> bool:
    """Return True if *filename* has an extension in the allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


async def save_upload(file: UploadFile, meeting_id: str) -> Path:
    """
    Stream *file* to disk inside UPLOAD_DIR.

    The saved filename is ``<meeting_id>_<original_filename>`` so that
    every upload is traceable back to its meeting ID at a glance.

    Returns the full Path to the saved file.
    """
    safe_name = f"{meeting_id}_{file.filename}"
    dest = UPLOAD_DIR / safe_name

    # Stream in 1 MB chunks to keep memory usage low for large files.
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    return dest
