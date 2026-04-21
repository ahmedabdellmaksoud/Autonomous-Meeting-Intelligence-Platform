"""
Phase 1 Tests – Ingestion Service
Run with:  pytest test_ingestion.py -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from main import app
from config import UPLOAD_DIR

client = TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────────────
def fake_file(name: str, content: bytes = b"fake audio data") -> dict:
    return {"file": (name, io.BytesIO(content), "application/octet-stream")}


# ── Tests ──────────────────────────────────────────────────────────────────────
class TestHealthCheck:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestUploadEndpoint:
    def test_upload_mp3_returns_201(self):
        r = client.post("/upload", files=fake_file("meeting.mp3"))
        assert r.status_code == 201
        data = r.json()
        assert "meeting_id" in data
        assert len(data["meeting_id"]) == 36          # UUID length
        assert data["stored_as"].endswith(".mp3")

    def test_upload_wav_returns_201(self):
        r = client.post("/upload", files=fake_file("standup.wav"))
        assert r.status_code == 201
        assert r.json()["stored_as"].endswith(".wav")

    def test_upload_mp4_returns_201(self):
        r = client.post("/upload", files=fake_file("video_call.mp4"))
        assert r.status_code == 201
        assert r.json()["stored_as"].endswith(".mp4")

    def test_each_upload_gets_unique_meeting_id(self):
        r1 = client.post("/upload", files=fake_file("a.mp3"))
        r2 = client.post("/upload", files=fake_file("b.mp3"))
        assert r1.json()["meeting_id"] != r2.json()["meeting_id"]

    def test_file_is_actually_saved_to_disk(self):
        r = client.post("/upload", files=fake_file("saved.mp3"))
        stored_as = r.json()["stored_as"]
        assert (UPLOAD_DIR / stored_as).exists()

    def test_upload_pdf_returns_400(self):
        r = client.post("/upload", files=fake_file("report.pdf"))
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]

    def test_upload_txt_returns_400(self):
        r = client.post("/upload", files=fake_file("notes.txt"))
        assert r.status_code == 400

    def test_upload_without_file_returns_422(self):
        r = client.post("/upload")
        assert r.status_code == 422   # FastAPI validation error


class TestListMeetings:
    def test_list_meetings_returns_200(self):
        r = client.get("/meetings")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "meetings" in data
        assert isinstance(data["meetings"], list)

    def test_list_meetings_reflects_new_upload(self):
        before = client.get("/meetings").json()["total"]
        client.post("/upload", files=fake_file("extra.mp3"))
        after = client.get("/meetings").json()["total"]
        assert after == before + 1
