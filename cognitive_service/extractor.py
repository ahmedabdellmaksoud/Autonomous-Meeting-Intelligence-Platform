"""
CorpBrain — Extractor
======================
Sends the raw transcript to Gemini and asks it to extract
action items, assignees, and deadlines as strict JSON.
"""

import json
import re
import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL

# Guard: fail fast at startup if the API key is missing
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "[extractor] GEMINI_API_KEY is not set. "
        "Check that cognitive_service/.env exists and contains GEMINI_API_KEY=..."
    )

# Configure Gemini once at import time.
genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)

# Regex to strip any ```[lang] ... ``` code fence Gemini might wrap around JSON
_CODE_FENCE_RE = re.compile(r"^```[a-z]*\n?(.*?)```$", re.DOTALL)


EXTRACTION_PROMPT = """
You are an expert meeting analyst. Read the transcript below and extract all action items.

You MUST respond with ONLY a valid JSON object.
No explanation, no markdown fences, no extra text — just raw JSON.

Use this exact structure:
{{
  "summary": "2-3 sentence summary of the meeting",
  "action_items": [
    {{
      "task":     "what needs to be done",
      "assignee": "person responsible (or 'unassigned' if not mentioned)",
      "deadline": "deadline if mentioned (or 'not specified')",
      "priority": "high / medium / low"
    }}
  ]
}}

TRANSCRIPT:
{transcript}
"""


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if Gemini adds them."""
    match = _CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def extract_tasks(transcript: str) -> dict:
    """
    Send *transcript* to Gemini and return a parsed dict of tasks.

    Retries once if Gemini returns malformed JSON.
    Raises ValueError if both attempts fail.
    """
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)

    for attempt in range(2):
        print(f"[extractor] Calling Gemini (attempt {attempt + 1}) ...")
        response = _gemini.generate_content(prompt)
        raw = _strip_code_fence(response.text.strip())

        try:
            data = json.loads(raw)
            print(f"[extractor] Got {len(data.get('action_items', []))} action items.")
            return data
        except json.JSONDecodeError:
            print(f"[extractor] Bad JSON on attempt {attempt + 1}, retrying ...")

    raise ValueError("Gemini did not return valid JSON after 2 attempts.")