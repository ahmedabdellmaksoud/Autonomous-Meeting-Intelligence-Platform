"""
CorpBrain — Extractor Model Inference
========================================
Load fine-tuned T5 model and extract structured JSON from a transcript.

Usage:
    python predict.py --text "Good morning everyone. Let's do our standup..."
    python predict.py --file transcript.txt
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "extractor"


def load_model():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_DIR}\n"
            "Run models/extractor/train.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model     = AutoModelForSeq2SeqLM.from_pretrained(str(CHECKPOINT_DIR))
    model.eval()
    return tokenizer, model


def predict(transcript: str) -> dict:
    """
    Extract structured JSON from transcript using the trained T5 model.
    Returns the same dict format as cognitive_service/extractor.py
    """
    tokenizer, model = load_model()

    input_text = "extract meeting context: " + transcript
    inputs     = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    raw    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = json.loads(raw)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Direct transcript text")
    group.add_argument("--file", help="Path to transcript .txt file")
    args = parser.parse_args()

    text   = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    result = predict(text)

    print("\nExtracted Context:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
