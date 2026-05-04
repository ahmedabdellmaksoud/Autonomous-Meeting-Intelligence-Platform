"""
CorpBrain — Meeting Classifier Inference
==========================================
Load the trained BERT checkpoint and classify a transcript.

Usage:
    python predict.py --text "Good morning everyone, let's go through sprint items..."
    python predict.py --file transcript.txt
"""

import argparse
from pathlib import Path

from transformers import pipeline


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "meeting_classifier"


def load_classifier():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_DIR}\n"
            "Run models/meeting_classifier/train.py first."
        )
    return pipeline("text-classification", model=str(CHECKPOINT_DIR))


def predict(text: str) -> dict:
    clf    = load_classifier()
    result = clf(text[:512])[0]
    return {
        "label":      result["label"].lower(),
        "confidence": round(result["score"], 3),
        "is_meeting": result["label"].lower() == "meeting",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Direct transcript text")
    group.add_argument("--file", help="Path to transcript .txt file")
    args = parser.parse_args()

    text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    result = predict(text)
    print(f"Label:      {result['label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Is meeting: {result['is_meeting']}")
