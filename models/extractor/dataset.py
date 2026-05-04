"""
CorpBrain — Extractor Model Dataset
=====================================
Prepares (transcript, json_output) pairs for fine-tuning T5.

Input format  (CSV): transcript,json_output
Output format: HuggingFace Dataset with tokenized inputs for seq2seq

Input prefix: "extract meeting context: <transcript>"
Target:       raw JSON string matching the extraction schema
"""

import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ExtractionDataset(Dataset):
    """
    Dataset for T5 fine-tuning on meeting context extraction.

    CSV format:
        transcript,json_output
        "Good morning everyone...", "{...}"
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int  = 512,
        max_target_length: int = 256,
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["transcript", "json_output"])

        # Prefix that tells T5 what task to do
        inputs  = ["extract meeting context: " + t for t in df["transcript"].tolist()]
        targets = df["json_output"].tolist()

        self.input_encodings = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
            return_tensors="pt",
        )

        # Tokenize targets (labels)
        with tokenizer.as_target_tokenizer():
            self.target_encodings = tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=max_target_length,
                return_tensors="pt",
            )

    def __len__(self) -> int:
        return self.input_encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        labels = self.target_encodings["input_ids"][idx].clone()
        # Replace padding token id with -100 so loss ignores padding
        labels[labels == 0] = -100

        return {
            "input_ids":      self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels":         labels,
        }


def build_sample_dataset(output_path: str = "extractor_dataset.csv"):
    """
    Build a small synthetic dataset for initial fine-tuning.
    In a real project: collect real meeting transcripts + their extracted JSON.
    """
    samples = [
        {
            "transcript": "Good morning team. Let's do our standup. Ahmed, you go first. Yesterday I finished the login module. Today I'll work on the dashboard. No blockers. Sara's turn. I completed the API integration. Today I'll fix the authentication bug. My blocker is the missing design specs from Omar.",
            "json_output": json.dumps({
                "summary": "Daily standup covering login module completion, API integration, and authentication bug fix.",
                "meeting_type": "standup",
                "project": "unknown",
                "sprint": "unknown",
                "participants": ["Ahmed", "Sara", "Omar"],
                "decisions": [],
                "blockers": ["missing design specs from Omar"],
                "action_items": [
                    {"task": "work on the dashboard", "assignee": "Ahmed", "deadline": "not specified", "priority": "medium", "type": "feature", "context": "Login module already done"},
                    {"task": "fix the authentication bug", "assignee": "Sara", "deadline": "not specified", "priority": "high", "type": "bug", "context": "Blocked by missing design specs"},
                ]
            })
        },
        {
            "transcript": "Sprint 14 planning meeting. We need to decide on the features for this sprint. The main priorities are: first, the mobile app redesign — Omar estimates 8 story points. Second, the payment gateway integration — Sara estimates 13 points, but we should split that into two tickets. We decided to move the reporting feature to sprint 15. Deadline for this sprint is March 20th.",
            "json_output": json.dumps({
                "summary": "Sprint 14 planning: mobile app redesign and payment gateway integration prioritized. Reporting deferred to sprint 15.",
                "meeting_type": "planning",
                "project": "unknown",
                "sprint": "Sprint 14",
                "participants": ["Omar", "Sara"],
                "decisions": ["Move reporting feature to sprint 15", "Split payment gateway into two tickets"],
                "blockers": [],
                "action_items": [
                    {"task": "mobile app redesign", "assignee": "Omar", "deadline": "March 20th", "priority": "high", "type": "feature", "context": "8 story points estimated"},
                    {"task": "payment gateway integration", "assignee": "Sara", "deadline": "March 20th", "priority": "high", "type": "feature", "context": "Split into two tickets, 13 points total"},
                ]
            })
        },
        {
            "transcript": "Team retrospective for sprint 13. What went well: we delivered all committed stories and the CI pipeline is faster now. What could be improved: code reviews are taking too long, sometimes 3 days. Action items: Khaled will set up a 24-hour code review policy. Ahmed will document the API endpoints this week. We need to improve test coverage — Sara will add unit tests for the payment module.",
            "json_output": json.dumps({
                "summary": "Sprint 13 retrospective. CI improvements praised, slow code review identified as main issue.",
                "meeting_type": "retrospective",
                "project": "unknown",
                "sprint": "Sprint 13",
                "participants": ["Khaled", "Ahmed", "Sara"],
                "decisions": ["Set up 24-hour code review policy"],
                "blockers": ["code reviews taking too long"],
                "action_items": [
                    {"task": "set up a 24-hour code review policy", "assignee": "Khaled", "deadline": "not specified", "priority": "high", "type": "admin", "context": "Code reviews taking up to 3 days"},
                    {"task": "document the API endpoints", "assignee": "Ahmed", "deadline": "this week", "priority": "medium", "type": "admin", "context": "Documentation needed for team"},
                    {"task": "add unit tests for the payment module", "assignee": "Sara", "deadline": "not specified", "priority": "medium", "type": "feature", "context": "Test coverage needs improvement"},
                ]
            })
        },
    ]

    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved: {output_path} ({len(df)} rows)")
    print("Add more rows to improve model quality.")
    return output_path
