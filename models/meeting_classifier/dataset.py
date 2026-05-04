"""
CorpBrain — Meeting Classifier Dataset
=========================================
PyTorch Dataset for fine-tuning BERT on meeting classification.

Expected CSV format:
    text,label
    "Good morning everyone, let's review the sprint...",1
    "Hello, I'm calling about your car warranty...",0

Labels: 0 = not_meeting, 1 = meeting
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MeetingDataset(Dataset):
    """
    Dataset for binary meeting classification.
    Tokenizes transcripts and returns tensors for Trainer.
    """

    def __init__(self, csv_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        df          = pd.read_csv(csv_path)
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
