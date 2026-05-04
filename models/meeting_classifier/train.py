"""
CorpBrain — BERT Meeting Classifier Training Script
=====================================================
Fine-tunes a BERT model to classify transcripts as 'meeting' or 'not_meeting'.
Run this on Kaggle (free GPU) using the notebook: 02_bert_classifier_training.ipynb

Dataset format expected: CSV with columns 'text' and 'label' (0=not_meeting, 1=meeting)

Usage:
    python train.py --data_path dataset.csv --output_dir ../../checkpoints/meeting_classifier
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from dataset import MeetingDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1":       f1_score(labels, predictions, average="binary"),
    }


def train(data_path: str, output_dir: str, model_name: str = "aubmindlab/bert-base-arabertv2"):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {device}")
    print(f"Base model:  {model_name}")
    print(f"Data:        {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not_meeting", 1: "meeting"},
        label2id={"not_meeting": 0, "meeting": 1},
    )

    full_dataset = MeetingDataset(data_path, tokenizer)
    train_size   = int(0.85 * len(full_dataset))
    eval_size    = len(full_dataset) - train_size
    train_ds, eval_ds = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(output_dir / "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save final model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to: {output_dir}")
    print("Use this path as CHECKPOINT_DIR in cognitive_service/classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="dataset.csv")
    parser.add_argument("--output_dir", default="../../checkpoints/meeting_classifier")
    parser.add_argument("--model_name", default="aubmindlab/bert-base-arabertv2")
    args = parser.parse_args()

    train(args.data_path, args.output_dir, args.model_name)
