"""
CorpBrain — Extractor Model Training
========================================
Fine-tunes a T5 model to extract structured JSON from meeting transcripts.

Model: google/flan-t5-base  (can handle seq2seq tasks out of the box)
Task:  "extract meeting context: <transcript>" → <json string>

Run on Kaggle with dual T4 GPUs for best results.

Usage:
    python train.py --data_path extractor_dataset.csv --output_dir ../../checkpoints/extractor
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from dataset import ExtractionDataset, build_sample_dataset


def train(
    data_path:  str,
    output_dir: str,
    model_name: str = "google/flan-t5-base",
):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:     {device}")
    print(f"Base model: {model_name}")
    print(f"Data:       {data_path}")
    print(f"Output:     {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    full_dataset = ExtractionDataset(data_path, tokenizer)
    split        = int(0.85 * len(full_dataset))
    train_ds, eval_ds = torch.utils.data.random_split(
        full_dataset, [split, len(full_dataset) - split]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=256,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to: {output_dir}")
    print("Use this path in cognitive_service/extractor.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="extractor_dataset.csv")
    parser.add_argument("--output_dir", default="../../checkpoints/extractor")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--build_sample", action="store_true",
                        help="Generate a sample dataset before training")
    args = parser.parse_args()

    if args.build_sample or not Path(args.data_path).exists():
        build_sample_dataset(args.data_path)

    train(args.data_path, args.output_dir, args.model_name)
