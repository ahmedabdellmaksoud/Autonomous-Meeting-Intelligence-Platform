# Meeting Classifier Model

## Overview
Fine-tuned BERT model for binary classification: `meeting` vs `not_meeting`.

Base model: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
Framework: HuggingFace Transformers + PyTorch

## Training

### 1. Prepare dataset
Create a `dataset.csv` with two columns:
```
text,label
"Good morning everyone. Let's go through the sprint backlog...",1
"Hey, I'm calling about your electricity bill...",0
```
Labels: `1` = meeting, `0` = not_meeting

### 2. Train (on Kaggle GPU or locally)
```bash
cd models/meeting_classifier
python train.py --data_path dataset.csv --output_dir ../../checkpoints/meeting_classifier
```

### 3. Verify checkpoint
```bash
ls ../../checkpoints/meeting_classifier/
# config.json  model.safetensors  tokenizer.json  tokenizer_config.json
```

### 4. Test inference
```bash
python predict.py --text "Good morning everyone, let's review the sprint items from last week."
```

## Results (expected after training)
| Metric   | Target |
|----------|--------|
| Accuracy | > 92%  |
| F1 Score | > 0.91 |

## Fallback Behavior
If no checkpoint exists, `cognitive_service/classifier.py` automatically falls
back to Gemini-based classification. The BERT model is optional — the system
works without it, just with a Gemini API call per classification.
