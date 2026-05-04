# Extractor Model (T5 Fine-Tuning)

## Overview
Fine-tunes `google/flan-t5-base` to extract structured meeting context from raw transcripts.

**Task type:** Sequence-to-Sequence (seq2seq)  
**Input:** `"extract meeting context: <raw transcript>"`  
**Output:** JSON string with project, sprint, participants, action items, etc.

## Why T5 / Flan-T5?
- Pre-trained on instruction-following tasks → follows JSON output format well
- `flan-t5-base` is only 250M parameters → trains in ~1 hour on Kaggle T4 GPU
- Much smaller than LLaMA but good enough for structured extraction

## Training

### 1. Prepare your dataset
Create `extractor_dataset.csv` with columns `transcript` and `json_output`.

Generate a starter sample:
```bash
cd models/extractor
python train.py --build_sample --data_path extractor_dataset.csv
```

For real quality: collect **50+ real meeting transcripts** and manually write their JSON extractions.
More data = better model. 200+ examples is ideal.

### 2. Train on Kaggle (recommended)
- Go to [kaggle.com](https://kaggle.com) → New Notebook
- Upload your `extractor_dataset.csv` as a dataset
- Copy `train.py` and `dataset.py` into the notebook
- Set accelerator to **GPU T4 x2**
- Run: `python train.py --data_path /kaggle/input/your-dataset/extractor_dataset.csv`

### 3. Download and deploy
```bash
# Download the checkpoint from Kaggle output
# Place it here:
models/checkpoints/extractor/
```

## Using the trained model

### In cognitive_service/extractor.py
The extractor automatically detects your checkpoint if you add this fallback:
```python
EXTRACTOR_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "extractor"

if EXTRACTOR_CHECKPOINT.exists():
    from models.extractor.predict import predict
    return predict(transcript)
```

### CLI
```bash
python predict.py --text "Good morning everyone. Let's review sprint 14..."
```

## Expected Results
| Dataset Size | Extraction Quality |
|---|---|
| 20-50 examples | Basic (gets meeting type, some names) |
| 100-200 examples | Good (action items, deadlines) |
| 500+ examples | Production-ready |
