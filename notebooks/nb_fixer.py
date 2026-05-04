"""Rewrites NB2 and NB3 with all bugs fixed."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

# ══════════════════════════════════════════════════════════════════════════════
# NB2 — fix: rewrite training cell to remove eval_df reference bug
#           and ensure clean variable scope
# ══════════════════════════════════════════════════════════════════════════════
nb2 = nbformat.read("02_bert_classifier_training.ipynb", as_version=4)

# Find and replace the training + eval cells
for i, cell in enumerate(nb2.cells):
    if cell.cell_type != "code":
        continue

    # Fix Cell 4 (TrainingArguments + Trainer.train)
    if "TrainingArguments(" in cell.source and "trainer.train()" in cell.source:
        nb2.cells[i].source = """\
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
)
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1":       round(f1_score(labels, preds, average="binary"), 4),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "not_meeting", 1: "meeting"},
    label2id={"not_meeting": 0, "meeting": 1},
)

training_args = TrainingArguments(
    output_dir="./bert_checkpoints",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="epoch",          # ← transformers >= 4.46
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(), # ← auto GPU/CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting training...")
print(f"  Model:  {MODEL_NAME}")
print(f"  Train:  {len(train_ds)} samples")
print(f"  Val:    {len(eval_ds)} samples")
print(f"  Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU (slow — use Colab/Kaggle GPU)'}")
print()
trainer.train()
"""
        print(f"Fixed Cell {i}: TrainingArguments")

    # Fix Cell 5 (eval + confusion matrix) — y_true from eval_df can fail if
    # eval_df index was reset; safer to use trainer.predict
    elif "confusion_matrix" in cell.source and "y_true" in cell.source:
        nb2.cells[i].source = """\
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results = trainer.evaluate()
print("\\nFinal Evaluation:")
print(f"  Accuracy: {results['eval_accuracy']:.1%}")
print(f"  F1 Score: {results['eval_f1']:.3f}")

# Confusion matrix — use trainer.predict so indices always match
preds_output = trainer.predict(eval_ds)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = preds_output.label_ids        # ← correct source, not eval_df

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["not_meeting", "meeting"],
            yticklabels=["not_meeting", "meeting"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix  (F1={results['eval_f1']:.3f})")
plt.tight_layout()
plt.show()
"""
        print(f"Fixed Cell {i}: confusion matrix (y_true from trainer.predict)")

nbformat.write(nb2, "02_bert_classifier_training.ipynb")
print("NB2 saved ✅\n")


# ══════════════════════════════════════════════════════════════════════════════
# NB3 — fix: Cell 6 imports task_splitter/task_agent/approval which internally
#            do `from agents.context_agent import AgentState`
#            This fails when sys.path only has agentic_service root
#            Fix: add agentic_service to sys.path AND patch imports before loading
# ══════════════════════════════════════════════════════════════════════════════
nb3 = nbformat.read("03_local_agent_pipeline_test.ipynb", as_version=4)

for i, cell in enumerate(nb3.cells):
    if cell.cell_type != "code":
        continue

    # Fix Cell 1 (setup) — add agents subdir to sys.path
    if "sys.path.insert" in cell.source and "agentic_service" in cell.source:
        nb3.cells[i].source = """\
import sys, re, json, concurrent.futures
from pathlib import Path

# Locate project root
PROJECT_ROOT = Path('.').resolve()
if (PROJECT_ROOT / 'cognitive_service').exists():
    ROOT = PROJECT_ROOT
elif (PROJECT_ROOT.parent / 'cognitive_service').exists():
    ROOT = PROJECT_ROOT.parent
else:
    raise RuntimeError("Run from project root or notebooks/ directory")

# Add ALL needed paths — order matters: agents/ must come before agentic_service/
# so that 'from context_agent import ...' resolves to the agents/ module correctly
sys.path.insert(0, str(ROOT / 'cognitive_service'))
sys.path.insert(0, str(ROOT / 'agentic_service' / 'agents'))
sys.path.insert(0, str(ROOT / 'agentic_service'))

print(f"Project root: {ROOT}")
print("Paths configured ✅")
print("  cognitive_service/  → classifier, extractor")
print("  agentic_service/    → config")
print("  agentic_service/agents/ → estimator, context_agent, task_splitter, task_agent, approval")
"""
        print(f"Fixed Cell {i}: sys.path setup")

    # Fix Cell 6 (full chain) — task_splitter.py does `from agents.context_agent import AgentState`
    # which fails even after path fix. Rewrite to import directly without subpackage
    elif "task_splitter" in cell.source and "task_agent" in cell.source and "approval_gate" in cell.source:
        nb3.cells[i].source = """\
# ── Full Agent Chain (direct imports — no LangGraph running) ──────────────────
# Import agents from agents/ directory (added to sys.path in Cell 1)
from context_agent import context_agent, AgentState
from task_splitter import task_splitter
from task_agent    import task_agent
from approval      import approval_gate

# Simulate the cognitive service output
full_state: AgentState = {
    "meeting_id": "test-001",
    "cognitive_data": {
        "project":      "CorpBrain Mobile App",
        "sprint":       "Sprint 14",
        "participants": ["Ahmed", "Sara", "Omar", "Khaled"],
        "decisions":    ["Deploy to staging on Monday"],
        "blockers":     ["CI pipeline too slow"],
        "action_items": [
            {"task": "Fix CI pipeline performance", "assignee": "Khaled",
             "priority": "high",   "type": "admin",   "context": "CI taking 45 min", "deadline": "this week"},
            {"task": "Complete React Native setup", "assignee": "Ahmed",
             "priority": "high",   "type": "feature", "context": "Sprint 14 goal",   "deadline": "Wednesday"},
            {"task": "Fix authentication bug",      "assignee": "Sara",
             "priority": "high",   "type": "bug",     "context": "Blocking all users","deadline": "today"},
            {"task": "Review all open PRs",         "assignee": "Omar",
             "priority": "medium", "type": "admin",   "context": "3 PRs pending",    "deadline": "end of day"},
        ],
        "is_meeting": True,
    },
    "context": {}, "tasks": [], "approved_tasks": [], "jira_tickets": [], "error": None,
}

print("=== RUNNING FULL AGENT CHAIN ===")
print()

print("[1/4] Context Agent...")
full_state = context_agent(full_state)
print(f"  project_key={full_state['context']['project_key']} | sprint={full_state['context']['sprint_name']}")
print()

print("[2/4] Task Splitter...")
full_state = task_splitter(full_state)
print(f"  Tasks split: {len(full_state['tasks'])}")
print()

print("[3/4] Task Agent (parallel story points + Jira payloads)...")
full_state = task_agent(full_state)
print()

print("[4/4] Approval Gate...")
full_state = approval_gate(full_state)
print()

print("=== RESULTS ===")
for t in full_state['tasks']:
    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t['priority'], "⚪")
    print(f"  {emoji} [{t['task_index']}] {t['story_points']}pts | {t['task'][:55]}")
    print(f"       assignee={t['assignee']} | status={t['status']}")
    print(f"       jira: {t['jira_payload']['summary'][:60]}")
    print()

print(f"✅ Pipeline complete — {len(full_state['tasks'])} tasks processed offline")
"""
        print(f"Fixed Cell {i}: full agent chain (direct imports)")

nbformat.write(nb3, "03_local_agent_pipeline_test.ipynb")
print("NB3 saved ✅")
