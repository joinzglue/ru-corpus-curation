# train_datamaps.csv, val.csv, test.csv, config.py, baseline_metrics.json, cleanlab_metrics.json

import os
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from config import (
    SEED, MODEL_NAME, MAX_LENGTH,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
)

OUTPUT_DIR  = "./checkpoints_datamaps_retrain"
RESULTS_DIR = "./results_datamaps"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

label_names = {0: "unacceptable", 1: "acceptable"}

train_df = pd.read_csv("train_datamaps.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

unique_labels = sorted(train_df["label"].unique())
label2id = {int(label): int(label) for label in unique_labels}
id2label = {int(label): label_names.get(int(label), str(label)) for label in unique_labels}
num_labels = len(unique_labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
val_dataset   = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
test_dataset  = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":  f1_score(labels, preds, average="macro"),
        "accuracy":  accuracy_score(labels, preds),
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(test_dataset)

test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_labels = np.array(test_dataset["label"])

target_names = [id2label[i] for i in unique_labels]

report = classification_report(
    test_labels, test_preds,
    target_names=target_names,
    digits=4,
    output_dict=True,
)

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (ПОСЛЕ DATAMAPS)")
print(f"{'=' * 60}")
print(f"F1-macro:  {test_results['eval_f1_macro']:.4f}")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))

datamaps_metrics = {
    "model_name": MODEL_NAME,
    "dataset": "RussianNLP/RuCoLA",
    "method": "DataMaps (Dataset Cartography)",
    "test_f1_macro": round(test_results["eval_f1_macro"], 4),
    "test_accuracy": round(test_results["eval_accuracy"], 4),
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "train_size": len(train_df),
    "per_class_f1": {
        name: round(report[name]["f1-score"], 4) for name in target_names
    },
}

metrics_path = os.path.join(RESULTS_DIR, "datamaps_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(datamaps_metrics, f, indent=2, ensure_ascii=False)

baseline_metrics = {}
if os.path.exists("baseline_metrics.json"):
    with open("baseline_metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

cleanlab_metrics_loaded = {}
cl_metrics_path = "results_cleanlab/cleanlab_metrics.json"
if os.path.exists(cl_metrics_path):
    with open(cl_metrics_path, "r", encoding="utf-8") as f:
        cleanlab_metrics_loaded = json.load(f)

dm_f1  = datamaps_metrics["test_f1_macro"]
dm_acc = datamaps_metrics["test_accuracy"]
dm_sz  = datamaps_metrics["train_size"]

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ МЕТОДОВ")
print(f"{'=' * 60}")

header = f"{'Метрика':<20}"
row_f1 = f"{'F1-macro':<20}"
row_acc = f"{'Accuracy':<20}"
row_sz = f"{'Train size':<20}"

methods = {}

if baseline_metrics:
    bl_f1  = baseline_metrics.get("test_f1_macro", 0)
    bl_acc = baseline_metrics.get("test_accuracy", 0)
    bl_sz  = baseline_metrics.get("train_size", "?")
    header  += f"{'Baseline':>10}"
    row_f1  += f"{bl_f1:>10.4f}"
    row_acc += f"{bl_acc:>10.4f}"
    row_sz  += f"{bl_sz:>10}"
    methods["baseline"] = {"f1_macro": bl_f1, "accuracy": bl_acc, "train_size": bl_sz}

if cleanlab_metrics_loaded:
    cl_f1  = cleanlab_metrics_loaded.get("test_f1_macro", 0)
    cl_acc = cleanlab_metrics_loaded.get("test_accuracy", 0)
    cl_sz  = cleanlab_metrics_loaded.get("train_size", "?")
    header  += f"{'Cleanlab':>10}"
    row_f1  += f"{cl_f1:>10.4f}"
    row_acc += f"{cl_acc:>10.4f}"
    row_sz  += f"{cl_sz:>10}"
    methods["cleanlab"] = {"f1_macro": cl_f1, "accuracy": cl_acc, "train_size": cl_sz}

header  += f"{'DataMaps':>10}"
row_f1  += f"{dm_f1:>10.4f}"
row_acc += f"{dm_acc:>10.4f}"
row_sz  += f"{dm_sz:>10}"
methods["datamaps"] = {"f1_macro": dm_f1, "accuracy": dm_acc, "train_size": dm_sz}

print(header)
print("-" * len(header))
print(row_f1)
print(row_acc)
print(row_sz)
print()

deltas = {}
if baseline_metrics:
    deltas["datamaps_vs_baseline"] = {
        "f1_macro": round(dm_f1 - bl_f1, 4),
        "accuracy": round(dm_acc - bl_acc, 4),
    }
    print(f"DataMaps vs Baseline:  F1 {dm_f1 - bl_f1:+.4f},  Acc {dm_acc - bl_acc:+.4f}")

if cleanlab_metrics_loaded:
    deltas["datamaps_vs_cleanlab"] = {
        "f1_macro": round(dm_f1 - cl_f1, 4),
        "accuracy": round(dm_acc - cl_acc, 4),
    }
    print(f"DataMaps vs Cleanlab:  F1 {dm_f1 - cl_f1:+.4f},  Acc {dm_acc - cl_acc:+.4f}")

comparison = {**methods, "delta": deltas}
comp_path = os.path.join(RESULTS_DIR, "comparison_all_methods.json")
with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
