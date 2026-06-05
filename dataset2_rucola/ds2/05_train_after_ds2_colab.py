# train_ds2.csv, val.csv, test.csv, config.py

import os
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from config import SEED, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

OUTPUT_DIR  = "./checkpoints_ds2_retrain"
RESULTS_DIR = "./results_ds2"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

label_names = {0: "unacceptable", 1: "acceptable"}

train_df = pd.read_csv("train_ds2.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

unique_labels = sorted(train_df["label"].unique())
label2id = {int(label): int(label) for label in unique_labels}
id2label = {int(label): label_names.get(int(label), str(label)) for label in unique_labels}
num_labels = len(unique_labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
val_dataset   = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro"), "accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
    logging_steps=50, seed=SEED, fp16=torch.cuda.is_available(), report_to="none",
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)
trainer.train()

test_results = trainer.evaluate(test_dataset)
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_labels = np.array(test_dataset["label"])
target_names = [id2label[i] for i in sorted(id2label.keys())]

report = classification_report(test_labels, test_preds, target_names=target_names, digits=4, output_dict=True)

print(f"\nF1-macro: {test_results['eval_f1_macro']:.4f}, Accuracy: {test_results['eval_accuracy']:.4f}")
print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))

ds2_metrics = {
    "model_name": MODEL_NAME, "dataset": "RuCoLA (RussianNLP/rucola)",
    "method": "DS2 (Data Selection via Scoring)",
    "test_f1_macro": round(test_results["eval_f1_macro"], 4),
    "test_accuracy": round(test_results["eval_accuracy"], 4),
    "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH, "seed": SEED, "train_size": len(train_df),
    "per_class_f1": {name: round(report[name]["f1-score"], 4) for name in target_names},
}
with open(os.path.join(RESULTS_DIR, "ds2_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(ds2_metrics, f, indent=2, ensure_ascii=False)

methods = {}
deltas = {}
ds_f1, ds_acc = ds2_metrics["test_f1_macro"], ds2_metrics["test_accuracy"]

for name, path in [("baseline", "baseline_metrics.json"), ("cleanlab", "results_cleanlab/cleanlab_metrics.json"), ("datamaps", "results_datamaps/datamaps_metrics.json")]:
    if os.path.exists(path):
        with open(path, "r") as f:
            m = json.load(f)
        methods[name] = {"f1_macro": m["test_f1_macro"], "accuracy": m["test_accuracy"], "train_size": m.get("train_size", "?")}
        deltas[f"ds2_vs_{name}"] = {"f1_macro": round(ds_f1 - m["test_f1_macro"], 4), "accuracy": round(ds_acc - m["test_accuracy"], 4)}

methods["ds2"] = {"f1_macro": ds_f1, "accuracy": ds_acc, "train_size": len(train_df)}
comparison = {**methods, "delta": deltas}
with open(os.path.join(RESULTS_DIR, "comparison_all_methods.json"), "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
