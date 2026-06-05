# train_union.csv, val.csv, test.csv, config.py, baseline_metrics.json, cleanlab_metrics.json, datamaps_metrics.json, ds2_metrics.json, intersection_metrics.json

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

OUTPUT_DIR  = "./checkpoints_hybrid_union"
RESULTS_DIR = "./results_hybrid"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("train_union.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

text_col = train_df.columns[0]
label_col = [c for c in train_df.columns if c != text_col][0]

unique_labels = sorted(train_df[label_col].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

for df in [train_df, val_df, test_df]:
    df["label_id"] = df[label_col].map(label2id)

print(f"Clean Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Классы ({num_labels}): {unique_labels}")
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def make_dataset(df):
    ds = Dataset.from_pandas(df[[text_col, "label_id"]], preserve_index=False)
    ds = ds.rename_column("label_id", "label")
    return ds.map(tokenize, batched=True)

train_dataset = make_dataset(train_df)
val_dataset   = make_dataset(val_df)
test_dataset  = make_dataset(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
)

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

report = classification_report(test_labels, test_preds, target_names=unique_labels, digits=4, output_dict=True)

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (ПОСЛЕ UNION)")
print(f"{'=' * 60}")
print(f"F1-macro:  {test_results['eval_f1_macro']:.4f}")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=unique_labels, digits=4))

union_metrics = {
    "model_name": MODEL_NAME,
    "dataset": "Dvach (Kostya165/ru_emotion_dvach)",
    "method": "Hybrid (Union >= 1 method)",
    "test_f1_macro": round(test_results["eval_f1_macro"], 4),
    "test_accuracy": round(test_results["eval_accuracy"], 4),
    "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE, "max_length": MAX_LENGTH,
    "seed": SEED, "train_size": len(train_df),
    "per_class_f1": {lbl: round(report[lbl]["f1-score"], 4) for lbl in unique_labels},
}

metrics_path = os.path.join(RESULTS_DIR, "union_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(union_metrics, f, indent=2, ensure_ascii=False)
print(f"Метрики сохранены: {metrics_path}")

my_f1  = union_metrics["test_f1_macro"]
my_acc = union_metrics["test_accuracy"]
my_sz  = union_metrics["train_size"]

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ МЕТОДОВ")
print(f"{'=' * 60}")

header  = f"{'Метрика':<20}"
row_f1  = f"{'F1-macro':<20}"
row_acc = f"{'Accuracy':<20}"
row_sz  = f"{'Train size':<20}"

methods = {}
deltas = {}

for name, path in [("Baseline", "baseline_metrics.json"), ("Cleanlab", "cleanlab_metrics.json"),
                    ("DataMaps", "datamaps_metrics.json"), ("DS2", "ds2_metrics.json"),
                    ("Intersect", "intersection_metrics.json")]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        o_f1  = m.get("test_f1_macro", 0)
        o_acc = m.get("test_accuracy", 0)
        o_sz  = m.get("train_size", "?")
        header  += f"{name:>12}"
        row_f1  += f"{o_f1:>12.4f}"
        row_acc += f"{o_acc:>12.4f}"
        row_sz  += f"{str(o_sz):>12}"
        methods[name.lower()] = {"f1_macro": o_f1, "accuracy": o_acc, "train_size": o_sz}
        deltas[f"union_vs_{name.lower()}"] = {
            "f1_macro": round(my_f1 - o_f1, 4), "accuracy": round(my_acc - o_acc, 4),
        }

header  += f"{'Union':>12}"
row_f1  += f"{my_f1:>12.4f}"
row_acc += f"{my_acc:>12.4f}"
row_sz  += f"{my_sz:>12}"
methods["union"] = {"f1_macro": my_f1, "accuracy": my_acc, "train_size": my_sz}

print(header)
print("-" * len(header))
print(row_f1)
print(row_acc)
print(row_sz)
print()

for key, d in deltas.items():
    print(f"  {key}: F1 {d['f1_macro']:+.4f}, Acc {d['accuracy']:+.4f}")

comparison = {**methods, "delta": deltas}
comp_path = os.path.join(RESULTS_DIR, "comparison_union.json")
with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
print(f"\nСравнение сохранено: {comp_path}")
