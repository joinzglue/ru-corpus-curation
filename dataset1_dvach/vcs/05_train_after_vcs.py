# train_ver_dedup.csv, val.csv, test.csv, config.py, all_results.json (для сравнения)

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

OUTPUT_DIR  = "./checkpoints_ver_dedup"
RESULTS_DIR = "./results_ver_dedup"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("train_ver_dedup.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

text_col = train_df.columns[0]
label_col = "label" if "label" in train_df.columns else [c for c in train_df.columns if c != text_col][0]

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

target_names = [str(l) for l in unique_labels]
report = classification_report(test_labels, test_preds, target_names=target_names, digits=4, output_dict=True)

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (ПОСЛЕ VERIFIED DEDUP)")
print(f"{'=' * 60}")
print(f"F1-macro:  {test_results['eval_f1_macro']:.4f}")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))

ver_dedup_metrics = {
    "model_name": MODEL_NAME,
    "dataset": "Dvach (Kostya165/ru_emotion_dvach)",
    "method": "Verified Dedup (Two-stage Smart Dedup)",
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

metrics_path = os.path.join(RESULTS_DIR, "ver_dedup_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(ver_dedup_metrics, f, indent=2, ensure_ascii=False)
print(f"Метрики сохранены: {metrics_path}")

my_f1  = ver_dedup_metrics["test_f1_macro"]
my_acc = ver_dedup_metrics["test_accuracy"]
my_sz  = ver_dedup_metrics["train_size"]

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ МЕТОДОВ")
print(f"{'=' * 60}")

header  = f"{'Метрика':<20}"
row_f1  = f"{'F1-macro':<20}"
row_acc = f"{'Accuracy':<20}"
row_sz  = f"{'Train size':<20}"

methods = {}
deltas = {}

ALL_RESULTS_PATH = "all_results.json"
all_results = {}
if os.path.exists(ALL_RESULTS_PATH):
    with open(ALL_RESULTS_PATH, "r", encoding="utf-8") as f:
        all_results = json.load(f)

other_methods = [
    ("Baseline",  "baseline"),
    ("Cleanlab",  "cleanlab"),
    ("DataMaps",  "datamaps"),
    ("DS2",       "ds2"),
    ("Intersect", "hybrid_intersection"),
    ("Union",     "hybrid_union"),
    ("Dedup",     "dedup"),
]

for name, key in other_methods:
    if key in all_results:
        m = all_results[key]
        o_f1  = m.get("f1_macro", 0)
        o_acc = m.get("accuracy", 0)
        o_sz  = m.get("train_size", "?")
        header  += f"{name:>12}"
        row_f1  += f"{o_f1:>12.4f}"
        row_acc += f"{o_acc:>12.4f}"
        row_sz  += f"{str(o_sz):>12}"
        methods[key] = {"f1_macro": o_f1, "accuracy": o_acc, "train_size": o_sz}
        deltas[f"ver_dedup_vs_{key}"] = {
            "f1_macro": round(my_f1 - o_f1, 4),
            "accuracy": round(my_acc - o_acc, 4),
        }

header  += f"{'VerDedup':>12}"
row_f1  += f"{my_f1:>12.4f}"
row_acc += f"{my_acc:>12.4f}"
row_sz  += f"{my_sz:>12}"
methods["ver_dedup"] = {"f1_macro": my_f1, "accuracy": my_acc, "train_size": my_sz}

print(header)
print("-" * len(header))
print(row_f1)
print(row_acc)
print(row_sz)
print()

for key, d in deltas.items():
    print(f"  {key}: F1 {d['f1_macro']:+.4f}, Acc {d['accuracy']:+.4f}")

all_results["ver_dedup"] = {"f1_macro": my_f1, "accuracy": my_acc, "train_size": my_sz}
with open(ALL_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nall_results.json обновлён: добавлен ver_dedup")

comparison = {**methods, "delta": deltas}
comp_path = os.path.join(RESULTS_DIR, "comparison_ver_dedup.json")
with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
print(f"Сравнение сохранено: {comp_path}")

print("\nГотово!")