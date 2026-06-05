# train_cleanlab.csv, val.csv, test.csv, config.py, baseline_metrics.json 
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

OUTPUT_DIR  = "./checkpoints_cleanlab"
RESULTS_DIR = "./results_cleanlab"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("train_cleanlab.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

unique_labels = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

for df in [train_df, val_df, test_df]:
    df["label_id"] = df["label"].map(label2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def make_dataset(df):
    ds = Dataset.from_pandas(df[["premise", "hypothesis", "label_id"]], preserve_index=False)
    ds = ds.rename_column("label_id", "label")
    return ds.map(tokenize, batched=True)


train_dataset = make_dataset(train_df)
val_dataset   = make_dataset(val_df)
test_dataset  = make_dataset(test_df)

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

report = classification_report(
    test_labels, test_preds,
    target_names=unique_labels,
    digits=4,
    output_dict=True,
)

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (ПОСЛЕ CLEANLAB)")
print(f"{'=' * 60}")
print(f"F1-macro:  {test_results['eval_f1_macro']:.4f}")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=unique_labels, digits=4))

baseline_metrics = {}
if os.path.exists("baseline_metrics.json"):
    with open("baseline_metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

cleanlab_metrics = {
    "model_name": MODEL_NAME,
    "dataset": "TERRa (RussianNLP/russian_super_glue)",
    "test_f1_macro": round(test_results["eval_f1_macro"], 4),
    "test_accuracy": round(test_results["eval_accuracy"], 4),
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "train_size": len(train_df),
    "per_class_f1": {
        lbl: round(report[lbl]["f1-score"], 4) for lbl in unique_labels
    },
}

metrics_path = os.path.join(RESULTS_DIR, "cleanlab_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(cleanlab_metrics, f, indent=2, ensure_ascii=False)

if baseline_metrics:
    print(f"\n{'=' * 60}")
    print("СРАВНЕНИЕ: BASELINE vs CLEANLAB")
    print(f"{'=' * 60}")

    bl_f1  = baseline_metrics.get("test_f1_macro", 0)
    cl_f1  = cleanlab_metrics["test_f1_macro"]
    bl_acc = baseline_metrics.get("test_accuracy", 0)
    cl_acc = cleanlab_metrics["test_accuracy"]
    bl_sz  = baseline_metrics.get("train_size", "?")
    cl_sz  = cleanlab_metrics["train_size"]

    print(f"{'Метрика':<20} {'Baseline':>10} {'Cleanlab':>10} {'Δ':>10}")
    print(f"{'-' * 50}")
    print(f"{'F1-macro':<20} {bl_f1:>10.4f} {cl_f1:>10.4f} {cl_f1 - bl_f1:>+10.4f}")
    print(f"{'Accuracy':<20} {bl_acc:>10.4f} {cl_acc:>10.4f} {cl_acc - bl_acc:>+10.4f}")
    print(f"{'Train size':<20} {bl_sz:>10} {cl_sz:>10}")
    print()

    comparison = {
        "baseline": {"f1_macro": bl_f1, "accuracy": bl_acc, "train_size": bl_sz},
        "cleanlab": {"f1_macro": cl_f1, "accuracy": cl_acc, "train_size": cl_sz},
        "delta": {
            "f1_macro": round(cl_f1 - bl_f1, 4),
            "accuracy": round(cl_acc - bl_acc, 4),
        },
    }

    comp_path = os.path.join(RESULTS_DIR, "comparison_baseline_vs_cleanlab.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
