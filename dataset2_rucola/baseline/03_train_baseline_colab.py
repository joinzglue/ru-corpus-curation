# train.csv, val.csv, test.csv, config.py
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
    SEED, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
)

OUTPUT_DIR  = "./checkpoints_baseline"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

label_names = {0: "unacceptable", 1: "acceptable"}
unique_labels = sorted(train_df["label"].unique())
label2id = {int(label): int(label) for label in unique_labels}
id2label = {int(label): label_names.get(int(label), str(label)) for label in unique_labels}
num_labels = len(unique_labels)
print(f"Классы ({num_labels}): {[f'{k} ({v})' for k, v in id2label.items()]}")

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

test_preds_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_output.predictions, axis=-1)
test_labels = np.array(test_dataset["label"])

target_names = [id2label[i] for i in unique_labels]

print(f"F1-macro:  {test_results['eval_f1_macro']:.4f}")
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"\n{classification_report(test_labels, test_preds, target_names=target_names)}")

report = classification_report(
    test_labels, test_preds, target_names=target_names, output_dict=True,
)

metrics = {
    "model_name":      MODEL_NAME,
    "dataset":         "RussianNLP/RuCoLA",
    "test_f1_macro":   round(test_results["eval_f1_macro"], 4),
    "test_accuracy":   round(test_results["eval_accuracy"], 4),
    "num_epochs":      NUM_EPOCHS,
    "batch_size":      BATCH_SIZE,
    "learning_rate":   LEARNING_RATE,
    "max_length":      MAX_LENGTH,
    "seed":            SEED,
    "train_size":      len(train_df),
    "per_class_f1": {
        id2label[label]: round(report[id2label[label]]["f1-score"], 4)
        for label in unique_labels
    },
}

metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
