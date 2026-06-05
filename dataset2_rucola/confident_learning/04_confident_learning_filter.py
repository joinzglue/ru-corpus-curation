# train.csv, val.csv, config.py
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from cleanlab.count import compute_confident_joint
from config import (
    SEED, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, N_FOLDS,
)


class EpochProgressCallback(TrainerCallback):
    def __init__(self, fold_idx: int, n_folds: int, n_epochs: int):
        self.fold_idx = fold_idx
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.epoch_bar = tqdm(
            total=n_epochs,
            desc=f"  Фолд {fold_idx + 1}/{n_folds} — эпохи",
            unit="ep",
            leave=True,
        )
        self._epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0
        loss = metrics.get("eval_loss", float("nan"))
        self.epoch_bar.set_postfix({"eval_loss": f"{loss:.4f}", "t": f"{elapsed:.0f}s"})
        self.epoch_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()


RESULTS_DIR = "./results_cleanlab"
CL_CHECKPOINT_DIR = "./checkpoints_cl_cv"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

label_names = {0: "unacceptable", 1: "acceptable"}

train_df = pd.read_csv("train.csv")

unique_labels = sorted(train_df["label"].unique())
num_labels = len(unique_labels)
label2id = {int(label): int(label) for label in unique_labels}
id2label = {int(label): label_names.get(int(label), str(label)) for label in unique_labels}
labels = train_df["label"].values

print(f"Train: {len(train_df)} примеров, {num_labels} классов")
print(f"Классы: {[f'{k} ({v})' for k, v in id2label.items()]}")
print(f"Кросс-валидация: {N_FOLDS} фолдов")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

pred_probs = np.zeros((len(train_df), num_labels), dtype=np.float32)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds_bar = tqdm(
    enumerate(skf.split(train_df, labels)),
    total=N_FOLDS,
    desc="Кросс-валидация",
    unit="fold",
)

for fold_idx, (train_indices, val_indices) in folds_bar:
    folds_bar.set_description(f"Кросс-валидация [фолд {fold_idx + 1}/{N_FOLDS}]")
    print(f"\n{'─' * 40}")
    print(f"Фолд {fold_idx + 1}/{N_FOLDS}  |  train={len(train_indices)}, held-out={len(val_indices)}")

    fold_train_df = train_df.iloc[train_indices]
    fold_val_df = train_df.iloc[val_indices]

    fold_train_ds = Dataset.from_pandas(
        fold_train_df[["text", "label"]],
        preserve_index=False,
    )
    fold_val_ds = Dataset.from_pandas(
        fold_val_df[["text", "label"]],
        preserve_index=False,
    )

    fold_train_ds = fold_train_ds.map(tokenize, batched=True, desc="  токенизация train")
    fold_val_ds = fold_val_ds.map(tokenize, batched=True, desc="  токенизация val")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    fold_output_dir = os.path.join(CL_CHECKPOINT_DIR, f"fold_{fold_idx}")

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        save_only_model=True,
        disable_tqdm=True,
        logging_steps=99999,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    epoch_cb = EpochProgressCallback(fold_idx, N_FOLDS, NUM_EPOCHS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fold_train_ds,
        eval_dataset=fold_val_ds,
        callbacks=[epoch_cb],
    )

    trainer.train()

    predictions = trainer.predict(fold_val_ds)
    logits = predictions.predictions
    fold_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_probs[val_indices] = fold_probs

    del model, trainer
    torch.cuda.empty_cache()

folds_bar.close()

issue_mask = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by=None,
)

quality_scores = get_label_quality_scores(
    labels=labels,
    pred_probs=pred_probs,
)

n_issues = issue_mask.sum()
print(f"Найдено label issues: {n_issues} из {len(train_df)} ({n_issues / len(train_df) * 100:.2f}%)")

confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)

train_df["is_label_issue"] = issue_mask
train_df["label_quality_score"] = quality_scores
train_df["predicted_label"] = pred_probs.argmax(axis=1)

issues_df = train_df[train_df["is_label_issue"]].copy()

print("Ошибки по классам:")
for lbl in unique_labels:
    class_issues = issues_df[issues_df["label"] == lbl]
    n_class = (train_df["label"] == lbl).sum()
    print(f"  {lbl} ({label_names.get(lbl, '?')}): {len(class_issues)} ошибок из {n_class} ({len(class_issues) / n_class * 100:.1f}%)")
    if len(class_issues) > 0:
        top_confused = class_issues["predicted_label"].value_counts().head(3)
        for pred_lbl, cnt in top_confused.items():
            print(f"    → {pred_lbl} ({label_names.get(pred_lbl, '?')}): {cnt}")
print()

print("Ошибки по типу разметки:")
if "error_type" in issues_df.columns:
    error_type_counts = issues_df["error_type"].value_counts()
    for err_type, cnt in error_type_counts.items():
        pct_of_issues = cnt / len(issues_df) * 100
        total_of_type = (train_df["error_type"] == err_type).sum()
        pct_of_type = cnt / total_of_type * 100 if total_of_type > 0 else 0
        print(f"  {err_type}: {cnt} ({pct_of_issues:.1f}% от всех issues, {pct_of_type:.1f}% от примеров этого типа)")
print()

tick_labels = [f"{lbl} ({label_names.get(lbl, '?')})" for lbl in unique_labels]

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    confident_joint,
    annot=True,
    fmt="d",
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cmap="YlOrRd",
    ax=ax,
)
ax.set_xlabel("Предсказанный класс (y*)")
ax.set_ylabel("Данная метка (ỹ)")
ax.set_title("Confident Joint — матрица C(ỹ, y*)")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confident_joint.png"), dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(quality_scores, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(
    quality_scores[issue_mask].max(),
    color="red", linestyle="--", label="Порог (макс. score среди issues)",
)
axes[0].set_xlabel("Label Quality Score")
axes[0].set_ylabel("Кол-во примеров")
axes[0].set_title("Распределение Label Quality Scores")
axes[0].legend()

class_issue_counts = issues_df["label"].value_counts().reindex(unique_labels, fill_value=0)
bar_labels = [f"{lbl} ({label_names.get(lbl, '?')})" for lbl in unique_labels]
axes[1].barh(bar_labels, class_issue_counts.values, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Кол-во label issues")
axes[1].set_title("Ошибки разметки по классам")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "label_quality_analysis.png"), dpi=150)
plt.show()

if "error_type" in issues_df.columns and len(error_type_counts) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    error_type_counts.plot(kind="barh", ax=ax, color="coral", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Кол-во label issues")
    ax.set_title("Label Issues по типу ошибки (RuCoLA)")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "issues_by_error_type.png"), dpi=150)
    plt.show()

clean_train_df = train_df[~train_df["is_label_issue"]][["text", "label", "error_type"]]
clean_train_df.to_csv("train_cleanlab.csv", index=False)

issues_export = issues_df[["text", "label", "error_type", "predicted_label", "label_quality_score"]].copy()
issues_export = issues_export.sort_values("label_quality_score")
issues_export.to_csv(os.path.join(RESULTS_DIR, "label_issues.csv"), index=False, encoding="utf-8-sig")

print("\nТоп-20 наиболее вероятных ошибок разметки:")
top_issues = issues_export.head(20)
for i, (_, row) in enumerate(top_issues.iterrows(), 1):
    text_preview = row["text"][:80].replace("\n", " ")
    given = label_names.get(row["label"], row["label"])
    predicted = label_names.get(row["predicted_label"], row["predicted_label"])
    err = row["error_type"] if str(row["error_type"]) != "0" else "—"
    print(f"  {i:2d}. [{given} → {predicted}] err={err} "
          f"(score={row['label_quality_score']:.4f}) {text_preview}...")

cl_stats = {
    "method": "cleanlab (Confident Learning)",
    "dataset": "RussianNLP/RuCoLA",
    "n_folds": N_FOLDS,
    "total_train": len(train_df),
    "n_label_issues": int(n_issues),
    "pct_label_issues": round(n_issues / len(train_df) * 100, 2),
    "clean_train_size": len(clean_train_df),
    "issues_per_class": {
        f"{lbl} ({label_names.get(lbl, '?')})": int(issues_df[issues_df["label"] == lbl].shape[0])
        for lbl in unique_labels
    },
    "issues_per_error_type": {
        str(err): int(cnt) for err, cnt in error_type_counts.items()
    } if "error_type" in issues_df.columns else {},
    "confident_joint": confident_joint.tolist(),
}

with open(os.path.join(RESULTS_DIR, "cleanlab_stats.json"), "w", encoding="utf-8") as f:
    json.dump(cl_stats, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print("CLEANLAB ФИЛЬТРАЦИЯ ЗАВЕРШЕНА")
print(f"  Исходный train:  {len(train_df)}")
print(f"  Удалено:         {n_issues} ({n_issues / len(train_df) * 100:.2f}%)")
print(f"  Очищенный train: {len(clean_train_df)}")
print(f"{'=' * 60}")
