# train.csv, config.py
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
    SEED, MODEL_NAME, MAX_LENGTH,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, N_FOLDS,
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

train_df = pd.read_csv("train.csv")

unique_labels = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

train_df["label_id"] = train_df["label"].map(label2id)
labels = train_df["label_id"].values

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
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

    fold_train_df = train_df.iloc[train_indices]
    fold_val_df = train_df.iloc[val_indices]

    fold_train_ds = Dataset.from_pandas(
        fold_train_df[["premise", "hypothesis", "label_id"]].rename(columns={"label_id": "label"}),
        preserve_index=False,
    )
    fold_val_ds = Dataset.from_pandas(
        fold_val_df[["premise", "hypothesis", "label_id"]].rename(columns={"label_id": "label"}),
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
print()

confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)

train_df["is_label_issue"] = issue_mask
train_df["label_quality_score"] = quality_scores
train_df["predicted_label_id"] = pred_probs.argmax(axis=1)
train_df["predicted_label"] = train_df["predicted_label_id"].map(id2label)

issues_df = train_df[train_df["is_label_issue"]].copy()

print("Ошибки по классам (данная метка → предсказанная):")
for lbl in unique_labels:
    class_issues = issues_df[issues_df["label"] == lbl]
    n_class = (train_df["label"] == lbl).sum()
    print(f"  {lbl}: {len(class_issues)} ошибок из {n_class} ({len(class_issues) / n_class * 100:.1f}%)")
    if len(class_issues) > 0:
        top_confused = class_issues["predicted_label"].value_counts().head(3)
        for pred_lbl, cnt in top_confused.items():
            print(f"    → {pred_lbl}: {cnt}")
print()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    confident_joint,
    annot=True,
    fmt="d",
    xticklabels=unique_labels,
    yticklabels=unique_labels,
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
if n_issues > 0:
    axes[0].axvline(
        quality_scores[issue_mask].max(),
        color="red", linestyle="--", label="Порог (макс. score среди issues)",
    )
axes[0].set_xlabel("Label Quality Score")
axes[0].set_ylabel("Кол-во примеров")
axes[0].set_title("Распределение Label Quality Scores")
axes[0].legend()

class_issue_counts = issues_df["label"].value_counts().reindex(unique_labels, fill_value=0)
axes[1].barh(unique_labels, class_issue_counts.values, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Кол-во label issues")
axes[1].set_title("Ошибки разметки по классам")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "label_quality_analysis.png"), dpi=150)
plt.show()

clean_train_df = train_df[~train_df["is_label_issue"]][["premise", "hypothesis", "label"]]
clean_train_df.to_csv("train_cleanlab.csv", index=False, encoding="utf-8-sig")

issues_export = issues_df[["premise", "hypothesis", "label", "predicted_label", "label_quality_score"]].copy()
issues_export = issues_export.sort_values("label_quality_score")
issues_export.to_csv(os.path.join(RESULTS_DIR, "label_issues.csv"), index=False, encoding="utf-8-sig")

print("\nТоп-20 наиболее вероятных ошибок разметки:")
top_issues = issues_export.head(20)
for i, (_, row) in enumerate(top_issues.iterrows(), 1):
    premise_preview = row["premise"][:60].replace("\n", " ")
    hyp_preview = row["hypothesis"][:40].replace("\n", " ")
    print(f"  {i:2d}. [{row['label']} → {row['predicted_label']}] "
          f"(score={row['label_quality_score']:.4f})")
    print(f"      P: {premise_preview}...")
    print(f"      H: {hyp_preview}...")

cl_stats = {
    "method": "cleanlab (Confident Learning)",
    "dataset": "TERRa (RussianNLP/russian_super_glue)",
    "n_folds": N_FOLDS,
    "total_train": len(train_df),
    "n_label_issues": int(n_issues),
    "pct_label_issues": round(n_issues / len(train_df) * 100, 2),
    "clean_train_size": len(clean_train_df),
    "issues_per_class": {
        lbl: int(issues_df[issues_df["label"] == lbl].shape[0])
        for lbl in unique_labels
    },
    "confident_joint": confident_joint.tolist(),
}

with open(os.path.join(RESULTS_DIR, "cleanlab_stats.json"), "w", encoding="utf-8") as f:
    json.dump(cl_stats, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print("CLEANLAB ФИЛЬТРАЦИЯ ЗАВЕРШЕНА")
print(f"  Исходный train: {len(train_df)}")
print(f"  Удалено:        {n_issues} ({n_issues / len(train_df) * 100:.2f}%)")
print(f"  Очищенный train: {len(clean_train_df)}")
print(f"{'=' * 60}")
