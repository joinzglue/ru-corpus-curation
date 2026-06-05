# train.csv, config.py, label_issues.csv
import os
import json
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
from config import (
    SEED, MODEL_NAME, MAX_LENGTH,
    BATCH_SIZE, LEARNING_RATE, DATAMAPS_EPOCHS,
    EASY_KEEP_RATIO,
)

class DataMapsCallback(TrainerCallback):

    def __init__(self, n_epochs: int):
        self.n_epochs = n_epochs
        self.epoch_logits = [] # list[np.ndarray], shape (n_train, num_labels)
        self.trainer_ref = None
        self.train_dataset_ref = None
        self.epoch_bar = tqdm(total=n_epochs, desc="DataMaps — эпохи", unit="ep")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer_ref is None or self.train_dataset_ref is None:
            return
        predictions = self.trainer_ref.predict(self.train_dataset_ref)
        self.epoch_logits.append(predictions.predictions.copy())
        epoch_num = len(self.epoch_logits)
        self.epoch_bar.set_postfix({"epoch": f"{epoch_num}/{self.n_epochs}"})
        self.epoch_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()

def compute_forgetfulness(correctness_trend):
    learnt = False
    times_forgotten = 0
    for is_correct in correctness_trend:
        if not learnt and is_correct:
            learnt = True
        elif learnt and not is_correct:
            learnt = False
            times_forgotten += 1
    return times_forgotten

RESULTS_DIR = "./results_datamaps"
CHECKPOINT_DIR = "./checkpoints_datamaps"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("train.csv")
text_col = train_df.columns[0]
label_col = [c for c in train_df.columns if c != text_col][0]

unique_labels = sorted(train_df[label_col].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

train_df["label_id"] = train_df[label_col].map(label2id)
labels = train_df["label_id"].values

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples[text_col],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_ds = Dataset.from_pandas(
    train_df[[text_col, "label_id"]].rename(columns={"label_id": "label"}),
    preserve_index=False,
)
train_ds = train_ds.map(tokenize, batched=True, desc="Токенизация train")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=DATAMAPS_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,
    disable_tqdm=True,
    logging_steps=99999,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

datamaps_cb = DataMapsCallback(n_epochs=DATAMAPS_EPOCHS)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    callbacks=[datamaps_cb],
)

datamaps_cb.trainer_ref = trainer
datamaps_cb.train_dataset_ref = train_ds

trainer.train()

n_train = len(train_df)
n_epochs = len(datamaps_cb.epoch_logits)

confidence = np.zeros(n_train)
variability = np.zeros(n_train)
correctness = np.zeros(n_train)
forgetfulness = np.zeros(n_train)

for i in tqdm(range(n_train), desc="Метрики DataMaps", unit="ex"):
    true_probs_trend = []
    correctness_trend = []

    for epoch in range(n_epochs):
        logits_i = datamaps_cb.epoch_logits[epoch][i]
        probs = torch.softmax(torch.tensor(logits_i), dim=-1).numpy()
        true_prob = float(probs[labels[i]])
        true_probs_trend.append(true_prob)
        correctness_trend.append(int(np.argmax(logits_i) == labels[i]))

    confidence[i] = np.mean(true_probs_trend)
    variability[i] = np.std(true_probs_trend)
    correctness[i] = sum(correctness_trend)
    forgetfulness[i] = compute_forgetfulness(correctness_trend)

train_df["confidence"] = confidence
train_df["variability"] = variability
train_df["correctness"] = correctness
train_df["forgetfulness"] = forgetfulness

print(f"\nСтатистика метрик:")
print(f"  Confidence:   mean={confidence.mean():.4f}, std={confidence.std():.4f}")
print(f"  Variability:  mean={variability.mean():.4f}, std={variability.std():.4f}")
print(f"  Correctness:  mean={correctness.mean():.2f}/{n_epochs}")
print(f"  Forgetfulness: mean={forgetfulness.mean():.2f}")

conf_median = np.median(confidence)
var_median = np.median(variability)


def categorize(row):
    if row["variability"] > var_median:
        return "ambiguous"
    elif row["confidence"] > conf_median:
        return "easy-to-learn"
    else:
        return "hard-to-learn"


train_df["category"] = train_df.apply(categorize, axis=1)

cat_counts = train_df["category"].value_counts()
print(f"\nКатегории примеров:")
for cat in ["easy-to-learn", "ambiguous", "hard-to-learn"]:
    cnt = cat_counts.get(cat, 0)
    print(f"  {cat}: {cnt} ({cnt / n_train * 100:.1f}%)")

hard_df = train_df[train_df["category"] == "hard-to-learn"]
ambiguous_df = train_df[train_df["category"] == "ambiguous"]
easy_df = train_df[train_df["category"] == "easy-to-learn"]

n_easy_keep = int(len(easy_df) * EASY_KEEP_RATIO)
rng = np.random.RandomState(SEED)
easy_keep_idx = rng.choice(easy_df.index, size=n_easy_keep, replace=False)
easy_keep_df = easy_df.loc[easy_keep_idx]
easy_remove_df = easy_df.drop(easy_keep_idx)

clean_df = pd.concat([ambiguous_df, easy_keep_df], ignore_index=True)
removed_df = pd.concat([hard_df, easy_remove_df], ignore_index=True)

n_hard_removed = len(hard_df)
n_easy_removed = len(easy_remove_df)
n_to_remove = len(removed_df)
pct_removed = n_to_remove / n_train * 100

print(f"  Hard-to-learn удалено: {n_hard_removed}")
print(f"  Easy-to-learn удалено: {n_easy_removed} (оставлено {n_easy_keep}, {EASY_KEEP_RATIO*100:.0f}%)")
print(f"  Ambiguous оставлено:   {len(ambiguous_df)}")
print(f"  Итого удалено:         {n_to_remove} ({pct_removed:.1f}%)")

print(f"\nУдалённые примеры по классам:")
for lbl in unique_labels:
    n_class = (train_df[label_col] == lbl).sum()
    n_removed_class = (removed_df[label_col] == lbl).sum()
    print(f"  {lbl}: {n_removed_class} из {n_class} ({n_removed_class / n_class * 100:.1f}%)")

clean_export = clean_df[[text_col, label_col]].copy()
clean_export.to_csv("train_datamaps.csv", index=False, encoding="utf-8-sig")
print(f"\nОчищенный train: {len(clean_export)} примеров")

removed_export = removed_df[[text_col, label_col, "confidence",
                              "variability", "correctness", "forgetfulness",
                              "category"]].copy()
removed_export = removed_export.sort_values("confidence")
removed_export.to_csv(
    os.path.join(RESULTS_DIR, "removed_examples.csv"),
    index=False, encoding="utf-8-sig",
)
print(f"Удалённые примеры: {len(removed_export)}")

full_metrics_export = train_df[[text_col, label_col, "confidence",
                                 "variability", "correctness", "forgetfulness",
                                 "category"]].copy()
full_metrics_export.to_csv(
    os.path.join(RESULTS_DIR, "train_dynamics_metrics.csv"),
    index=False, encoding="utf-8-sig",
)

max_corr = correctness.max() if correctness.max() > 0 else 1
corr_frac = correctness / max_corr
train_df["corr_frac"] = corr_frac
train_df["correct."] = [f"{x:.1f}" for x in corr_frac]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, width_ratios=[5, 1], hspace=0.3, wspace=0.3)

ax0 = fig.add_subplot(gs[:, 0])

num_hues = len(train_df["correct."].unique())
pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

plot_df = train_df.copy()
sns.scatterplot(
    x="variability", y="confidence",
    hue="correct.", palette=pal,
    s=10, alpha=0.5,
    data=plot_df, ax=ax0,
)

bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction",
             fontsize=14, color="black", va="center", ha="center",
             rotation=350, bbox=bb("black"))
ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction",
             fontsize=14, color="black", va="center", ha="center",
             bbox=bb("red"))
ax0.annotate("hard-to-learn", xy=(0.35, 0.2), xycoords="axes fraction",
             fontsize=14, color="black", va="center", ha="center",
             bbox=bb("blue"))

ax0.set_xlabel("Variability (std P(gold label))", fontsize=12)
ax0.set_ylabel("Confidence (mean P(gold label))", fontsize=12)
ax0.legend(title="Correctness", loc="upper right", fontsize=9)

ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(confidence, bins=30, color="#622a87", edgecolor="black", alpha=0.7)
ax1.set_xlabel("Confidence", fontsize=10)
ax1.set_ylabel("Кол-во", fontsize=10)

ax2 = fig.add_subplot(gs[1, 1])
ax2.hist(variability, bins=30, color="teal", edgecolor="black", alpha=0.7)
ax2.set_xlabel("Variability", fontsize=10)
ax2.set_ylabel("Кол-во", fontsize=10)

ax3 = fig.add_subplot(gs[2, 1])
corr_vals = sorted(train_df["correct."].unique())
corr_counts = [int((train_df["correct."] == v).sum()) for v in corr_vals]
ax3.bar(corr_vals, corr_counts, color="#86bf91", edgecolor="black", alpha=0.7)
ax3.set_xlabel("Correctness", fontsize=10)
ax3.set_ylabel("Кол-во", fontsize=10)

fig.suptitle(f"DataMap — Dvach ({MODEL_NAME})", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "datamap.png"), dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

categories = ["easy-to-learn", "ambiguous", "hard-to-learn"]
colors_cat = ["#2ecc71", "#f39c12", "#e74c3c"]
counts_cat = [int(cat_counts.get(c, 0)) for c in categories]

axes[0].bar(categories, counts_cat, color=colors_cat, edgecolor="black", alpha=0.8)
for i, (c, cnt) in enumerate(zip(categories, counts_cat)):
    axes[0].text(i, cnt + 50, str(cnt), ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Кол-во примеров")
axes[0].set_title("Распределение категорий DataMaps")

hard_df = train_df[train_df["category"] == "hard-to-learn"]
class_hard_counts = [int((hard_df[label_col] == lbl).sum()) for lbl in unique_labels]
axes[1].barh(unique_labels, class_hard_counts, edgecolor="black", alpha=0.7, color="#e74c3c")
for i, cnt in enumerate(class_hard_counts):
    axes[1].text(cnt + 1, i, str(cnt), va="center", fontsize=11)
axes[1].set_xlabel("Кол-во hard-to-learn примеров")
axes[1].set_title("Hard-to-learn по классам")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "category_distribution.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for cat, color in [("easy-to-learn", "#2ecc71"), ("ambiguous", "#f39c12"), ("hard-to-learn", "#e74c3c")]:
    mask = train_df["category"] == cat
    ax.hist(confidence[mask.values], bins=30, alpha=0.5, color=color, edgecolor="black",
            label=f"{cat} ({mask.sum()})")
ax.set_xlabel("Confidence (mean P(gold label))", fontsize=12)
ax.set_ylabel("Кол-во примеров", fontsize=12)
ax.set_title("Распределение Confidence по категориям — DataMaps (ru_emotion_dvach)")
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"), dpi=150)
plt.show()

overlap_info = {}
cl_issues_path = "label_issues.csv"
if os.path.exists(cl_issues_path):
    cl_issues = pd.read_csv(cl_issues_path)

    removed_keys = set(removed_df[text_col].str.strip())
    cl_keys = set(cl_issues[text_col].str.strip())

    overlap = removed_keys & cl_keys
    only_datamaps = removed_keys - cl_keys
    only_cleanlab = cl_keys - removed_keys

    print(f"  DataMaps удалил: {len(removed_keys)}")
    print(f"  Cleanlab удалил: {len(cl_keys)}")
    print(f"  Пересечение:     {len(overlap)} ({len(overlap) / max(len(removed_keys), 1) * 100:.1f}% от DataMaps)")
    print(f"  Только DataMaps: {len(only_datamaps)}")
    print(f"  Только Cleanlab: {len(only_cleanlab)}")

    overlap_info = {
        "datamaps_removed": len(removed_keys),
        "cleanlab_removed": len(cl_keys),
        "overlap": len(overlap),
        "overlap_pct_of_datamaps": round(len(overlap) / max(len(removed_keys), 1) * 100, 2),
        "only_datamaps": len(only_datamaps),
        "only_cleanlab": len(only_cleanlab),
    }
else:
    print("\nlabel_issues.csv не найден")

print("\nТоп-20 hard-to-learn примеров:")
top_hard = removed_export[removed_export["category"] == "hard-to-learn"].head(20)
for i, (_, row) in enumerate(top_hard.iterrows(), 1):
    text_preview = str(row[text_col])[:100].replace("\n", " ")
    print(f"  {i:2d}. [{row[label_col]}] conf={row['confidence']:.4f} var={row['variability']:.4f}")
    print(f"      {text_preview}...")

dm_stats = {
    "method": "DataMaps (Dataset Cartography)",
    "dataset": "Kostya165/ru_emotion_dvach",
    "model_name": MODEL_NAME,
    "datamaps_epochs": DATAMAPS_EPOCHS,
    "total_train": n_train,
    "easy_keep_ratio": EASY_KEEP_RATIO,
    "n_removed": n_to_remove,
    "n_hard_removed": n_hard_removed,
    "n_easy_removed": n_easy_removed,
    "n_easy_kept": n_easy_keep,
    "pct_removed": round(pct_removed, 2),
    "clean_train_size": len(clean_df),
    "category_counts": {
        cat: int(cat_counts.get(cat, 0)) for cat in categories
    },
    "removed_per_class": {
        lbl: int((removed_df[label_col] == lbl).sum())
        for lbl in unique_labels
    },
    "mean_confidence": round(float(confidence.mean()), 4),
    "std_confidence": round(float(confidence.std()), 4),
    "mean_variability": round(float(variability.mean()), 4),
    "std_variability": round(float(variability.std()), 4),
    "mean_correctness": round(float(correctness.mean()), 4),
    "mean_forgetfulness": round(float(forgetfulness.mean()), 4),
}

if overlap_info:
    dm_stats["overlap_with_cleanlab"] = overlap_info

with open(os.path.join(RESULTS_DIR, "datamaps_stats.json"), "w", encoding="utf-8") as f:
    json.dump(dm_stats, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print(f"  Исходный train:    {n_train}")
print(f"  Hard-to-learn:     -{n_hard_removed}")
print(f"  Easy-to-learn:     -{n_easy_removed} (оставлено {n_easy_keep})")
print(f"  Ambiguous:         {len(ambiguous_df)} (все)")
print(f"  Итого удалено:     {n_to_remove} ({pct_removed:.1f}%)")
print(f"  Очищенный train:   {len(clean_df)}")
print(f"{'=' * 60}")
