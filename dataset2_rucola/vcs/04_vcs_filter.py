# train.csv, config.py, label_issues.csv (cleanlab), hard_to_learn.csv (datamaps), removed_examples.csv (ds2)


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import faiss
import Levenshtein

from transformers import AutoTokenizer, AutoModel

from config import (
    SEED,
    DS2_EMBEDDING_MODEL, DS2_EMBEDDING_BATCH_SIZE,
    DEDUP_COSINE_THRESHOLD, DEDUP_KNN_K,
    VER_DEDUP_MIN_VOTES,
)

RESULTS_DIR = "./results_ver_dedup"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

label_names = {0: "unacceptable", 1: "acceptable"}

train_df = pd.read_csv("train.csv")
n_total = len(train_df)

texts = train_df["text"].astype(str).tolist()
labels = train_df["label"].tolist()

KEY_COL = "text"

def get_method_indices(train, csv_path, key_col):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    train_keys = train[key_col].astype(str).str.strip()
    other_keys = df[key_col].astype(str).str.strip()
    other_set = set(other_keys.tolist())

    indices = set()
    for idx, val in train_keys.items():
        if val in other_set:
            indices.add(idx)
    return indices

flagged_cleanlab = get_method_indices(train_df, "label_issues.csv", KEY_COL)
flagged_datamaps = get_method_indices(train_df, "hard_to_learn.csv", KEY_COL)
flagged_ds2      = get_method_indices(train_df, "removed_examples.csv", KEY_COL)

def votes_for(idx):
    return (
        int(idx in flagged_cleanlab)
        + int(idx in flagged_datamaps)
        + int(idx in flagged_ds2)
    )

emb_tokenizer = AutoTokenizer.from_pretrained(DS2_EMBEDDING_MODEL)
emb_model = AutoModel.from_pretrained(DS2_EMBEDDING_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_model = emb_model.to(device)
emb_model.eval()

all_embeddings = []
for i in tqdm(range(0, n_total, DS2_EMBEDDING_BATCH_SIZE), desc="Embedding"):
    batch = texts[i:i + DS2_EMBEDDING_BATCH_SIZE]
    encoded = emb_tokenizer(
        batch,
        padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = emb_model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    emb = (output[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)
    all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy().astype("float32")

del emb_model, emb_tokenizer
torch.cuda.empty_cache()

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

k = min(DEDUP_KNN_K, n_total)
sims, neighbors = index.search(embeddings, k)

pairs = []
seen = set()
for i in range(n_total):
    for rank in range(k):
        j = int(neighbors[i, rank])
        if j == i:
            continue
        s = float(sims[i, rank])
        if s < DEDUP_COSINE_THRESHOLD:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b, s))

to_remove = set()
same_target_pairs = 0
diff_target_pairs = 0
diff_confirmed_pairs = 0
diff_kept_pairs = 0

conflicting_pair_rows = []
has_error_type = "error_type" in train_df.columns
error_types = train_df["error_type"].tolist() if has_error_type else None

for (a, b, s) in pairs:
    if labels[a] == labels[b]:
        same_target_pairs += 1
        to_remove.add(b)
    else:
        diff_target_pairs += 1
        votes_a = votes_for(a)
        votes_b = votes_for(b)
        confirmed_a = votes_a >= VER_DEDUP_MIN_VOTES
        confirmed_b = votes_b >= VER_DEDUP_MIN_VOTES

        action = "kept"
        if confirmed_a and confirmed_b:
            to_remove.add(a)
            to_remove.add(b)
            action = "removed_both"
            diff_confirmed_pairs += 1
        elif confirmed_a:
            to_remove.add(a)
            action = "removed_a"
            diff_confirmed_pairs += 1
        elif confirmed_b:
            to_remove.add(b)
            action = "removed_b"
            diff_confirmed_pairs += 1
        else:
            diff_kept_pairs += 1

        row_info = {
            "idx_a": a, "idx_b": b,
            "cos_sim": round(s, 4),
            "label_a": labels[a], "label_b": labels[b],
            "label_a_name": label_names.get(int(labels[a]), str(labels[a])),
            "label_b_name": label_names.get(int(labels[b]), str(labels[b])),
            "text_a": texts[a], "text_b": texts[b],
            "votes_a": votes_a, "votes_b": votes_b,
            "flagged_cleanlab_a": a in flagged_cleanlab,
            "flagged_datamaps_a": a in flagged_datamaps,
            "flagged_ds2_a":      a in flagged_ds2,
            "flagged_cleanlab_b": b in flagged_cleanlab,
            "flagged_datamaps_b": b in flagged_datamaps,
            "flagged_ds2_b":      b in flagged_ds2,
            "action": action,
            "levenshtein": Levenshtein.distance(texts[a], texts[b]),
        }
        if has_error_type:
            row_info["error_type_a"] = error_types[a]
            row_info["error_type_b"] = error_types[b]
        conflicting_pair_rows.append(row_info)

kept_indices = sorted(set(range(n_total)) - to_remove)
train_clean = train_df.loc[kept_indices].reset_index(drop=True)
train_clean.to_csv("train_ver_dedup.csv", index=False, encoding="utf-8-sig")

if conflicting_pair_rows:
    conflicting_df = pd.DataFrame(conflicting_pair_rows).sort_values("cos_sim", ascending=False).reset_index(drop=True)
    conflicting_path = os.path.join(RESULTS_DIR, "conflicting_pairs.csv")
    conflicting_df.to_csv(conflicting_path, index=False, encoding="utf-8-sig")

removed_path = os.path.join(RESULTS_DIR, "removed_indices.json")
with open(removed_path, "w", encoding="utf-8") as f:
    json.dump(sorted(to_remove), f)

removed_rows = []
removed_in_diff = set()
for row in conflicting_pair_rows:
    if row["action"] in ("removed_a", "removed_both"):
        removed_in_diff.add(row["idx_a"])
    if row["action"] in ("removed_b", "removed_both"):
        removed_in_diff.add(row["idx_b"])

for idx in sorted(to_remove):
    reason = "diff_target_confirmed" if idx in removed_in_diff else "same_target_duplicate"
    r = train_df.iloc[idx].to_dict()
    r["reason"] = reason
    r["votes"] = votes_for(idx)
    removed_rows.append(r)

removed_df = pd.DataFrame(removed_rows)
removed_csv = os.path.join(RESULTS_DIR, "removed_examples.csv")
removed_df.to_csv(removed_csv, index=False, encoding="utf-8-sig")

unique_labels = sorted(train_df["label"].unique())
removed_per_class = {
    label_names.get(int(lbl), str(lbl)): int((train_df.loc[sorted(to_remove), "label"] == lbl).sum())
    for lbl in unique_labels
} if to_remove else {label_names.get(int(lbl), str(lbl)): 0 for lbl in unique_labels}

stats = {
    "method": "Verified Dedup (Two-stage Smart Dedup)",
    "dataset": "RuCoLA (RussianNLP/rucola)",
    "embedding_model": DS2_EMBEDDING_MODEL,
    "cosine_threshold": DEDUP_COSINE_THRESHOLD,
    "knn_k": DEDUP_KNN_K,
    "min_votes": VER_DEDUP_MIN_VOTES,
    "total_train": n_total,
    "n_pairs_found": len(pairs),
    "same_target_pairs": same_target_pairs,
    "diff_target_pairs": diff_target_pairs,
    "diff_confirmed_pairs": diff_confirmed_pairs,
    "diff_kept_pairs": diff_kept_pairs,
    "n_removed": len(to_remove),
    "pct_removed": round(len(to_remove) / n_total * 100, 2),
    "clean_train_size": len(train_clean),
    "removed_per_class": removed_per_class,
    "flagged_by_other_methods": {
        "cleanlab": len(flagged_cleanlab),
        "datamaps": len(flagged_datamaps),
        "ds2": len(flagged_ds2),
    },
}
stats_path = os.path.join(RESULTS_DIR, "ver_dedup_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Same target\n(remove 2nd)",
              f"Diff target\nconfirmed (≥{VER_DEDUP_MIN_VOTES} votes)",
              "Diff target\nnot confirmed (kept)"]
counts = [same_target_pairs, diff_confirmed_pairs, diff_kept_pairs]
colors = ["#2ecc71", "#e74c3c", "#3498db"]
bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(cnt), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of pairs")
ax.set_title("Verified Dedup: Pair Breakdown (RuCoLA)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pair_breakdown.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
class_total = [int((train_df["label"] == lbl).sum()) for lbl in unique_labels]
class_removed = [removed_per_class[label_names.get(int(lbl), str(lbl))] for lbl in unique_labels]
x = np.arange(len(unique_labels))
width = 0.35
ax.bar(x - width / 2, class_total, width, label="Total", alpha=0.6, edgecolor="black")
bars = ax.bar(x + width / 2, class_removed, width, label="Removed",
              alpha=0.85, edgecolor="black", color="tomato")
ax.set_xticks(x)
ax.set_xticklabels([label_names.get(int(l), str(l)) for l in unique_labels])
ax.set_ylabel("Count")
ax.set_title("Verified Dedup: Removed per Class (RuCoLA)", fontsize=13)
ax.legend()
for bar, cnt in zip(bars, class_removed):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_total) * 0.01,
            str(cnt), ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "removal_by_class.png"), dpi=150)
plt.show()

if conflicting_pair_rows:
    pair_max_votes = [max(r["votes_a"], r["votes_b"]) for r in conflicting_pair_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
    ax.hist(pair_max_votes, bins=bins, color="#9b59b6", edgecolor="black", alpha=0.8)
    ax.axvline(VER_DEDUP_MIN_VOTES - 0.5, color="red", linestyle="--",
               label=f"min_votes = {VER_DEDUP_MIN_VOTES}")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Max votes in pair (cleanlab/datamaps/ds2)")
    ax.set_ylabel("Number of diff-target pairs")
    ax.set_title("Diff-target pairs: max votes from other methods (RuCoLA)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "diff_pairs_votes.png"), dpi=150)
    plt.show()

print(f"\n{'=' * 60}")
print("ИТОГО:")
print(f"  Исходный train:        {n_total}")
print(f"  Same-target пар:       {same_target_pairs}")
print(f"  Diff-target пар:       {diff_target_pairs} "
      f"(подтверждены {diff_confirmed_pairs}, оставлены {diff_kept_pairs})")
print(f"  Удалено:               {len(to_remove)} ({len(to_remove) / n_total * 100:.2f}%)")
print(f"  Очищенный train:       {len(train_clean)}")
print(f"{'=' * 60}")