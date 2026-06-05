# train.csv, config.py, ds2_compressed_scores.pt, ds2_report.pt, label_issues.csv (cleanlab), removed_examples.csv (datamaps)

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import SEED, DS2_NUM_CLASSES, DS2_CONFIDENCE_PROB, DS2_KEEP_RATIO

RESULTS_DIR = "./results_ds2"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)

train_df = pd.read_csv("train.csv")
compressed_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_compressed_scores.pt"), weights_only=False)
report = torch.load(os.path.join(RESULTS_DIR, "ds2_report.pt"), weights_only=False)

print(f"Train: {len(train_df)} примеров")
print(f"Score errors: {len(report.detection['score_error'])}")
print(f"Score curations: {len(report.curation['score_curation'])}")

def score_curating(report, compressed_scores, confidence_prob):
    scores = list(compressed_scores)
    curated_count = 0
    for sample in report.curation['score_curation']:
        idx, suggested_score, confidence = int(sample[0]), int(sample[1]), sample[2]
        if confidence >= confidence_prob:
            scores[idx] = suggested_score
            curated_count += 1

    print(f"Curated {curated_count} scores (confidence >= {confidence_prob})")
    return scores

curated_scores = score_curating(report, compressed_scores, DS2_CONFIDENCE_PROB)
curated_scores = np.array(curated_scores)

torch.save(curated_scores.tolist(), os.path.join(RESULTS_DIR, "ds2_curated_scores.pt"))

print(f"Compressed distribution: {dict(zip(*np.unique(compressed_scores, return_counts=True)))}")
print(f"Curated distribution:    {dict(zip(*np.unique(curated_scores, return_counts=True)))}")

rare_examples = report.detection.get("rare_example", [])
if rare_examples:
    diversity_scores = np.array([item[1] for item in rare_examples])

M = int(DS2_KEEP_RATIO * len(train_df))
print(f"\nDS2_KEEP_RATIO = {DS2_KEEP_RATIO}")
print(f"Бюджет отбора M = {M} из {len(train_df)}")
sort_keys = np.lexsort((diversity_scores, curated_scores))[::-1]

keep_indices = sort_keys[:M]
remove_indices = sort_keys[M:]

keep_indices_sorted = np.sort(keep_indices)
remove_indices_sorted = np.sort(remove_indices)

n_removed = len(remove_indices)
print(f"Отобрано: {M} примеров (top-{DS2_KEEP_RATIO*100:.0f}%)")
print(f"Удалено:  {n_removed} примеров ({n_removed / len(train_df) * 100:.2f}%)")

print(f"\nРаспределение curated scores в ОТОБРАННЫХ:")
kept_scores = curated_scores[keep_indices_sorted]
print(f"  {dict(zip(*np.unique(kept_scores, return_counts=True)))}")
print(f"Распределение curated scores в УДАЛЁННЫХ:")
removed_scores = curated_scores[remove_indices_sorted]
print(f"  {dict(zip(*np.unique(removed_scores, return_counts=True)))}")

train_clean = train_df.iloc[keep_indices_sorted].reset_index(drop=True)
train_removed = train_df.iloc[remove_indices_sorted].copy()
train_removed["curated_score"] = curated_scores[remove_indices_sorted]
train_removed["compressed_score"] = np.array(compressed_scores)[remove_indices_sorted]
train_removed["diversity_score"] = diversity_scores[remove_indices_sorted]

train_clean.to_csv("train_ds2.csv", index=False)
train_removed.to_csv(os.path.join(RESULTS_DIR, "removed_examples.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, scores, title in zip(
    axes,
    [compressed_scores, curated_scores],
    ["Compressed (до курации)", "Curated (после курации)"],
):
    counter = Counter(scores)
    labels_sorted = sorted(counter.keys())
    ax.bar([str(k) for k in labels_sorted], [counter[k] for k in labels_sorted],
           edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
fig.suptitle("DS2: Score Distribution Before/After Curation (TERRa)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "compressed_vs_curated.png"), dpi=150)
plt.show()

if rare_examples:
    fig, ax = plt.subplots(figsize=(8, 6))
    removed_set_vis = set(remove_indices_sorted.tolist())
    colors = ["red" if i in removed_set_vis else "steelblue" for i in range(len(curated_scores))]
    ax.scatter(curated_scores, diversity_scores, c=colors, alpha=0.5, s=10)
    ax.set_xlabel("Curated Quality Score", fontsize=12)
    ax.set_ylabel("Diversity Score (long-tail)", fontsize=12)
    ax.set_title("DS2: Quality vs Diversity (TERRa)", fontsize=13)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", label=f"Removed ({n_removed})"),
        Patch(facecolor="steelblue", label=f"Kept ({M})"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "diversity_vs_quality.png"), dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
removed_labels = train_removed["label"].value_counts()
removed_labels.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
ax.set_title("DS2: Removed Examples per Class (TERRa)", fontsize=13)
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "removed_per_class.png"), dpi=150)
plt.show()

removed_set = set(remove_indices_sorted.tolist())

overlap_cleanlab = {}
if os.path.exists("label_issues.csv"):
    cl_issues = pd.read_csv("label_issues.csv")
    if "original_index" in cl_issues.columns:
        cl_indices = set(cl_issues["original_index"].tolist())
    else:
        cl_indices = set(cl_issues.index.tolist())

    overlap = removed_set & cl_indices
    overlap_cleanlab = {
        "ds2_removed": len(removed_set),
        "cleanlab_removed": len(cl_indices),
        "overlap": len(overlap),
        "overlap_pct_of_ds2": round(len(overlap) / max(len(removed_set), 1) * 100, 2),
        "only_ds2": len(removed_set - cl_indices),
        "only_cleanlab": len(cl_indices - removed_set),
    }

overlap_datamaps = {}
dm_removed_path = "results_datamaps/removed_examples.csv"
if os.path.exists(dm_removed_path):
    dm_removed = pd.read_csv(dm_removed_path)
    if "original_index" in dm_removed.columns:
        dm_indices = set(dm_removed["original_index"].tolist())
    else:
        dm_indices = set(dm_removed.index.tolist())

    overlap = removed_set & dm_indices
    overlap_datamaps = {
        "ds2_removed": len(removed_set),
        "datamaps_removed": len(dm_indices),
        "overlap": len(overlap),
        "overlap_pct_of_ds2": round(len(overlap) / max(len(removed_set), 1) * 100, 2),
        "only_ds2": len(removed_set - dm_indices),
        "only_datamaps": len(dm_indices - removed_set),
    }
raw_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_raw_scores.pt"), weights_only=False)
raw_overall = [s[-1] for s in raw_scores]

ds2_stats = {
    "method": "DS2 (Data Selection via Scoring) — dual-sorting",
    "dataset": "TERRa (RussianNLP/russian_super_glue)",
    "model_name": "DeepPavlov/rubert-base-cased",
    "llm_scorer": "Qwen/Qwen3-Coder-Next",
    "embedding_model": "BAAI/bge-m3",
    "total_train": len(train_df),
    "keep_ratio": DS2_KEEP_RATIO,
    "n_kept": M,
    "n_removed": n_removed,
    "pct_removed": round(n_removed / len(train_df) * 100, 2),
    "clean_train_size": len(train_clean),
    "score_distribution_raw": dict(Counter(raw_overall)),
    "score_distribution_compressed": {int(k): int(v) for k, v in zip(*np.unique(compressed_scores, return_counts=True))},
    "score_distribution_curated": {int(k): int(v) for k, v in zip(*np.unique(curated_scores, return_counts=True))},
    "score_distribution_kept": {int(k): int(v) for k, v in zip(*np.unique(kept_scores, return_counts=True))},
    "score_distribution_removed": {int(k): int(v) for k, v in zip(*np.unique(removed_scores, return_counts=True))},
    "n_corrupted_scores": len(report.detection["score_error"]),
    "n_curated_scores": sum(1 for s in report.curation["score_curation"] if s[2] >= DS2_CONFIDENCE_PROB),
    "removed_per_class": dict(train_removed["label"].value_counts()),
}

if overlap_cleanlab:
    ds2_stats["overlap_with_cleanlab"] = overlap_cleanlab
if overlap_datamaps:
    ds2_stats["overlap_with_datamaps"] = overlap_datamaps

stats_path = os.path.join(RESULTS_DIR, "ds2_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(ds2_stats, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'=' * 60}")
print(f"ИТОГО: отобрано {M} из {len(train_df)} (keep_ratio={DS2_KEEP_RATIO})")
print(f"Удалено: {n_removed} ({ds2_stats['pct_removed']}%)")
print(f"Чистый train: {len(train_clean)} примеров → train_ds2.csv")
print(f"{'=' * 60}")
