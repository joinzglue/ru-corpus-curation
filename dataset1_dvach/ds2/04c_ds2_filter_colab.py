# train.csv, config.py, ds2_compressed_scores.pt, ds2_report.pt

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import Counter

from config import SEED, DS2_NUM_CLASSES, DS2_CONFIDENCE_PROB, DS2_KEEP_RATIO

RESULTS_DIR = "./results_ds2"
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)

train_df = pd.read_csv("train.csv")
text_col = train_df.columns[0]
label_col = [c for c in train_df.columns if c != text_col][0]

compressed_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_compressed_scores.pt"), weights_only=False)
report = torch.load(os.path.join(RESULTS_DIR, "ds2_report.pt"), weights_only=False)

def score_curating(report, compressed_scores, confidence_prob):
    scores = list(compressed_scores)
    curated_count = 0
    for sample in report.curation['score_curation']:
        idx, suggested_score, confidence = int(sample[0]), int(sample[1]), sample[2]
        if confidence >= confidence_prob:
            scores[idx] = suggested_score
            curated_count += 1
    print(f"Curated {curated_count} scores")
    return scores

curated_scores = np.array(score_curating(report, compressed_scores, DS2_CONFIDENCE_PROB))
torch.save(curated_scores.tolist(), os.path.join(RESULTS_DIR, "ds2_curated_scores.pt"))

print(f"Compressed distribution: {dict(zip(*np.unique(compressed_scores, return_counts=True)))}")
print(f"Curated distribution:    {dict(zip(*np.unique(curated_scores, return_counts=True)))}")

rare_examples = report.detection.get("rare_example", [])
if rare_examples:
    diversity_scores = np.array([item[1] for item in rare_examples])
else:
    diversity_scores = np.zeros(len(train_df))

M = int(DS2_KEEP_RATIO * len(train_df))

sort_keys = np.lexsort((diversity_scores, curated_scores))[::-1]

keep_indices = sort_keys[:M]
remove_indices = sort_keys[M:]
keep_indices_sorted = np.sort(keep_indices)
remove_indices_sorted = np.sort(remove_indices)
n_removed = len(remove_indices)

print(f"Отобрано: {M}, удалено: {n_removed} ({n_removed / len(train_df) * 100:.2f}%)")

kept_scores = curated_scores[keep_indices_sorted]
removed_scores = curated_scores[remove_indices_sorted]

train_clean = train_df.iloc[keep_indices_sorted].reset_index(drop=True)
train_removed = train_df.iloc[remove_indices_sorted].copy()
train_removed["curated_score"] = curated_scores[remove_indices_sorted]
train_removed["diversity_score"] = diversity_scores[remove_indices_sorted]

train_clean.to_csv("train_ds2.csv", index=False)
train_removed.to_csv(os.path.join(RESULTS_DIR, "removed_examples.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, scores, title in zip(axes, [compressed_scores, curated_scores], ["Compressed", "Curated"]):
    counter = Counter(scores)
    ls = sorted(counter.keys())
    ax.bar([str(k) for k in ls], [counter[k] for k in ls], edgecolor="black", alpha=0.7)
    ax.set_title(title)
fig.suptitle("DS2: Score Distribution (Dvach)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "compressed_vs_curated.png"), dpi=150)
plt.show()

if rare_examples:
    fig, ax = plt.subplots(figsize=(8, 6))
    removed_set_vis = set(remove_indices_sorted.tolist())
    colors = ["red" if i in removed_set_vis else "steelblue" for i in range(len(curated_scores))]
    ax.scatter(curated_scores, diversity_scores, c=colors, alpha=0.3, s=5)
    ax.set_xlabel("Curated Quality Score")
    ax.set_ylabel("Diversity Score")
    ax.set_title("DS2: Quality vs Diversity (Dvach)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="red", label=f"Removed ({n_removed})"),
                       Patch(facecolor="steelblue", label=f"Kept ({M})")])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "diversity_vs_quality.png"), dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
train_removed[label_col].value_counts().plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
ax.set_title("DS2: Removed per Class (Dvach)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "removed_per_class.png"), dpi=150)
plt.show()

removed_set = set(remove_indices_sorted.tolist())
overlap_cleanlab = {}
overlap_datamaps = {}

if os.path.exists("label_issues.csv"):
    cl = pd.read_csv("label_issues.csv")
    cl_indices = set(cl.get("original_index", cl.index).tolist())
    ov = removed_set & cl_indices
    overlap_cleanlab = {"ds2_removed": len(removed_set), "cleanlab_removed": len(cl_indices),
        "overlap": len(ov), "only_ds2": len(removed_set - cl_indices), "only_cleanlab": len(cl_indices - removed_set)}

dm_path = "results_datamaps/removed_examples.csv"
if os.path.exists(dm_path):
    dm = pd.read_csv(dm_path)
    dm_indices = set(dm.get("original_index", dm.index).tolist())
    ov = removed_set & dm_indices
    overlap_datamaps = {"ds2_removed": len(removed_set), "datamaps_removed": len(dm_indices),
        "overlap": len(ov), "only_ds2": len(removed_set - dm_indices), "only_datamaps": len(dm_indices - removed_set)}

raw_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_raw_scores.pt"), weights_only=False)
ds2_stats = {
    "method": "DS2 (Data Selection via Scoring) — dual-sorting",
    "dataset": "Dvach (Kostya165/ru_emotion_dvach)",
    "model_name": "DeepPavlov/rubert-base-cased",
    "llm_scorer": "Qwen/Qwen3-Coder-Next",
    "embedding_model": "BAAI/bge-m3",
    "total_train": len(train_df),
    "keep_ratio": DS2_KEEP_RATIO,
    "n_kept": M,
    "n_removed": n_removed,
    "pct_removed": round(n_removed / len(train_df) * 100, 2),
    "clean_train_size": len(train_clean),
    "score_distribution_raw": dict(Counter([s[-1] for s in raw_scores])),
    "score_distribution_compressed": {int(k): int(v) for k, v in zip(*np.unique(compressed_scores, return_counts=True))},
    "score_distribution_curated": {int(k): int(v) for k, v in zip(*np.unique(curated_scores, return_counts=True))},
    "score_distribution_kept": {int(k): int(v) for k, v in zip(*np.unique(kept_scores, return_counts=True))},
    "score_distribution_removed": {int(k): int(v) for k, v in zip(*np.unique(removed_scores, return_counts=True))},
    "n_corrupted_scores": len(report.detection["score_error"]),
    "removed_per_class": dict(train_removed[label_col].value_counts()),
}
if overlap_cleanlab: ds2_stats["overlap_with_cleanlab"] = overlap_cleanlab
if overlap_datamaps: ds2_stats["overlap_with_datamaps"] = overlap_datamaps

with open(os.path.join(RESULTS_DIR, "ds2_stats.json"), "w", encoding="utf-8") as f:
    json.dump(ds2_stats, f, indent=2, ensure_ascii=False, default=str)

print(f"\nИТОГО: отобрано {M} из {len(train_df)} (keep_ratio={DS2_KEEP_RATIO})")
print(f"Удалено: {n_removed} ({ds2_stats['pct_removed']}%)")
