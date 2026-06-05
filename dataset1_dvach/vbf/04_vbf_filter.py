# train.csv, config.py, label_issues.csv (cleanlab), hard_to_learn.csv (datamaps), removed_examples.csv (ds2)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config import SEED

RESULTS_DIR = "./results_hybrid"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)

train_df = pd.read_csv("train.csv")
n_total = len(train_df)
print(f"Train: {n_total} примеров")

cleanlab_df = pd.read_csv("label_issues.csv")
datamaps_df = pd.read_csv("hard_to_learn.csv")
ds2_df = pd.read_csv("removed_examples.csv")

print(f"Cleanlab label_issues: {len(cleanlab_df)}")
print(f"DataMaps hard_to_learn: {len(datamaps_df)}")
print(f"DS2 removed_examples: {len(ds2_df)}")

def get_removed_indices(train, removed, key_col="text"):
    removed_set = set(removed[key_col].str.strip())
    indices = set()
    for idx, val in train[key_col].items():
        if str(val).strip() in removed_set:
            indices.add(idx)
    return indices

removed_cl = get_removed_indices(train_df, cleanlab_df)
removed_dm = get_removed_indices(train_df, datamaps_df)
removed_ds2 = get_removed_indices(train_df, ds2_df)

print(f"\nMatched:")
print(f"  Cleanlab: {len(removed_cl)}")
print(f"  DataMaps: {len(removed_dm)}")
print(f"  DS2:      {len(removed_ds2)}")

cl_dm = removed_cl & removed_dm
cl_ds2 = removed_cl & removed_ds2
dm_ds2 = removed_dm & removed_ds2
all_three = removed_cl & removed_dm & removed_ds2

print(f"  Cleanlab ∩ DataMaps: {len(cl_dm)}")
print(f"  Cleanlab ∩ DS2:      {len(cl_ds2)}")
print(f"  DataMaps ∩ DS2:      {len(dm_ds2)}")
print(f"  Все три метода:      {len(all_three)}")

only_cl = removed_cl - removed_dm - removed_ds2
only_dm = removed_dm - removed_cl - removed_ds2
only_ds2 = removed_ds2 - removed_cl - removed_dm

print(f"  Только Cleanlab: {len(only_cl)}")
print(f"  Только DataMaps: {len(only_dm)}")
print(f"  Только DS2:      {len(only_ds2)}")

votes = defaultdict(int)
for idx in removed_cl:
    votes[idx] += 1
for idx in removed_dm:
    votes[idx] += 1
for idx in removed_ds2:
    votes[idx] += 1

intersection_removed = {idx for idx, count in votes.items() if count >= 2}
intersection_kept = set(range(n_total)) - intersection_removed

train_intersection = train_df.loc[sorted(intersection_kept)].reset_index(drop=True)
train_intersection.to_csv("train_intersection.csv", index=False, encoding="utf-8-sig")

n_intersection_removed = len(intersection_removed)
print(f"\nПересечение:")
print(f"  Удалено: {n_intersection_removed} ({n_intersection_removed / n_total * 100:.2f}%)")
print(f"  Осталось: {len(train_intersection)} train_intersection.csv")

union_removed = removed_cl | removed_dm | removed_ds2
union_kept = set(range(n_total)) - union_removed

train_union = train_df.loc[sorted(union_kept)].reset_index(drop=True)
train_union.to_csv("train_union.csv", index=False, encoding="utf-8-sig")

n_union_removed = len(union_removed)
print(f"\nОбъединение:")
print(f"  Удалено: {n_union_removed} ({n_union_removed / n_total * 100:.2f}%)")
print(f"  Осталось: {len(train_union)} train_union.csv")

def build_removed_df(train, indices, removed_cl, removed_dm, removed_ds2):
    rows = []
    for idx in sorted(indices):
        methods = []
        if idx in removed_cl:
            methods.append("cleanlab")
        if idx in removed_dm:
            methods.append("datamaps")
        if idx in removed_ds2:
            methods.append("ds2")
        row = train.iloc[idx].to_dict()
        row["methods"] = ", ".join(methods)
        row["n_methods"] = len(methods)
        rows.append(row)
    return pd.DataFrame(rows)


intersection_removed_df = build_removed_df(
    train_df, intersection_removed, removed_cl, removed_dm, removed_ds2
)
intersection_removed_df.to_csv(
    os.path.join(RESULTS_DIR, "intersection_removed.csv"),
    index=False, encoding="utf-8-sig",
)
print(f"\nУдалённые (intersection): {len(intersection_removed_df)}")

union_removed_df = build_removed_df(
    train_df, union_removed, removed_cl, removed_dm, removed_ds2
)
union_removed_df.to_csv(
    os.path.join(RESULTS_DIR, "union_removed.csv"),
    index=False, encoding="utf-8-sig",
)
print(f"Удалённые (union): {len(union_removed_df)}")

unique_labels = sorted(train_df["label"].unique())

methods_names = ["Cleanlab", "DataMaps", "DS2"]
sets = [removed_cl, removed_dm, removed_ds2]
overlap_matrix = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        overlap_matrix[i, j] = len(sets[i] & sets[j])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    overlap_matrix, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=methods_names, yticklabels=methods_names, ax=ax,
    linewidths=1, linecolor="white",
)
ax.set_title("Overlap Matrix: Pairwise Intersections (Dvach)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overlap_heatmap.png"), dpi=150)
plt.show()


from matplotlib_venn import venn3
fig, ax = plt.subplots(figsize=(8, 7))
venn3(
    [removed_cl, removed_dm, removed_ds2],
    set_labels=("Cleanlab", "DataMaps", "DS2"),
    ax=ax,)
ax.set_title("Venn Diagram: Removed Examples (Dvach)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "venn_diagram.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
class_removed = [
    int((intersection_removed_df["label"] == lbl).sum()) for lbl in unique_labels
]
class_total = [int((train_df["label"] == lbl).sum()) for lbl in unique_labels]
x = np.arange(len(unique_labels))
width = 0.35
bars1 = ax.bar(x - width / 2, class_total, width, label="Total", alpha=0.6, edgecolor="black")
bars2 = ax.bar(x + width / 2, class_removed, width, label="Removed (intersection)", alpha=0.8, edgecolor="black", color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(unique_labels, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Intersection: Removed per Class (Dvach)", fontsize=13)
ax.legend()
for bar, cnt in zip(bars2, class_removed):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(cnt), ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "removal_by_class_intersection.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
class_removed_u = [
    int((union_removed_df["label"] == lbl).sum()) for lbl in unique_labels
]
bars1 = ax.bar(x - width / 2, class_total, width, label="Total", alpha=0.6, edgecolor="black")
bars2 = ax.bar(x + width / 2, class_removed_u, width, label="Removed (union)", alpha=0.8, edgecolor="black", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(unique_labels, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Union: Removed per Class (Dvach)", fontsize=13)
ax.legend()
for bar, cnt in zip(bars2, class_removed_u):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(cnt), ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "removal_by_class_union.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
vote_counts = {1: 0, 2: 0, 3: 0}
for idx, count in votes.items():
    vote_counts[count] += 1
bars = ax.bar(
    [str(k) for k in sorted(vote_counts.keys())],
    [vote_counts[k] for k in sorted(vote_counts.keys())],
    color=["#3498db", "#e67e22", "#e74c3c"],
    edgecolor="black", alpha=0.8,
)
for bar, cnt in zip(bars, [vote_counts[k] for k in sorted(vote_counts.keys())]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(cnt), ha="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Number of methods flagging the example")
ax.set_ylabel("Count")
ax.set_title("Vote Distribution: How Many Methods Flag Each Example (Dvach)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "vote_distribution.png"), dpi=150)
plt.show()

hybrid_stats = {
    "method": "Hybrid (Intersection + Union)",
    "dataset": "Dvach (Kostya165/ru_emotion_dvach)",
    "total_train": n_total,
    "per_method_removed": {
        "cleanlab": len(removed_cl),
        "datamaps": len(removed_dm),
        "ds2": len(removed_ds2),
    },
    "overlap": {
        "cleanlab_datamaps": len(cl_dm),
        "cleanlab_ds2": len(cl_ds2),
        "datamaps_ds2": len(dm_ds2),
        "all_three": len(all_three),
    },
    "unique_per_method": {
        "only_cleanlab": len(only_cl),
        "only_datamaps": len(only_dm),
        "only_ds2": len(only_ds2),
    },
    "vote_distribution": {
        "1_method": vote_counts.get(1, 0),
        "2_methods": vote_counts.get(2, 0),
        "3_methods": vote_counts.get(3, 0),
    },
    "intersection": {
        "n_removed": n_intersection_removed,
        "pct_removed": round(n_intersection_removed / n_total * 100, 2),
        "clean_train_size": len(train_intersection),
        "removed_per_class": {
            lbl: int((intersection_removed_df["label"] == lbl).sum())
            for lbl in unique_labels
        },
    },
    "union": {
        "n_removed": n_union_removed,
        "pct_removed": round(n_union_removed / n_total * 100, 2),
        "clean_train_size": len(train_union),
        "removed_per_class": {
            lbl: int((union_removed_df["label"] == lbl).sum())
            for lbl in unique_labels
        },
    },
}

stats_path = os.path.join(RESULTS_DIR, "hybrid_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(hybrid_stats, f, indent=2, ensure_ascii=False)
print(f"\nСтатистика сохранена: {stats_path}")

assert len(train_intersection) + n_intersection_removed == n_total, \
    f"Intersection: {len(train_intersection)} + {n_intersection_removed} != {n_total}"
print(f"  ✓ Intersection: {len(train_intersection)} + {n_intersection_removed} = {n_total}")

assert len(train_union) + n_union_removed == n_total, \
    f"Union: {len(train_union)} + {n_union_removed} != {n_total}"
print(f"  ✓ Union: {len(train_union)} + {n_union_removed} = {n_total}")

assert union_removed == removed_cl | removed_dm | removed_ds2
print(f"  ✓ Union = Cleanlab ∪ DataMaps ∪ DS2")

assert intersection_removed <= union_removed
print(f"  ✓ Intersection ⊆ Union")

print(f"\n{'=' * 60}")
print("ИТОГО:")
print(f"  Исходный train:  {n_total}")
print(f"  Intersection:    удалено {n_intersection_removed} ({n_intersection_removed / n_total * 100:.1f}%), "
      f"осталось {len(train_intersection)}")
print(f"  Union:           удалено {n_union_removed} ({n_union_removed / n_total * 100:.1f}%), "
      f"осталось {len(train_union)}")
print(f"{'=' * 60}")
