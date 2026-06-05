# train.csv, config.py, conflicting_pairs.csv, removed_indices.json


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm

import faiss
import umap

from transformers import AutoTokenizer, AutoModel

from config import (
    SEED,
    DS2_EMBEDDING_MODEL, DS2_EMBEDDING_BATCH_SIZE,
    DEDUP_COSINE_THRESHOLD, DEDUP_KNN_K,
)

RESULTS_DIR = "./results_ver_dedup"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.5
FORCE_RECOMPUTE_UMAP = False


def find_input(filename):
    p1 = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(filename):
        return filename

LABEL_COLORS_DEFAULT = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

train_df = pd.read_csv("train.csv")
n_total = len(train_df)
print(f"Train: {n_total} примеров")

text_col = train_df.columns[0]
label_col = "label" if "label" in train_df.columns else \
    [c for c in train_df.columns if c != text_col][0]
print(f"Text column: {text_col} | Label column: {label_col}")

texts = train_df[text_col].astype(str).tolist()

raw_labels = train_df[label_col].tolist()
unique_raw = sorted(set(raw_labels))
print(f"Уникальные значения label в train.csv: {unique_raw}")

if all(isinstance(l, (int, np.integer)) for l in raw_labels):
    labels = np.array(raw_labels, dtype=int)
    label_names = {int(l): str(l) for l in unique_raw}
else:
    str_to_int = {s: i for i, s in enumerate(unique_raw)}
    labels = np.array([str_to_int[l] for l in raw_labels], dtype=int)
    label_names = {i: s for s, i in str_to_int.items()}

n_classes = len(label_names)
LABEL_COLORS = {
    int(lbl): LABEL_COLORS_DEFAULT[i % len(LABEL_COLORS_DEFAULT)]
    for i, lbl in enumerate(sorted(label_names.keys()))
}

with open(find_input("removed_indices.json"), "r", encoding="utf-8") as f:
    removed_indices = set(json.load(f))
print(f"Удалено методом: {len(removed_indices)}")

conflicting_df = pd.read_csv(find_input("conflicting_pairs.csv"))
print(f"Diff-target пар: {len(conflicting_df)}")

emb_cache = os.path.join(RESULTS_DIR, "embeddings.npy")
if os.path.exists(emb_cache):
    embeddings = np.load(emb_cache)
else:
    emb_tokenizer = AutoTokenizer.from_pretrained(DS2_EMBEDDING_MODEL)
    emb_model = AutoModel.from_pretrained(DS2_EMBEDDING_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model = emb_model.to(device).eval()
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
    np.save(emb_cache, embeddings)

    del emb_model, emb_tokenizer
    torch.cuda.empty_cache()

umap_cache = os.path.join(RESULTS_DIR, "umap_coords.npy")
if os.path.exists(umap_cache) and not FORCE_RECOMPUTE_UMAP:
    coords = np.load(umap_cache)
    print(f"UMAP shape: {coords.shape}")
else:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=SEED,
        low_memory=True,
    )
    coords = reducer.fit_transform(embeddings)
    np.save(umap_cache, coords)

removed_mask = np.array([i in removed_indices for i in range(n_total)])
kept_mask = ~removed_mask

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

ax = axes[0]
for lbl in sorted(label_names.keys()):
    mask = labels == lbl
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=LABEL_COLORS[lbl], s=2, alpha=0.45, edgecolors="none",
        label=f"{label_names[lbl]} (n={int(mask.sum())})",
    )
ax.set_title("Dvach: UMAP-проекция эмбеддингов\n(окраска по классам)", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="best", fontsize=9, markerscale=4)
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(
    coords[kept_mask, 0], coords[kept_mask, 1],
    c="#bdc3c7", s=2, alpha=0.3, edgecolors="none",
    label=f"Оставлены (n={int(kept_mask.sum())})",
)
ax.scatter(
    coords[removed_mask, 0], coords[removed_mask, 1],
    c="#e74c3c", s=6, alpha=0.85, edgecolors="black", linewidths=0.15,
    label=f"Удалены VerDedup (n={int(removed_mask.sum())})",
)
ax.set_title("Dvach: оставленные vs удалённые точки\n(Verified Dedup)", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="best", fontsize=10, markerscale=3)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_overview.png"), dpi=150)
plt.show()

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
k = min(DEDUP_KNN_K, n_total)
sims, neighbors = index.search(embeddings, k)

same_target_pairs = []
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
        if labels[a] == labels[b]:
            same_target_pairs.append((a, b, s))

diff_kept_pairs = []
diff_removed_pairs = []
for _, row in conflicting_df.iterrows():
    a, b = int(row["idx_a"]), int(row["idx_b"])
    s = float(row["cos_sim"])
    if row["action"] == "kept":
        diff_kept_pairs.append((a, b, s))
    else:
        diff_removed_pairs.append((a, b, s))

print(f"  Same-target пар:        {len(same_target_pairs)}")
print(f"  Diff-target removed:    {len(diff_removed_pairs)}")
print(f"  Diff-target kept:       {len(diff_kept_pairs)}")

MAX_LINES_PER_TYPE = 3000


def maybe_subsample(pairs, max_n):
    if len(pairs) <= max_n:
        return pairs, False
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(pairs), size=max_n, replace=False)
    return [pairs[i] for i in idx], True


same_show, same_subs = maybe_subsample(same_target_pairs, MAX_LINES_PER_TYPE)
diff_rem_show, diff_rem_subs = maybe_subsample(diff_removed_pairs, MAX_LINES_PER_TYPE)
diff_kept_show, diff_kept_subs = maybe_subsample(diff_kept_pairs, MAX_LINES_PER_TYPE)

fig, ax = plt.subplots(figsize=(16, 13))

for lbl in sorted(label_names.keys()):
    mask = labels == lbl
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=LABEL_COLORS[lbl], s=2, alpha=0.15, edgecolors="none",
    )

def draw_pairs(pairs, color, lw=0.5, alpha=0.4):
    for (a, b, s) in pairs:
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            color=color, linewidth=lw, alpha=alpha, zorder=2,
        )

draw_pairs(same_show,     color="#27ae60", lw=0.4, alpha=0.35)
draw_pairs(diff_rem_show, color="#c0392b", lw=0.7, alpha=0.7)
draw_pairs(diff_kept_show, color="#2980b9", lw=0.4, alpha=0.35)

legend_handles = [
    Line2D([0], [0], color="#27ae60", lw=2,
           label=f"Same-target пары (удалены вторые) — {len(same_target_pairs)}"),
    Line2D([0], [0], color="#c0392b", lw=2,
           label=f"Diff-target подтверждены (удалены) — {len(diff_removed_pairs)}"),
    Line2D([0], [0], color="#2980b9", lw=2,
           label=f"Diff-target оставлены (контрастные) — {len(diff_kept_pairs)}"),
]
for lbl in sorted(label_names.keys()):
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=LABEL_COLORS[lbl], markersize=8,
               label=label_names[lbl])
    )
ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.95)

total_pairs = len(same_target_pairs) + len(diff_removed_pairs) + len(diff_kept_pairs)
ax.set_title(
    f"Dvach: найденные пары на UMAP-проекции\n"
    f"(cos ≥ {DEDUP_COSINE_THRESHOLD}, всего пар: {total_pairs})",
    fontsize=12,
)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_pairs.png"), dpi=150)
plt.show()
