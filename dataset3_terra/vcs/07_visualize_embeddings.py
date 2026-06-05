# Загрузить на Colab: train.csv, config.py, conflicting_pairs.csv, removed_indices.json

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

def find_input(filename):
    p1 = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(filename):
        return filename

LABEL_NAMES = {0: "entailment", 1: "not_entailment"}
LABEL_COLORS = {0: "#3498db", 1: "#e74c3c"}

print("=" * 60)
print("ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ ")
print("=" * 60)

train_df = pd.read_csv("train.csv")
n_total = len(train_df)
print(f"Train: {n_total} примеров")

premises = train_df["premise"].tolist()
hypotheses = train_df["hypothesis"].tolist()

raw_labels = train_df["label"].tolist()
unique_raw = sorted(set(raw_labels))
print(f"Уникальные значения label в train.csv: {unique_raw}")

if all(isinstance(l, (int, np.integer)) for l in raw_labels):
    labels = np.array(raw_labels, dtype=int)
else:
    str_to_int = {"entailment": 0, "not_entailment": 1}
    if not set(unique_raw).issubset(set(str_to_int.keys())):
        str_to_int = {s: i for i, s in enumerate(unique_raw)}
    labels = np.array([str_to_int[l] for l in raw_labels], dtype=int)
print(f"Метки приведены к int: {dict(zip(*np.unique(labels, return_counts=True)))}")

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
        batch_p = premises[i:i + DS2_EMBEDDING_BATCH_SIZE]
        batch_h = hypotheses[i:i + DS2_EMBEDDING_BATCH_SIZE]
        encoded = emb_tokenizer(
            batch_p, batch_h,
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

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=SEED,
)
coords = reducer.fit_transform(embeddings)

removed_mask = np.array([i in removed_indices for i in range(n_total)])
kept_mask = ~removed_mask

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
for lbl, color in LABEL_COLORS.items():
    mask = labels == lbl
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=color, s=8, alpha=0.6, edgecolors="none",
        label=f"{LABEL_NAMES[lbl]} (n={int(mask.sum())})",
    )
ax.set_title("TERRa: UMAP-проекция эмбеддингов\n(окраска по классам)", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(
    coords[kept_mask, 0], coords[kept_mask, 1],
    c="#bdc3c7", s=6, alpha=0.5, edgecolors="none",
    label=f"Оставлены (n={int(kept_mask.sum())})",
)
ax.scatter(
    coords[removed_mask, 0], coords[removed_mask, 1],
    c="#e74c3c", s=12, alpha=0.9, edgecolors="black", linewidths=0.3,
    label=f"Удалены VerDedup (n={int(removed_mask.sum())})",
)
ax.set_title("TERRa: оставленные vs удалённые точки\n(Verified Dedup)", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="best", fontsize=10)
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

fig, ax = plt.subplots(figsize=(12, 10))

for lbl, color in LABEL_COLORS.items():
    mask = labels == lbl
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=color, s=6, alpha=0.25, edgecolors="none",
    )

def draw_pairs(pairs, color, label, lw=0.7, alpha=0.5):
    for (a, b, s) in pairs:
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            color=color, linewidth=lw, alpha=alpha, zorder=2,
        )

draw_pairs(same_target_pairs,   color="#27ae60", label="same-target",     lw=0.8, alpha=0.6)
draw_pairs(diff_removed_pairs,  color="#c0392b", label="diff confirmed",  lw=1.0, alpha=0.8)
draw_pairs(diff_kept_pairs,     color="#2980b9", label="diff kept",       lw=0.6, alpha=0.45)

legend_handles = [
    Line2D([0], [0], color="#27ae60", lw=2,
           label=f"Same-target пары (удалены вторые) — {len(same_target_pairs)}"),
    Line2D([0], [0], color="#c0392b", lw=2,
           label=f"Diff-target подтверждены (удалены) — {len(diff_removed_pairs)}"),
    Line2D([0], [0], color="#2980b9", lw=2,
           label=f"Diff-target оставлены (контрастные) — {len(diff_kept_pairs)}"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0],
           markersize=8, label=LABEL_NAMES[0]),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1],
           markersize=8, label=LABEL_NAMES[1]),
]
ax.legend(handles=legend_handles, loc="best", fontsize=10, framealpha=0.95)

ax.set_title(
    f"TERRa: найденные пары на UMAP-проекции\n"
    f"(cos ≥ {DEDUP_COSINE_THRESHOLD}, всего пар: "
    f"{len(same_target_pairs) + len(diff_removed_pairs) + len(diff_kept_pairs)})",
    fontsize=12,
)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_pairs.png"), dpi=150)
plt.show()

print(f"\n{'=' * 60}")
print("ИТОГО:")
print(f"  Точек на графике:       {n_total}")
print(f"  Same-target пар:        {len(same_target_pairs)}")
print(f"  Diff-target removed:    {len(diff_removed_pairs)}")
print(f"  Diff-target kept:       {len(diff_kept_pairs)}")
print(f"  Эмбеддинги:             {emb_cache}")
print(f"  Графики:                umap_overview.png, umap_pairs.png")
print(f"{'=' * 60}")
