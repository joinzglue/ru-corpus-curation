# train.csv, config.py, docta_lite/, ds2_compressed_scores.pt

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from tqdm.notebook import tqdm

from config import (
    SEED,
    DS2_EMBEDDING_MODEL, DS2_NUM_CLASSES,
    DS2_KNN_K_DETECT, DS2_KNN_K_DIVERSITY,
    DS2_CONFIDENCE_PROB, DS2_HOC_MAX_STEP, DS2_HOC_LR,
    DS2_HOC_NUM_ROUNDS, DS2_DETECT_NUM_EPOCH,
    DS2_EMBEDDING_BATCH_SIZE,
)

RESULTS_DIR = "./results_ds2"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


train_df = pd.read_csv("train.csv")
compressed_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_compressed_scores.pt"), weights_only=False)
compressed_scores = np.array(compressed_scores)

print(f"Train: {len(train_df)} примеров")
print(f"Compressed scores: {len(compressed_scores)} (distribution: {dict(zip(*np.unique(compressed_scores, return_counts=True)))})")

raw_scores = torch.load(os.path.join(RESULTS_DIR, "ds2_raw_scores.pt"), weights_only=False)
api_failures = [i for i, s in enumerate(raw_scores) if all(v == 0 for v in s)]
if api_failures:
    median_score = int(np.median(compressed_scores[compressed_scores > 0])) if np.any(compressed_scores > 0) else 1
    print(f"WARNING: {len(api_failures)} примеров с нулевыми raw scores (ошибки API)")
    print(f"  Заменяем compressed score 0 → {median_score} (медиана), чтобы не искажать T-матрицу")
    for idx in api_failures:
        compressed_scores[idx] = median_score

from transformers import AutoTokenizer, AutoModel

emb_tokenizer = AutoTokenizer.from_pretrained(DS2_EMBEDDING_MODEL)
emb_model = AutoModel.from_pretrained(DS2_EMBEDDING_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_model = emb_model.to(device)
emb_model.eval()

premises = train_df['premise'].tolist()
hypotheses = train_df['hypothesis'].tolist()

all_embeddings = []
for i in tqdm(range(0, len(premises), DS2_EMBEDDING_BATCH_SIZE), desc="Embedding"):
    batch_premises = premises[i:i + DS2_EMBEDDING_BATCH_SIZE]
    batch_hypotheses = hypotheses[i:i + DS2_EMBEDDING_BATCH_SIZE]
    encoded = emb_tokenizer(
        batch_premises, batch_hypotheses,
        padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = emb_model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    embeddings = (output[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(embeddings.cpu())

embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
del emb_model, emb_tokenizer
torch.cuda.empty_cache()
from docta_lite.datasets.customize import CustomizedDataset
from docta_lite.core.report import Report

dataset = CustomizedDataset(
    feature=embeddings_np,
    label=compressed_scores,
    index=np.arange(len(compressed_scores)),
)

effective_sample_size = min(50000, int(len(dataset) * 0.9))

cfg = SimpleNamespace(
    seed=SEED,
    modality="text",
    num_classes=DS2_NUM_CLASSES,
    feature_type="embedding",
    details=False,
    embedding_model=DS2_EMBEDDING_MODEL,
    embedding_cfg=SimpleNamespace(
        shuffle=False,
        batch_size=DS2_EMBEDDING_BATCH_SIZE,
        save_num=800,
        num_workers=0,
        use_pca=False,
        use_mi=False,
        n_neighbors=DS2_KNN_K_DETECT,
        n_neighbors_diversity=DS2_KNN_K_DIVERSITY,
    ),
    hoc_cfg=SimpleNamespace(
        max_step=DS2_HOC_MAX_STEP,
        T0=None,
        p0=None,
        lr=DS2_HOC_LR,
        num_rounds=DS2_HOC_NUM_ROUNDS,
        sample_size=effective_sample_size,
        already_2nn=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ),
    detect_cfg=SimpleNamespace(
        num_epoch=DS2_DETECT_NUM_EPOCH,
        sample_size=effective_sample_size,
        k=DS2_KNN_K_DETECT,
        name="simifeat",
        method="rank",
    ),
    save_path=RESULTS_DIR + "/",
    dataset_type="ds2_terra",
    device=device,
)
from docta_lite.apis import DetectLabel, DetectFeature

report = Report()

detector = DetectLabel(cfg, dataset, report=report)
detector.detect()

print(f"\nT matrix (noise transition):\n{np.round(report.diagnose['T'], 3)}")
print(f"Score errors found: {len(report.detection['score_error'])}")
print(f"Score curations: {len(report.curation['score_curation'])}")

detector_feature = DetectFeature(cfg, dataset, report=report)
detector_feature.rare_score()

print(f"Diversity scores computed: {len(report.detection['rare_example'])}")

report_path = os.path.join(RESULTS_DIR, "ds2_report.pt")
torch.save(report, report_path)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    report.diagnose["T"],
    annot=True, fmt=".2f", cmap="YlGnBu",
    xticklabels=range(DS2_NUM_CLASSES),
    yticklabels=range(DS2_NUM_CLASSES),
    ax=ax,
)
ax.set_title("Score Transition Matrix T (TERRa)", fontsize=14)
ax.set_xlabel("Observed Score", fontsize=12)
ax.set_ylabel("True Score", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_transition_heatmap.png"), dpi=150)
plt.show()
