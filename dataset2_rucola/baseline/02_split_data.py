import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(DATASET_DIR, "results")
SPLITS_DIR  = os.path.join(DATASET_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

csv_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
df = pd.read_csv(csv_path)

before = len(df)
df = df.dropna(subset=["text", "label"])
print(f"Удалено пропусков (NaN): {before - len(df)}")

before = len(df)
df = df[df["text"].fillna("").astype(str).str.strip() != ""]
print(f"Удалено пустых строк: {before - len(df)}")

before = len(df)
df = df.drop_duplicates(subset=["text"], keep="first")
print(f"Удалено дубликатов: {before - len(df)}")

df["label"] = df["label"].astype(int)

df = df[["text", "label", "error_type"]]

train_val, test = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"]
)
train, val = train_test_split(
    train_val, test_size=1/9, random_state=SEED, stratify=train_val["label"]
)

label_names = {0: "unacceptable", 1: "acceptable"}
for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"Распределение классов в {name}:")
    counts = split["label"].value_counts().sort_index()
    for lbl, cnt in counts.items():
        print(f"  {lbl} ({label_names.get(lbl, '?')}): {cnt} ({cnt / len(split) * 100:.1f}%)")
    print()

train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False, encoding="utf-8-sig")
val.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False, encoding="utf-8-sig")
test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False, encoding="utf-8-sig")

