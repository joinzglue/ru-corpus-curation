import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(DATASET_DIR, "results")
SPLITS_DIR  = os.path.join(DATASET_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

csv_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
df = pd.read_csv(csv_path)

before = len(df)
df = df.dropna(subset=["premise", "hypothesis", "label"])
print(f"Удалено пропусков: {before - len(df)}")

before = len(df)
df = df.drop_duplicates(subset=["premise", "hypothesis"], keep="first")
print(f"Удалено дубликатов: {before - len(df)}")

print(f"Размер после чистки: {len(df)}")
print()

df = df[["premise", "hypothesis", "label"]]

train_val, test = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"]
)
train, val = train_test_split(
    train_val, test_size=1/9, random_state=SEED, stratify=train_val["label"]
)

print("Размеры сплитов:")
print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
print()

for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"Распределение классов в {name}:")
    print(split["label"].value_counts(normalize=True).round(3))
    print()

train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False, encoding="utf-8-sig")
val.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False, encoding="utf-8-sig")
test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False, encoding="utf-8-sig")
