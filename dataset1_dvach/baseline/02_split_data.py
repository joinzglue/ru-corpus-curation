import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATASET_NAME, SEED

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

print("Загружаю датасет...")
dataset = load_dataset(DATASET_NAME)

dfs = []
for split_name in dataset:
    dfs.append(dataset[split_name].to_pandas())
df = pd.concat(dfs, ignore_index=True)

text_col = df.columns[0]
label_col = [c for c in df.columns if c != text_col][0]
print(f"Текстовая колонка: '{text_col}', колонка меток: '{label_col}'")
print(f"Исходный размер: {len(df)}")

before = len(df)
df = df.dropna(subset=[text_col, label_col])
print(f"Удалено пропусков: {before - len(df)}")

before = len(df)
df = df.drop_duplicates(subset=[text_col], keep="first")
print(f"Удалено дубликатов: {before - len(df)}")

print(f"Размер после чистки: {len(df)}")
print()


train_val, test = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df[label_col]
)
train, val = train_test_split(
    train_val, test_size=1/9, random_state=SEED, stratify=train_val[label_col]
)

print("Размеры сплитов:")
print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
print()

print("Распределение классов в train:")
print(train[label_col].value_counts(normalize=True).round(3))
print()
print("Распределение классов в val:")
print(val[label_col].value_counts(normalize=True).round(3))
print()
print("Распределение классов в test:")
print(test[label_col].value_counts(normalize=True).round(3))
print()

train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

print(f"Сплиты сохранены в {SPLITS_DIR}/")
