import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import json
import zipfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH_TERRA as MAX_LENGTH

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(DATASET_DIR, "results")
EDA_DIR     = os.path.join(RESULTS_DIR, "EDA")
ZIP_PATH    = os.path.join(DATASET_DIR, "TERRa.zip")
os.makedirs(EDA_DIR, exist_ok=True)

TERRA_URL = (
    "https://huggingface.co/datasets/RussianNLP/russian_super_glue"
    "/resolve/main/data/TERRa.zip"
)

if not os.path.exists(ZIP_PATH):
    urllib.request.urlretrieve(TERRA_URL, ZIP_PATH)

def read_jsonl_from_zip(zip_path, inner_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(inner_path) as f:
            lines = f.read().decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]

train_records = read_jsonl_from_zip(ZIP_PATH, "TERRa/train.jsonl")
val_records   = read_jsonl_from_zip(ZIP_PATH, "TERRa/val.jsonl")

df_train = pd.DataFrame(train_records)
df_train["original_split"] = "train"

df_val = pd.DataFrame(val_records)
df_val["original_split"] = "validation"

df = pd.concat([df_train, df_val], ignore_index=True)

df = df[["premise", "hypothesis", "label", "original_split"]]

raw_csv_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

print("=" * 60)
print("ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 60)
print(f"Размер датасета: {len(df)} строк")
print(f"Колонки: {list(df.columns)}")
print(f"Типы данных:\n{df.dtypes}")
print()

print("=" * 60)
print("ПРОПУСКИ")
print("=" * 60)
print(df.isnull().sum())
print()

print("=" * 60)
print("ДУБЛИКАТЫ")
print("=" * 60)
dup_premise = df.duplicated(subset=["premise"]).sum()
dup_pair    = df.duplicated(subset=["premise", "hypothesis"]).sum()
print(f"Дубликатов по premise:                {dup_premise}")
print(f"Дубликатов по паре (premise+hypothesis): {dup_pair}")
print()

print("=" * 60)
print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("=" * 60)
class_counts = df["label"].value_counts()
print(class_counts)
print(f"\nСоотношение: {class_counts.to_dict()}")
print()

print("=" * 60)
print("ПРИМЕРЫ ТЕКСТОВ (первые 5)")
print("=" * 60)
for _, row in df.head(5).iterrows():
    print(f"[{row['label']}]")
    print(f"  Premise:    {row['premise'][:200]}")
    print(f"  Hypothesis: {row['hypothesis'][:200]}")
    print()

df["premise_length"]    = df["premise"].fillna("").astype(str).apply(len)
df["hypothesis_length"] = df["hypothesis"].fillna("").astype(str).apply(len)
df["combined_length"]   = df["premise_length"] + df["hypothesis_length"]

print("=" * 60)
print("СТАТИСТИКА ДЛИН ТЕКСТОВ (символы)")
print("=" * 60)
print("--- Premise ---")
print(df["premise_length"].describe())
print("\n--- Hypothesis ---")
print(df["hypothesis_length"].describe())
print("\n--- Premise + Hypothesis (суммарно) ---")
print(df["combined_length"].describe())
print()

print("=" * 60)
print(f"СТАТИСТИКА ДЛИН В ТОКЕНАХ (модель: {MODEL_NAME})")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

token_lengths = df.apply(
    lambda row: len(tokenizer.encode(
        str(row["premise"]), str(row["hypothesis"]),
        add_special_tokens=True
    )),
    axis=1,
)
df["token_length"] = token_lengths

print(df["token_length"].describe())
pct_truncated = (df["token_length"] > MAX_LENGTH).mean() * 100
print(f"\nДоля пар длиннее MAX_LENGTH={MAX_LENGTH}: {pct_truncated:.1f}%")
print()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["premise_length"], bins=40, edgecolor="black", color="steelblue")
axes[0].set_title("Длины premise (символы)")
axes[0].set_xlabel("Длина (символы)")
axes[0].set_ylabel("Количество")
axes[1].hist(df["hypothesis_length"], bins=40, edgecolor="black", color="coral")
axes[1].set_title("Длины hypothesis (символы)")
axes[1].set_xlabel("Длина (символы)")
axes[1].set_ylabel("Количество")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "premise_hypothesis_lengths.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(df["combined_length"], bins=50, edgecolor="black")
plt.title("Суммарная длина premise + hypothesis (символы)")
plt.xlabel("Длина (символы)")
plt.ylabel("Количество")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "text_length_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(df["token_length"], bins=50, edgecolor="black")
plt.axvline(MAX_LENGTH, color="red", linestyle="--", label=f"MAX_LENGTH={MAX_LENGTH}")
plt.title("Длины в токенах (парная токенизация premise + hypothesis)")
plt.xlabel("Количество токенов")
plt.ylabel("Количество пар")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "token_length_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 7))
plt.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Доля классов в датасете TERRa")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "class_pie_chart.png"), dpi=150)
plt.close()
