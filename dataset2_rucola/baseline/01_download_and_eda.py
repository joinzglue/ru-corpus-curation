import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(DATASET_DIR, "data_rucola")
RESULTS_DIR = os.path.join(DATASET_DIR, "results")
EDA_DIR     = os.path.join(RESULTS_DIR, "EDA")
os.makedirs(EDA_DIR, exist_ok=True)

files = {
    "in_domain_train":   os.path.join(DATA_DIR, "in_domain_train.csv"),
    "in_domain_dev":     os.path.join(DATA_DIR, "in_domain_dev.csv"),
    "out_of_domain_dev": os.path.join(DATA_DIR, "out_of_domain_dev.csv"),
}

dfs = []
for split_name, path in files.items():
    df_split = pd.read_csv(path)
    df_split["original_split"] = split_name
    dfs.append(df_split)
    print(f"  {len(df_split)} строк")

df = pd.concat(dfs, ignore_index=True)

df = df.rename(columns={"sentence": "text", "acceptable": "label"})

print(f"\nОбщий размер (с метками): {len(df)} строк")
print()

raw_csv_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
print(f"Полный датасет сохранён: {raw_csv_path}")
print()

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
duplicates = df.duplicated(subset=["text"]).sum()
print(f"Дубликатов по колонке 'text': {duplicates}")
print()

print("=" * 60)
print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("=" * 60)

label_names = {0: "unacceptable", 1: "acceptable"}
class_counts = df["label"].value_counts().sort_index()
print("По числовым меткам:")
print(class_counts)
print()
print("По текстовым меткам:")
for label_id, count in class_counts.items():
    pct = count / len(df) * 100
    print(f"  {label_id} ({label_names.get(label_id, '?')}): {count} ({pct:.1f}%)")
print()

print("=" * 60)
print("РАСПРЕДЕЛЕНИЕ ПО ДОМЕНАМ")
print("=" * 60)
for split_name in df["original_split"].unique():
    subset = df[df["original_split"] == split_name]
    lbl_counts = subset["label"].value_counts().sort_index()
    print(f"\n{split_name} ({len(subset)} строк):")
    for lbl, cnt in lbl_counts.items():
        print(f"  {lbl} ({label_names.get(lbl, '?')}): {cnt} ({cnt / len(subset) * 100:.1f}%)")
print()

print("=" * 60)
print("ТИПЫ ОШИБОК (unacceptable предложения)")
print("=" * 60)
unacceptable = df[df["label"] == 0]
error_counts = unacceptable["error_type"].value_counts()
print(f"Всего unacceptable: {len(unacceptable)}")
for err_type, cnt in error_counts.items():
    if str(err_type) != "0":
        print(f"  {err_type}: {cnt} ({cnt / len(unacceptable) * 100:.1f}%)")
print()

print("=" * 60)
print("ПРИМЕРЫ ТЕКСТОВ (первые 5)")
print("=" * 60)
for _, row in df.head(5).iterrows():
    lbl = row["label"]
    err = row["error_type"] if lbl == 0 and str(row["error_type"]) != "0" else ""
    err_str = f" [{err}]" if err else ""
    print(f"[{lbl} — {label_names.get(lbl, '?')}{err_str}] {row['text'][:200]}")
    print()

df["text_length"] = df["text"].fillna("").astype(str).apply(len)

print("=" * 60)
print("СТАТИСТИКА ДЛИН ТЕКСТОВ (символы)")
print("=" * 60)
print(df["text_length"].describe())
print()

print("=" * 60)
print(f"СТАТИСТИКА ДЛИН ТЕКСТОВ (токены, модель: {MODEL_NAME})")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
token_lengths = df["text"].fillna("").astype(str).apply(
    lambda t: len(tokenizer.encode(t, add_special_tokens=True))
)
df["token_length"] = token_lengths

print(df["token_length"].describe())
pct_truncated = (df["token_length"] > MAX_LENGTH).mean() * 100
print(f"\nДоля текстов длиннее MAX_LENGTH={MAX_LENGTH}: {pct_truncated:.1f}%")
print()

plt.figure(figsize=(10, 5))
plt.hist(df["text_length"], bins=50, edgecolor="black")
plt.title("RuCoLA — Распределение длин текстов (символы)")
plt.xlabel("Длина текста (символы)")
plt.ylabel("Количество")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "text_length_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(df["token_length"], bins=50, edgecolor="black")
plt.axvline(MAX_LENGTH, color="red", linestyle="--", label=f"MAX_LENGTH={MAX_LENGTH}")
plt.title("RuCoLA — Распределение длин текстов в токенах")
plt.xlabel("Количество токенов")
plt.ylabel("Количество текстов")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "token_length_distribution.png"), dpi=150)
plt.close()

class_labels_for_plot = [f"{lid} ({label_names.get(lid, '?')})" for lid in class_counts.index]
plt.figure(figsize=(7, 7))
plt.pie(class_counts.values, labels=class_labels_for_plot, autopct="%1.1f%%", startangle=140)
plt.title("RuCoLA — Доля классов")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "class_pie_chart.png"), dpi=150)
plt.close()

error_types_clean = unacceptable[unacceptable["error_type"].astype(str) != "0"]["error_type"].value_counts()
if len(error_types_clean) > 0:
    plt.figure(figsize=(8, 5))
    error_types_clean.plot(kind="barh", edgecolor="black", color="coral")
    plt.title("RuCoLA — Типы ошибок (unacceptable)")
    plt.xlabel("Количество")
    plt.ylabel("Тип ошибки")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "error_types.png"), dpi=150)
    plt.close()
