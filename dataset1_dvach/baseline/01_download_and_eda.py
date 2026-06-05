import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import DATASET_NAME1, MODEL_NAME, MAX_LENGTH

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

dataset = load_dataset(DATASET_NAME1)
print(f"Датасет загружен: {dataset}")

dfs = []
for split_name in dataset:
    df_split = dataset[split_name].to_pandas()
    df_split["original_split"] = split_name
    dfs.append(df_split)
df = pd.concat(dfs, ignore_index=True)

raw_csv_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
df.to_csv(raw_csv_path, index=False)
print(f"Полный датасет сохранён: {raw_csv_path}")

print("=" * 50)
print(f"Размер датасета: {len(df)} строк")
print(f"Колонки: {list(df.columns)}")
print(f"Типы данных:\n{df.dtypes}")
print()

print("=" * 50)
print("Пропуски")
missing = df.isnull().sum()
print(missing)
print()

print("=" * 50)
print("Дубликаты")
text_col = df.columns[0] 
duplicates = df.duplicated(subset=[text_col]).sum()
print(f"Полных дубликатов по колонке '{text_col}': {duplicates}")
print()

print("=" * 50)
print("Распределение классов")
label_col = [c for c in df.columns if c not in [text_col, "original_split"]][0]
class_counts = df[label_col].value_counts()
print(class_counts)

print("=" * 50)
print("Примеры")
for i, row in df.head(5).iterrows():
    print(f"[{row[label_col]}] {row[text_col][:200]}")
    print()

df["text_length"] = df[text_col].fillna("").astype(str).apply(len)
print("=" * 50)
print("СТАТИСТИКА ДЛИН ТЕКСТОВ (символы)")
print("=" * 50)
print(df["text_length"].describe())
print()

print("=" * 50)
print(f"СТАТИСТИКА ДЛИН ТЕКСТОВ (токены, модель: {MODEL_NAME})")
print("=" * 50)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
token_lengths = df[text_col].fillna("").astype(str).apply(
    lambda t: len(tokenizer.encode(t, add_special_tokens=True))
)
df["token_length"] = token_lengths
print(df["token_length"].describe())
pct_truncated = (df["token_length"] > MAX_LENGTH).mean() * 100
print(f"\nДоля текстов длиннее MAX_LENGTH={MAX_LENGTH}: {pct_truncated:.1f}%")
print()

plt.figure(figsize=(10, 5))
plt.hist(df["text_length"], bins=50, edgecolor="black")
plt.title("Распределение длин текстов")
plt.xlabel("Длина текста (символы)")
plt.ylabel("Количество")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "text_length_distribution.png"), dpi=150)
plt.close()
print("Сохранено: results/text_length_distribution.png")

plt.figure(figsize=(10, 5))
plt.hist(df["token_length"], bins=50, edgecolor="black")
plt.axvline(MAX_LENGTH, color="red", linestyle="--", label=f"MAX_LENGTH={MAX_LENGTH}")
plt.title("Распределение длин текстов в токенах")
plt.xlabel("Количество токенов")
plt.ylabel("Количество текстов")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "token_length_distribution.png"), dpi=150)
plt.close()
print("Сохранено: results/token_length_distribution.png")

plt.figure(figsize=(7, 7))
plt.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Доля классов в датасете")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_pie_chart.png"), dpi=150)
plt.close()
print("Сохранено: results/class_pie_chart.png")
