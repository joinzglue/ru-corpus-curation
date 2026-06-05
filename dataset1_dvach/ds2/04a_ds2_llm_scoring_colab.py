# train.csv, config.py

import os
import json
import numpy as np
import pandas as pd
import torch
import regex as re
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
import concurrent.futures

from openai import OpenAI
from config import (
    SEED, DS2_API_MAX_WORKERS, DS2_API_MAX_RETRIES, DS2_API_BATCH_SIZE,
)

RESULTS_DIR = "./results_ds2"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://foundation-models.api.cloud.ru/v1"
)
DEPLOYMENT_MODEL = "Qwen/Qwen3-Coder-Next"

train_df = pd.read_csv("train.csv")
text_col = train_df.columns[0]
print(f"Train: {len(train_df)} примеров, text_col={text_col}")

pre_prompt = """Ты — эксперт по оценке качества данных. Твоя задача — оценить качество текстового примера по критериям: Rarity (редкость), Complexity (сложность), Informativeness (информативность).
Оцени каждый критерий по шкале от 1 до 10 и выведи итоговую оценку от 1 до 10, где более высокий балл означает более высокое качество примера.
Старайся не концентрировать оценки вокруг одного значения. Если примеры похожи по качеству, распределяй баллы равномерно, отражая тонкие различия.
Внимательно оцени следующий пример и верни целочисленные оценки в формате JSON:
{
    "Rarity": <число, 1-10>,
    "Complexity": <число, 1-10>,
    "Informativeness": <число, 1-10>,
    "Overall rating": <число, 1-10>
}
"""

inputs = []
for _, row in train_df.iterrows():
    data_text = f"## Пример данных:\n{row[text_col]}"
    inputs.append(pre_prompt + data_text + "\n### Оценка:")

def score_compress(original_scores):
    overall_scores = [score[-1] for score in original_scores]
    print(f"Original score distribution: {Counter(overall_scores)}")
    scores_revised = []
    for score in overall_scores:
        if score <= 4:
            scores_revised.append(4)
        elif score >= 9:
            scores_revised.append(9)
        else:
            scores_revised.append(score)
    scores_revised = [score - 4 for score in scores_revised]
    print(f"Revised scores distribution: {Counter(scores_revised)}")
    return scores_revised

json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

def fetch_content(input_text, idx):
    completion = client.chat.completions.create(
        model=DEPLOYMENT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
        max_tokens=2500,
        temperature=0.5,
        top_p=0.95,
        presence_penalty=0,
    )
    msg = completion.choices[0].message
    content = msg.content
    if not content:
        content = getattr(msg, "reasoning_content", None) or ""
    return idx, content

total_output_scores = []
batch_size = DS2_API_BATCH_SIZE
split_size = len(inputs) // batch_size + (1 if len(inputs) % batch_size else 0)

print(f"Total: {len(inputs)}, Batch size: {batch_size}, Batches: {split_size}")

for batch_idx in tqdm(range(split_size), desc="API scoring"):
    batch_path = os.path.join(RESULTS_DIR, f"output_scores_batch_{batch_idx}.pt")
    if os.path.exists(batch_path):
        batch_scores = torch.load(batch_path, weights_only=False)
        total_output_scores.extend(batch_scores)
        continue

    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, len(inputs))
    batch_inputs = inputs[batch_start:batch_end]
    output_scores = [None] * len(batch_inputs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=DS2_API_MAX_WORKERS) as executor:
        future_to_input = {
            executor.submit(fetch_content, inp, idx): idx
            for idx, inp in enumerate(batch_inputs)
        }
        for future in concurrent.futures.as_completed(future_to_input):
            idx = future_to_input[future]
            try:
                idx, content = future.result()
                matches = json_pattern.findall(content)
                retry_count = 0
                while not matches and retry_count < DS2_API_MAX_RETRIES:
                    retry_count += 1
                    idx, content = fetch_content(batch_inputs[idx], idx)
                    matches = json_pattern.findall(content)
                if matches:
                    try:
                        json_obj = json.loads(matches[-1])
                        keys_lower = {k.lower(): v for k, v in json_obj.items()}
                        output_scores[idx] = [
                            int(keys_lower['rarity']), int(keys_lower['complexity']),
                            int(keys_lower['informativeness']), int(keys_lower['overall rating']),
                        ]
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        print(f"Parse error idx {batch_start + idx}: {e}, keys={list(json_obj.keys())}")
                        output_scores[idx] = [0, 0, 0, 0]
                else:
                    output_scores[idx] = [0, 0, 0, 0]
            except Exception as exc:
                print(f"Exception idx {batch_start + idx}: {exc}")
                output_scores[idx] = [0, 0, 0, 0]

    torch.save(output_scores, batch_path)
    total_output_scores.extend(output_scores)
    print(f"Batch {batch_idx}/{split_size}: done")

print(f"\nTotal scored: {len(total_output_scores)}")
unrated = sum(1 for s in total_output_scores if s[-1] == 0)
print(f"Unrated: {unrated} ({unrated / len(total_output_scores) * 100:.1f}%)")

compressed_scores = score_compress(total_output_scores)
torch.save(total_output_scores, os.path.join(RESULTS_DIR, "ds2_raw_scores.pt"))
torch.save(compressed_scores, os.path.join(RESULTS_DIR, "ds2_compressed_scores.pt"))

raw_arr = np.array(total_output_scores)
criteria = ["Rarity", "Complexity", "Informativeness", "Overall rating"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (ax, name) in enumerate(zip(axes.flat, criteria)):
    ax.hist(raw_arr[:, i], bins=range(0, 12), edgecolor="black", alpha=0.7)
    ax.set_title(name)
fig.suptitle("DS2: Raw LLM Scores — Dvach", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_distribution_raw.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
counter = Counter(compressed_scores)
labels_sorted = sorted(counter.keys())
ax.bar([str(k) for k in labels_sorted], [counter[k] for k in labels_sorted], edgecolor="black", alpha=0.7)
ax.set_title("DS2: Compressed Scores (0-5) — Dvach")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_distribution_compressed.png"), dpi=150)
plt.show()

