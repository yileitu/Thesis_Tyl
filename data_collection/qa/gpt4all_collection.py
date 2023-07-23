# -*- coding: utf-8 -*-
from typing import List

import pandas as pd
from datasets import DatasetDict, load_dataset

SEED: int = 21946520
NUM_SAMPLES_EACH_CLS: int = 3000

FILTERED_CLS: List[str] = [
	"pacovaldez/stackoverflow-questions", "nomic-ai", "unified_unifiedskg_instructions", "unified_multi_sum",
	"unified_chip2", "unified_hc3_human"
	]

dataset: DatasetDict = load_dataset('nomic-ai/gpt4all-j-prompt-generations')
df = pd.DataFrame(dataset['train'])
source_counts = df['source'].value_counts()
for source, count in source_counts.items():
	print(f"Classification: {source}, Count: {count}")
print()

df_filtered = df[df['source'].isin(FILTERED_CLS)]
sampled_df = df_filtered.groupby('source').apply(
	lambda x: x.sample(n=NUM_SAMPLES_EACH_CLS, random_state=SEED)
	).reset_index(drop=True)
sampled_df = sampled_df.rename(columns={'prompt': 'input', 'source': 'category'})
sampled_df = sampled_df[['input', 'response', 'category']]
sampled_df.to_csv('../../data/sampled/gpt4all.csv', index=False)

source_counts = sampled_df['category'].value_counts()
for source, count in source_counts.items():
	print(f"Classification: {source}, Count: {count}")
