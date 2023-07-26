# -*- coding: utf-8 -*-
import json
from typing import Dict

import pandas as pd
from datasets import DatasetDict, load_dataset

SEED: int = 21946520
NUM_SAMPLES: int = 15000
DATASETS_PATHS_DIR: str = "qa_datasets_hf_path.json"

with open(DATASETS_PATHS_DIR, 'r') as f:
	dataset_name2hf_path: Dict[str, str] = json.load(f)

for name, path in dataset_name2hf_path.items():
	dataset: DatasetDict = load_dataset(path)
	df = pd.DataFrame(dataset['train'])
	sampled_df = df.sample(n=NUM_SAMPLES, random_state=SEED)
	sampled_df['input'] = sampled_df.apply(
		lambda row: row['instruction'] if row['input'] == "" else row['instruction'] + "\n\n" + row['input'], axis=1
		)
	sampled_df.drop(columns=["instruction"], inplace=True)
	sampled_df.rename(columns={"output": "response"}, inplace=True)
	sampled_df.to_csv(f'../../data/sampled/qa/{name}.csv', index=False)
	print(f"Finished sampling '{name}' dataset.")
