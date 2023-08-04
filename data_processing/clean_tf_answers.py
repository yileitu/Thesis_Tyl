# -*- coding: utf-8 -*-
import re
from typing import Optional

import pandas as pd


def extract_bool(response: Optional[str]) -> Optional[str]:
	"""
	Extracts boolean value from a response
	:param response: response from the LLM
	:return: TRUE or FALSE or None
	"""
	response = str(response)
	match = re.search(r'\b(TRUE|FALSE)\b', response, re.IGNORECASE)
	return match.group(0).upper() if match else None


DATA_PATH = '../data/output/boolq.csv'
df = pd.read_csv(DATA_PATH)
df.reset_index(level=0, inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)
df.to_csv(DATA_PATH, index=False)

df_answers = df[['id', 'answer']].copy()
df_answers.rename(columns={'answer': 'answer_ground_truth'}, inplace=True)

for col in ['response_FLAN-T5-large', 'response_Vicuna-13b', 'response_gpt-3.5-turbo']:
	df_answers[f'{col.replace("response_", "answer_")}'] = df.apply(
		lambda row: extract_bool(response=row[col]), axis=1
		)
missing_values_info = ""
for col in ['answer_FLAN-T5-large', 'answer_Vicuna-13b', 'answer_gpt-3.5-turbo']:
	missing_values = df_answers[col].isna()
	missing_indices = missing_values.where(missing_values).dropna().index.tolist()
	missing_values_info += f"Column {col} has {len(missing_indices)} missing values at indices {missing_indices}\n"

# Write missing values info to a text file
with open("../data/processed/tf/tf_missing_values_info.txt", "w") as text_file:
	text_file.write(missing_values_info)

# Drop rows with any missing values
df_clean = df_answers.dropna()
df_clean.to_csv('../data/processed/tf/tf_answers.csv', index=False)
