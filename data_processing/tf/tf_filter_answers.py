# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from data_processing.constants_for_clean import LLM_NAMES
from util.struct import Task
from util.util_func import save_df_to_csv

TASK_NAME = 'tf'
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]

df_clean = pd.read_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_answers.csv')
df_clean.set_index('id', inplace=True)

consistent_indices = []
# Iterate over rows in the DataFrame
for idx, row in df_clean.iterrows():
	correct_answer_cnt = 0
	# Compare each LLM's answer to the ground truth
	for col in ANSWER_COL_NAMES:
		if pd.isna(row[col]):
			continue
		elif row[col] == row['answer_ground_truth']:
			correct_answer_cnt += 1
	# If all LLM's answers are correct, record the index
	if correct_answer_cnt == len(ANSWER_COL_NAMES):
		consistent_indices.append(idx)

# Save unfiltered answers
df_unfiltered = df_clean.dropna(subset=ANSWER_COL_NAMES)
save_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_answers.csv'
save_df_to_csv(df=df_unfiltered, path=save_path, index=True)

# Filter out rows where all LLM's answers are correct
df_filtered = df_unfiltered.drop(consistent_indices).dropna(subset=ANSWER_COL_NAMES)
save_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv'
save_df_to_csv(df=df_filtered, path=save_path, index=True)

summary = f"There were {len(consistent_indices)} rows deleted, and {len(df_clean) - len(consistent_indices)} rows remaining.\n"
summary += f"The deleted rows' indices are: {consistent_indices}"
# Write summary to a text file
with open(f"../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtration_summary.txt", "w") as text_file:
	text_file.write(summary)
