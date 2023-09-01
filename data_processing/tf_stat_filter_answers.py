# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from constants_for_clean import LLM_NAMES
from util.struct import Task

TASK: Task = Task.TF
TASK_NAME = 'tf'
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]

df_clean = pd.read_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answers.csv')
df_clean.set_index('id', inplace=True)

missing_cnt_list = []
for column in ANSWER_COL_NAMES:
	missing_count = df_clean[column].isna().sum()
	missing_cnt_list.append(missing_count)

# Initialize counts
correct_counts = {llm: 0 for llm in ANSWER_COL_NAMES}
incorrect_counts = {llm: 0 for llm in ANSWER_COL_NAMES}
consistent_indices = []

# Iterate over rows in the DataFrame
for idx, row in df_clean.iterrows():
	correct_answers = 0
	# Compare each LLM's answer to the ground truth
	for col in ANSWER_COL_NAMES:
		if pd.isna(row[col]):
			continue
		elif row[col] == row['answer_ground_truth']:
			correct_counts[col] += 1
			correct_answers += 1
		else:
			incorrect_counts[col] += 1
	# If all LLM's answers are correct, record the index
	if correct_answers == len(ANSWER_COL_NAMES):
		consistent_indices.append(idx)

# Create lists to hold the data
correct_list = []
incorrect_list = []

# Populate the lists from the counts
for llm in LLM_NAMES:
	correct_list.append(correct_counts[f'answer_{llm}'])
	incorrect_list.append(incorrect_counts[f'answer_{llm}'])

nonempty_cnt_list = [correct + incorrect for correct, incorrect in zip(correct_list, incorrect_list)]
accuracy_excl_nan_list = [round((correct / total) * 100, 2) if total > 0 else 0 for correct, total in
                          zip(correct_list, nonempty_cnt_list)]

total_cnt_list = [correct + incorrect + missing for correct, incorrect, missing in
                  zip(correct_list, incorrect_list, missing_cnt_list)]
accuracy_incl_nan_list = [round((correct / total) * 100, 2) if total > 0 else 0 for correct, total in
                          zip(correct_list, total_cnt_list)]

# Create the DataFrame
stat_df = pd.DataFrame(
	{
		'llm_name'         : LLM_NAMES,
		'correct'          : correct_list,
		'incorrect'        : incorrect_list,
		'nonempty'         : nonempty_cnt_list,
		'accuracy_excl_nan': accuracy_excl_nan_list,
		'missing'          : missing_cnt_list,
		'total'            : total_cnt_list,
		'accuracy_incl_nan': accuracy_incl_nan_list,
		}
	)
stat_df.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_stats.csv', index=False)

# Summarize the results
summary = f"There were {len(consistent_indices)} rows deleted, and {len(df_clean) - len(consistent_indices)} rows remaining.\n"
summary += f"The deleted rows' indices are: {consistent_indices}"
# Write summary to a text file
with open(f"../data/processed/{TASK_NAME}/{TASK_NAME}_summary.txt", "w") as text_file:
	text_file.write(summary)

df_filtered = df_clean.drop(consistent_indices).dropna()  # Remove rows where all LLM's answers are correct
df_filtered.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answers_filtered.csv', index=True)
