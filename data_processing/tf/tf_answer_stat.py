# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from data_processing.constants_for_clean import LLM_NAMES, OPTION_E_DATASETS
from util.util_func import save_df_to_csv

FILTERED: bool = True
TASK_NAME = 'tf'
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]

if FILTERED:
	df_answer = pd.read_csv(f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv')
else:
	df_answer = pd.read_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_answers.csv')
df_answer.set_index('id', inplace=True)

missing_cnt_list = []
for column in ANSWER_COL_NAMES:
	missing_count = df_answer[column].isna().sum()
	missing_cnt_list.append(missing_count)

# Initialize counts
correct_counts = {llm: 0 for llm in ANSWER_COL_NAMES}
incorrect_counts = {llm: 0 for llm in ANSWER_COL_NAMES}

# Iterate over rows in the DataFrame
for idx, row in df_answer.iterrows():
	# Compare each LLM's answer to the ground truth
	for col in ANSWER_COL_NAMES:
		if pd.isna(row[col]):
			continue
		elif row[col] == row['answer_ground_truth']:
			correct_counts[col] += 1
		else:
			incorrect_counts[col] += 1

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

if not FILTERED:
	df_answer.dropna(subset=ANSWER_COL_NAMES, inplace=True)

for llm, col in zip(LLM_NAMES, ANSWER_COL_NAMES):
	tf_cnts = df_answer[col].value_counts(normalize=True)
	for answer in [True, False]:
		stat_df.loc[stat_df['llm_name'] == llm, f'prop_{answer}'] = tf_cnts.get(answer, 0)


if FILTERED:
	save_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_stats.csv'
else:
	save_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_stats.csv'
save_df_to_csv(df=stat_df, path=save_path, index=False)
