# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from constants_for_clean import LLM_NAMES, OPTION_E_DATASETS
from util.struct import Task
from util.util_func import save_df_to_csv

FILTERED: bool = False
TASK: Task = Task.MC
TASK_NAME = 'mc'
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]

if FILTERED:
	df_answer = pd.read_csv(f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv')
else:
	df_answer = pd.read_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answers.csv')
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
# Initialize additional lists to hold the data for "with E" and "without E" options
correct_with_e_list = [0] * len(LLM_NAMES)
correct_without_e_list = [0] * len(LLM_NAMES)
total_with_e_list = [0] * len(LLM_NAMES)
total_without_e_list = [0] * len(LLM_NAMES)
accuracy_with_e_list = [0] * len(LLM_NAMES)
accuracy_without_e_list = [0] * len(LLM_NAMES)
stat_df = pd.DataFrame(
	{
		'llm_name'                   : LLM_NAMES,
		'correct'                    : correct_list,
		'incorrect'                  : incorrect_list,
		'nonempty'                   : nonempty_cnt_list,
		'accuracy_excl_nan'          : accuracy_excl_nan_list,
		'missing'                    : missing_cnt_list,
		'total'                      : total_cnt_list,
		'accuracy_incl_nan'          : accuracy_incl_nan_list,
		'total_with_E_excl_nan'      : total_with_e_list,
		'accuracy_with_E_excl_nan'   : accuracy_with_e_list,
		'total_without_E_excl_nan'   : total_without_e_list,
		'accuracy_without_E_excl_nan': accuracy_without_e_list,
		'prop_A_with_E'              : [0] * len(LLM_NAMES),
		'prop_B_with_E'              : [0] * len(LLM_NAMES),
		'prop_C_with_E'              : [0] * len(LLM_NAMES),
		'prop_D_with_E'              : [0] * len(LLM_NAMES),
		'prop_E_with_E'              : [0] * len(LLM_NAMES),
		'prop_A_without_E'           : [0] * len(LLM_NAMES),
		'prop_B_without_E'           : [0] * len(LLM_NAMES),
		'prop_C_without_E'           : [0] * len(LLM_NAMES),
		'prop_D_without_E'           : [0] * len(LLM_NAMES),
		}
	)

if not FILTERED:
	df_answer.dropna(subset=ANSWER_COL_NAMES, inplace=True)

# Separate df_clean into two groups: with Option E and without Option E
df_with_e = df_answer[df_answer.apply(lambda x: (x['data_source'], x['subject']) in OPTION_E_DATASETS, axis=1)]
df_without_e = df_answer[df_answer.apply(lambda x: (x['data_source'], x['subject']) not in OPTION_E_DATASETS, axis=1)]

# Calculate accuracy for "with E" group
for idx, row in df_with_e.iterrows():
	for i, col in enumerate(ANSWER_COL_NAMES):
		if pd.isna(row[col]):
			continue
		total_with_e_list[i] += 1
		if row[col] == row['answer_ground_truth']:
			correct_with_e_list[i] += 1

# Calculate accuracy for "without E" group
for idx, row in df_without_e.iterrows():
	for i, col in enumerate(ANSWER_COL_NAMES):
		if pd.isna(row[col]):
			continue
		total_without_e_list[i] += 1
		if row[col] == row['answer_ground_truth']:
			correct_without_e_list[i] += 1

# Calculate accuracy percentages
accuracy_with_e_list = [round((correct / total) * 100, 2) if total > 0 else 0 for correct, total in
                        zip(correct_with_e_list, total_with_e_list)]
accuracy_without_e_list = [round((correct / total) * 100, 2) if total > 0 else 0 for correct, total in
                           zip(correct_without_e_list, total_without_e_list)]

# Extend stat_df DataFrame
stat_df['total_with_E_excl_nan'] = total_with_e_list
stat_df['accuracy_with_E_excl_nan'] = accuracy_with_e_list
stat_df['total_without_E_excl_nan'] = total_without_e_list
stat_df['accuracy_without_E_excl_nan'] = accuracy_without_e_list

# Calculate proportions for the group with Option E
for llm, col in zip(LLM_NAMES, ANSWER_COL_NAMES):
	counts_with_e = df_with_e[col].value_counts(normalize=True)
	for letter in 'ABCDE':
		stat_df.loc[stat_df['llm_name'] == llm, f'prop_{letter}_with_E'] = counts_with_e.get(letter, 0)

# Calculate proportions for the group without Option E
for llm, col in zip(LLM_NAMES, ANSWER_COL_NAMES):
	counts_without_e = df_without_e[col].value_counts(normalize=True)
	for letter in 'ABCD':  # No 'E' in this group
		stat_df.loc[stat_df['llm_name'] == llm, f'prop_{letter}_without_E'] = counts_without_e.get(letter, 0)

if FILTERED:
	save_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_stats.csv'
else:
	save_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_stats.csv'
save_df_to_csv(df=stat_df, path=save_path, index=False)
