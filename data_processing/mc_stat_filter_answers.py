# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from constants_for_clean import LLM_NAMES, OPTION_E_DATASETS
from util.struct import Task

TASK: Task = Task.MC
TASK_NAME = 'mc'
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

# Separate df_clean into two groups: with Option E and without Option E
df_with_e = df_clean[df_clean.apply(lambda x: (x['data_source'], x['subject']) in OPTION_E_DATASETS, axis=1)]
df_without_e = df_clean[df_clean.apply(lambda x: (x['data_source'], x['subject']) not in OPTION_E_DATASETS, axis=1)]

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

stat_df.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_stats.csv', index=False)

# Summarize the results
summary = f"There were {len(consistent_indices)} rows deleted, and {len(df_clean) - len(consistent_indices)} rows remaining.\n"
summary += f"The deleted rows' indices are: {consistent_indices}"
# Write summary to a text file
with open(f"../data/processed/{TASK_NAME}/{TASK_NAME}_summary.txt", "w") as text_file:
	text_file.write(summary)

df_filtered = df_clean.drop(consistent_indices).dropna()  # Remove rows where all LLM's answers are correct
df_filtered.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answers_filtered.csv', index=True)

# Group statistics by 'data_source' and 'subject' columns
# Create an empty DataFrame to store grouped statistics
grouped_stats = pd.DataFrame(columns=['data_source', 'subject', 'llm_name', 'accuracy excl nan (%)'])

# Group by 'data_source' and 'subject' columns
for (data_source, subject), group_df in df_clean.groupby(['data_source', 'subject']):
	# Initialize counts for correct and total answers
	correct_counts = {llm: 0 for llm in ANSWER_COL_NAMES}
	total_counts = {llm: 0 for llm in ANSWER_COL_NAMES}

	# Iterate over rows in the grouped DataFrame
	for _, row in group_df.iterrows():
		# Compare each LLM's answer to the ground truth
		for col in ANSWER_COL_NAMES:
			if pd.isna(row[col]):
				continue
			elif row[col] == row['answer_ground_truth']:
				correct_counts[col] += 1
			total_counts[col] += 1

	# Calculate and store accuracy in DataFrame
	for llm in LLM_NAMES:
		accuracy = (correct_counts[f'answer_{llm}'] / total_counts[f'answer_{llm}']) * 100 \
			if total_counts[f'answer_{llm}'] > 0 else 0
		new_row = pd.DataFrame(
			{
				'data_source'          : [data_source],
				'subject'              : [subject],
				'llm_name'             : [llm],
				'accuracy excl nan (%)': [round(accuracy, 2)]
				}
			)
		grouped_stats = pd.concat([grouped_stats, new_row], ignore_index=True)

# Save the grouped statistics as a CSV file
grouped_stats.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_grouped_stats.csv', index=False)

# Proportion of ground truth answers
# Initialize DataFrame to store ground truth statistics
gt_stat_df = pd.DataFrame(
	{
		'group'    : ['with_E', 'without_E'],
		'prop_gt_A': [0, 0],
		'prop_gt_B': [0, 0],
		'prop_gt_C': [0, 0],
		'prop_gt_D': [0, 0],
		'prop_gt_E': [0, None]
		}
	)

# Calculate ground truth proportions for the group with Option E
counts_gt_with_e = df_with_e['answer_ground_truth'].value_counts(normalize=True)
for letter in 'ABCDE':
	gt_stat_df.loc[gt_stat_df['group'] == 'with_E', f'prop_gt_{letter}'] = counts_gt_with_e.get(letter, 0)

# Calculate ground truth proportions for the group without Option E
counts_gt_without_e = df_without_e['answer_ground_truth'].value_counts(normalize=True)
for letter in 'ABCD':  # No 'E' in this group
	gt_stat_df.loc[gt_stat_df['group'] == 'without_E', f'prop_gt_{letter}'] = counts_gt_without_e.get(letter, 0)

gt_stat_df.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_gt_stats.csv', index=False)

# # Calculate the proportion of answers for each LLM
# result_storage = []
# # Loop over each data_source and subject
# for data_source, subject in OPTION_E_DATASETS:
# 	df_group = df_clean[(df_clean['data_source'] == data_source) & (df_clean['subject'] == subject)]
#
# 	# For each column related to LLMs, calculate the frequency of 'A', 'B', 'C', 'D', 'E'
# 	for col in ANSWER_COL_NAMES:
# 		counts_with_e = df_group[col].value_counts(normalize=True)  # Proportion
# 		counts_with_e = counts_with_e.loc[[c for c in 'ABCDE' if c in counts_with_e.index]].fillna(0)
#
# 		result_storage.append(
# 			{
# 				'data_source': data_source,
# 				'subject'    : subject,
# 				'llm_name'   : col,
# 				'option'     : 'With E',
# 				'A'          : counts_with_e.get('A', 0),
# 				'B'          : counts_with_e.get('B', 0),
# 				'C'          : counts_with_e.get('C', 0),
# 				'D'          : counts_with_e.get('D', 0),
# 				'E'          : counts_with_e.get('E', 0),
# 				}
# 			)
#
# # Do the same for groups without 'E'
# for (data_source, subject), group_df in df_clean.groupby(['data_source', 'subject']):
# 	if (data_source, subject) in OPTION_E_DATASETS:
# 		continue  # Skip, because we already processed these groups
#
# 	# For each column related to LLMs, calculate the frequency of 'A', 'B', 'C', 'D'
# 	for col in ANSWER_COL_NAMES:
# 		counts_without_e = group_df[col].value_counts(normalize=True)  # Proportion
# 		counts_without_e = counts_without_e.loc[[c for c in 'ABCD' if c in counts_without_e.index]].fillna(0)
#
# 		result_storage.append(
# 			{
# 				'data_source': data_source,
# 				'subject'    : subject,
# 				'llm_name'   : col,
# 				'option'     : 'Without E',
# 				'A'          : counts_without_e.get('A', 0),
# 				'B'          : counts_without_e.get('B', 0),
# 				'C'          : counts_without_e.get('C', 0),
# 				'D'          : counts_without_e.get('D', 0),
# 				'E'          : None,  # E is not applicable here
# 				}
# 			)
#
# # Convert to DataFrame for easier manipulation and saving to CSV
# result_df = pd.DataFrame(result_storage)
# result_df.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answer_proportions.csv', index=False)
