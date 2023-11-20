# -*- coding: utf-8 -*-
import pandas as pd
from data_processing.mc_tf_common.constants_for_clean import OPTION_E_DATASETS

TASK_NAME = 'mc'
FILTERED: bool = True
ORIGINAL: bool = False

if ORIGINAL:
	data_path = f'../../data/output/mc/mc.csv'
else:
	if FILTERED:
		data_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv'
	else:
		data_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_answers.csv'
mc_data = pd.read_csv(data_path)

# 2. Proportion of ground truth answers
df_with_e = mc_data[mc_data.apply(lambda x: (x['data_source'], x['subject']) in OPTION_E_DATASETS, axis=1)]
df_without_e = mc_data[mc_data.apply(lambda x: (x['data_source'], x['subject']) not in OPTION_E_DATASETS, axis=1)]

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
gt_col_name = 'answer' if ORIGINAL else 'answer_ground_truth'
counts_gt_with_e = df_with_e[gt_col_name].value_counts(normalize=True)
for letter in 'ABCDE':
	gt_stat_df.loc[gt_stat_df['group'] == 'with_E', f'prop_gt_{letter}'] = counts_gt_with_e.get(letter, 0)

# Calculate ground truth proportions for the group without Option E
counts_gt_without_e = df_without_e[gt_col_name].value_counts(normalize=True)
for letter in 'ABCD':  # No 'E' in this group
	gt_stat_df.loc[gt_stat_df['group'] == 'without_E', f'prop_gt_{letter}'] = counts_gt_without_e.get(letter, 0)

if ORIGINAL:
	gt_stat_save_path = f'../data/processed/{TASK_NAME}/{TASK_NAME}_gt_stats.csv'
else:
	if FILTERED:
		gt_stat_save_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_gt_stats.csv'
	else:
		gt_stat_save_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_gt_stats.csv'
gt_stat_df.to_csv(gt_stat_save_path, index=False)

# 3. Majority Accuracy
letter_percentages = mc_data[gt_col_name].value_counts(normalize=True)
max_percentage = letter_percentages.max()
max_letter = letter_percentages.idxmax()

max_percentage_with_e = counts_gt_with_e.max()
max_letter_with_e = counts_gt_with_e.idxmax()

max_percentage_without_e = counts_gt_without_e.max()
max_letter_without_e = counts_gt_without_e.idxmax()

majority = pd.DataFrame(
	{
		'llm_name'                   : ["Majority (Baseline)"],
		'accuracy_excl_nan'          : [round(max_percentage * 100, 2)],
		'majority_letter'            : [max_letter],
		'accuracy_with_E_excl_nan'   : [round(max_percentage_with_e * 100, 2)],
		'majority_letter_with_E'     : [max_letter_with_e],
		'accuracy_without_E_excl_nan': [round(max_percentage_without_e * 100, 2)],
		'majority_letter_without_E'  : [max_letter_without_e],
		}
	)
if ORIGINAL:
	majority_save_path = f'../data/processed/{TASK_NAME}/{TASK_NAME}_majority.csv'
else:
	if FILTERED:
		majority_save_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_majority.csv'
	else:
		majority_save_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_majority.csv'
majority.to_csv(majority_save_path, index=False)
