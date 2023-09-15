# -*- coding: utf-8 -*-
import pandas as pd
from constants_for_clean import LLM_NAMES, OPTION_E_DATASETS

TASK_NAME = 'mc'
mc_data = pd.read_csv('../data/output/mc/mc.csv')

# 1. Check if Option E is empty
groups = mc_data.groupby(['data_source', 'subject'])

nonempty_option_e_list = []
empty_option_e_list = []
for name, group in groups:
	if group['option_E'].notna().any():
		nonempty_option_e_list.append(name)
	else:
		empty_option_e_list.append(name)

print(f'Nonempty Option E List: {nonempty_option_e_list}')
print(f'Empty Option E lList: {empty_option_e_list}')

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
counts_gt_with_e = df_with_e['answer'].value_counts(normalize=True)
for letter in 'ABCDE':
	gt_stat_df.loc[gt_stat_df['group'] == 'with_E', f'prop_gt_{letter}'] = counts_gt_with_e.get(letter, 0)

# Calculate ground truth proportions for the group without Option E
counts_gt_without_e = df_without_e['answer'].value_counts(normalize=True)
for letter in 'ABCD':  # No 'E' in this group
	gt_stat_df.loc[gt_stat_df['group'] == 'without_E', f'prop_gt_{letter}'] = counts_gt_without_e.get(letter, 0)

gt_stat_df.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_gt_stats.csv', index=False)

# 3. Majority Accuracy
letter_percentages = mc_data['answer'].value_counts(normalize=True)
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
majority.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_majority.csv', index=False)
