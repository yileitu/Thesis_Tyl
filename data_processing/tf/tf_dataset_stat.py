# -*- coding: utf-8 -*-
import pandas as pd

TASK_NAME = 'tf'
FILTERED: bool = False
ORIGINAL: bool = False

if ORIGINAL:
	data_path = f'../../data/output/{TASK_NAME}/{TASK_NAME}.csv'
else:
	if FILTERED:
		data_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv'
	else:
		data_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_answers.csv'
tf_data = pd.read_csv(data_path)

# Initialize DataFrame to store ground truth statistics
gt_stat_df = pd.DataFrame(
	{
		'prop_gt_True' : [0],
		'prop_gt_False': [0],
		}
	)

# Calculate ground truth proportions for the group with Option E
gt_col_name = 'answer' if ORIGINAL else 'answer_ground_truth'
tf_cnts = tf_data[gt_col_name].value_counts(normalize=True)
for answer in [True, False]:
	gt_stat_df[f'prop_gt_{answer}'] = tf_cnts.get(answer, 0)

if ORIGINAL:
	gt_stat_save_path = f'../../data/processed/{TASK_NAME}/{TASK_NAME}_gt_stats.csv'
else:
	if FILTERED:
		gt_stat_save_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_gt_stats.csv'
	else:
		gt_stat_save_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_gt_stats.csv'
gt_stat_df.to_csv(gt_stat_save_path, index=False)

# 3. Majority Accuracy
bool_percentages = tf_data[gt_col_name].value_counts(normalize=True)
max_percentage = bool_percentages.max()
max_bool = bool_percentages.idxmax()

majority = pd.DataFrame(
	{
		'llm_name'         : ["Majority (Baseline)"],
		'accuracy_excl_nan': [round(max_percentage * 100, 2)],
		'majority_bool'    : [max_bool],
		}
	)
if ORIGINAL:
	majority_save_path = f'../../data/processed/{TASK_NAME}/{TASK_NAME}_majority.csv'
else:
	if FILTERED:
		majority_save_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_majority.csv'
	else:
		majority_save_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_majority.csv'
majority.to_csv(majority_save_path, index=False)
