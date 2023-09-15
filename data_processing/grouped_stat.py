from typing import List

import pandas as pd

from constants_for_clean import LLM_NAMES
from util.struct import Task
from util.util_func import save_df_to_csv

FILTERED: bool = False
TASK: Task = Task.MC
TASK_NAME = 'mc'
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]

# Group statistics by 'data_source' and 'subject' columns
# Create an empty DataFrame to store grouped statistics
grouped_stats = pd.DataFrame(columns=['data_source', 'subject', 'llm_name', 'accuracy excl nan (%)'])
if FILTERED:
	df_for_group_stat = pd.read_csv(f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv')
else:
	df_for_group_stat = pd.read_csv(f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_answers.csv')
df_for_group_stat.set_index('id', inplace=True)

# Group by 'data_source' and 'subject' columns
for (data_source, subject), group_df in df_for_group_stat.groupby(['data_source', 'subject']):
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
if FILTERED:
	save_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_grouped_stats.csv'
else:
	save_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_grouped_stats.csv'
save_df_to_csv(df=grouped_stats, path=save_path, index=False)
