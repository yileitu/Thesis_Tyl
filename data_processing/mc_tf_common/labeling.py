# -*- coding: utf-8 -*-
import os
from collections import Counter
from typing import List

import pandas as pd

from data_processing.mc_tf_common.constants_for_clean import LLM_NAMES
from util.struct import Task
from util.util_func import save_df_to_csv

TASK: Task = Task.TF
if TASK == Task.TF:
	TASK_NAME = 'tf'
elif TASK == Task.MC:
	TASK_NAME = 'mc'
else:
	raise ValueError(f'Unsupported task: {TASK}')
ANSWER_COL_NAMES: List[str] = [f'answer_{llm_name}' for llm_name in LLM_NAMES]
FILTERED: bool = False


# Define a function to determine the label
def determine_label(row) -> str:
	"""
	Determine the "difficulty" label for each question-answer pair.

	:param row: data point
	:return: Difficulty label
	"""
	if row['answer_pythia-2.8b'] == row['answer_ground_truth']:
		return 'Simple'
	elif row['answer_Llama-2-7b-chat'] == row['answer_ground_truth']:
		return 'Middle'
	elif row['answer_Llama-2-13b-chat'] == row['answer_ground_truth']:
		return 'Difficult'
	elif row['answer_gpt-3.5-turbo'] == row['answer_ground_truth']:
		return 'Extremely Difficult'
	else:
		return 'Unsolvable'


def determine_anomaly_and_model(row):
	# Define the order of answer capability
	model_order = ['answer_pythia-2.8b', 'answer_Llama-2-7b-chat', 'answer_Llama-2-13b-chat', 'answer_gpt-3.5-turbo']

	# Find the ground truth
	ground_truth = row['answer_ground_truth']

	# This list will keep track of models causing anomalies
	anomalous_models = []

	# Check for anomalies: a stronger model should not fail if a weaker model succeeds
	for i, model in enumerate(model_order[:-1]):  # Skip the last model, as there's no stronger model to compare
		# If the current model's answer is correct
		if row[model] == ground_truth:
			# Check if any stronger model has answered incorrectly
			for stronger_model in model_order[i + 1:]:
				if row[stronger_model] != ground_truth:
					# Record the stronger model that failed
					# Remove the "answer_" prefix from the model name
					anomalous_models.append(stronger_model.replace('answer_', ''))

	# If there are any anomalous models, return a string indicating the anomaly and the models
	if anomalous_models:
		return 'Anomalous', ', '.join(anomalous_models)
	return 'Normal', ''


def split_and_count_models(anomalous_models_column):
	"""
	Count the occurrences of each model in the anomalous cases.
	:param anomalous_models_column: column of data containing the models causing anomalies
	:return: occurrences
	"""
	return Counter([model for models in anomalous_models_column for model in models.split(', ') if model])

# Read processed MC/TF answers
if FILTERED:
	answers_path = f'../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_answers.csv'
else:
	answers_path = f'../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_answers.csv'
df_answers: pd.DataFrame = pd.read_csv(answers_path, index_col='id')
directory_path = os.path.dirname(answers_path)

# Label each MC/TF question-answer pair
df_answers['label'] = df_answers.apply(determine_label, axis=1)
df_answers['anomaly'], df_answers['anomalous_models'] = zip(
	*df_answers.apply(determine_anomaly_and_model, axis=1)
	)

labeled_dir_path = os.path.join(directory_path, 'labeled')
labeled_answers_path = os.path.join(labeled_dir_path, f'{TASK_NAME}_labeled_answers.csv')
save_df_to_csv(df=df_answers, path=labeled_answers_path, index=True)

# # Statistics
# Calculate the total proportion of Anomalous cases
total_anomalous_proportion = (df_answers['anomaly'] == 'Anomalous').mean()

# Count occurrences of each model in the anomalous cases
model_counts = split_and_count_models(df_answers[df_answers['anomaly'] == 'Anomalous']['anomalous_models'])

# Total number of anomalies for calculating proportions
total_anomalies = sum(model_counts.values())

# Prepare the content to be written to the file
stat_content = f"Total Proportion of Anomalous cases: {total_anomalous_proportion:.2%}\n\n"
stat_content += "Anomalous Counts and Proportions by Model:\n"
for model, count in model_counts.items():
	proportion = count / total_anomalies
	stat_content += f"{model}: Count = {count}, Proportion = {proportion:.2%}\n"

stat_txt_path = os.path.join(labeled_dir_path, f'{TASK_NAME}_labeled_data_stats.txt')
with open(stat_txt_path, 'w') as file:
	file.write(stat_content)
