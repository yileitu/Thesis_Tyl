# -*- coding: utf-8 -*-
import pandas as pd

df_clean = pd.read_csv('../data/processed/mc/mc_answers.csv')
df_clean.set_index('id', inplace=True)
LLM_COLS = ['answer_FLAN-T5-large', 'answer_Vicuna-13b', 'answer_gpt-3.5-turbo']

# Initialize counts
correct_counts = {
	'answer_FLAN-T5-large': 0,
	'answer_Vicuna-13b'   : 0,
	'answer_gpt-3.5-turbo': 0
	}
incorrect_counts = {
	'answer_FLAN-T5-large': 0,
	'answer_Vicuna-13b'   : 0,
	'answer_gpt-3.5-turbo': 0
	}
consistent_indices = []

# Iterate over rows in the DataFrame
for idx, row in df_clean.iterrows():
	correct_answers = 0
	# Compare each LLM's answer to the ground truth
	for col in LLM_COLS:
		if row[col] == row['answer_ground_truth']:
			correct_counts[col] += 1
			correct_answers += 1
		else:
			incorrect_counts[col] += 1

	# If all LLM's answers are correct, record the index
	if correct_answers == len(LLM_COLS):
		consistent_indices.append(idx)

# Summarize the results
summary = ""
for col in LLM_COLS:
	summary += f"For {col}, there were {correct_counts[col]} correct answers and {incorrect_counts[col]} incorrect answers.\n"
summary += f"\nThere were {len(consistent_indices)} rows deleted, and {len(df_clean) - len(consistent_indices)} rows remaining.\n"
summary += f"The deleted rows' indices are: {consistent_indices}"

# Write summary to a text file
with open("../data/processed/mc/mc_summary.txt", "w") as text_file:
	text_file.write(summary)

# Remove rows where all LLM's answers are correct
df_filtered = df_clean.drop(consistent_indices)

# Save the filtered dataframe to a csv file
df_filtered.to_csv('../data/processed/mc/mc_answers_filtered.csv', index=False)
