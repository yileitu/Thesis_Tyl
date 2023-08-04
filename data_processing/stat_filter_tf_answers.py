# -*- coding: utf-8 -*-
import pandas as pd

LLMS = ['FLAN-T5-large', 'Vicuna-13b', 'gpt-3.5-turbo']

df_answers_clean = pd.read_csv('../data/processed/tf/tf_answers.csv')
df_answers_clean.set_index('id', inplace=True)

#
# # Initialize counts
# correct_counts = {
# 	'answer_FLAN-T5-large': 0,
# 	'answer_Vicuna-13b'   : 0,
# 	'answer_gpt-3.5-turbo': 0
# 	}
# incorrect_counts = {
# 	'answer_FLAN-T5-large': 0,
# 	'answer_Vicuna-13b'   : 0,
# 	'answer_gpt-3.5-turbo': 0
# 	}
consistent_indices = []
# Iterate over rows in the DataFrame
for idx, row in df_answers_clean.iterrows():
	correct_answers = 0
	# Compare each LLM's answer to the ground truth
	for llm in LLMS:
		col = f'answer_{llm}'
		if row[col] == row['answer_ground_truth']:
			correct_answers += 1

	# If all LLM's answers are correct, record the index
	if correct_answers == len(LLMS):
		consistent_indices.append(idx)

# 正确和错误的答案统计
correct_counts = {f'answer_{llm}': (df_answers_clean[f'answer_{llm}'] == df_answers_clean['answer_ground_truth']).sum()
                  for llm in LLMS}
total_counts = {f'answer_{llm}': len(df_answers_clean) for llm in LLMS}
accuracy_rates = {f'answer_{llm}': correct_counts[f'answer_{llm}'] / total_counts[f'answer_{llm}'] for llm in LLMS}

# 保存统计结果
with open("../data/processed/tf/tf_summary.txt", 'w') as file:
	file.write('Number of correct answers, wrong answers and accuracy rate for each LLM:\n')
	for llm in LLMS:
		file.write(
			f'{llm}: correct - {correct_counts[f"answer_{llm}"]}, total - {total_counts[f"answer_{llm}"]}, '
			f'accuracy rate - {accuracy_rates[f"answer_{llm}"] * 100:.2f}%\n'
			)
	file.write(f"\nThere were {len(consistent_indices)} rows deleted, and {len(df_answers_clean) - len(consistent_indices)} rows remaining.\n")
	file.write(f"The deleted rows' indices are: {consistent_indices}")

# Remove rows where all LLM's answers are correct
df_filtered = df_answers_clean.drop(consistent_indices)
df_filtered.to_csv('../data/processed/tf/tf_answers_filtered.csv', index=False)
