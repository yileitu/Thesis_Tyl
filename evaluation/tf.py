# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd

# Merge data
df_tf_orig = pd.read_csv(
	'/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/tf/unfiltered/labeled/tf_labeled_answers.csv'
	)
df_tf_pred = pd.read_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/tf_test_data.csv')
df_merged = pd.merge(df_tf_orig, df_tf_pred.drop(columns=['filled_template']), on='id', how='inner')


def determine_predicted_answer(row):
	label_to_answer = {
		'<|Simple|>'    : row['answer_pythia-2.8b'],
		'<|Unsolvable|>': row['answer_pythia-2.8b'],
		'<|Middle|>'    : row['answer_Llama-2-7b-chat'],
		'<|Difficult|>' : row['answer_Llama-2-13b-chat'],
		'<|Arduous|>'   : row['answer_gpt-3.5-turbo'],
		}
	return label_to_answer.get(row['predicted_label'], None)


def determine_oracle_answer(row):
	label_to_answer = {
		'<|Simple|>'    : row['answer_pythia-2.8b'],
		'<|Unsolvable|>': row['answer_pythia-2.8b'],
		'<|Middle|>'    : row['answer_Llama-2-7b-chat'],
		'<|Difficult|>' : row['answer_Llama-2-13b-chat'],
		'<|Arduous|>'   : row['answer_gpt-3.5-turbo'],
		}
	return label_to_answer.get(row['label_y'], None)


df_merged['predicted_answer'] = df_merged.apply(determine_predicted_answer, axis=1)
df_merged['oracle_answer'] = df_merged.apply(determine_oracle_answer, axis=1)
df_merged.to_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/tf_test_merged.csv', index=False)

# # Read merged data
df_tf_pred = pd.read_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/tf_test_merged.csv')

# Oracle
df_tf_pred['is_oracle_correct'] = df_tf_pred['oracle_answer'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_oracle_correct'].mean()
print("*** Oracle ***")
print("整个数据集的准确率:", overall_accuracy)

# Routing
df_tf_pred['is_routing_correct'] = df_tf_pred['predicted_answer'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_routing_correct'].mean()
print("*** Routing ***")
print("整个数据集的准确率:", overall_accuracy)

# pythia
df_tf_pred['is_pythia_correct'] = df_tf_pred['answer_pythia-2.8b'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_pythia_correct'].mean()
print("*** Pythia ***")
print("整个数据集的准确率:", overall_accuracy)

# Llama-2-7b-chat
df_tf_pred['is_llama_7b_correct'] = df_tf_pred['answer_Llama-2-7b-chat'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_llama_7b_correct'].mean()
print("*** Llama-2-7b-chat ***")
print("整个数据集的准确率:", overall_accuracy)

# Llama-2-13b-chat
df_tf_pred['is_llama_13b_correct'] = df_tf_pred['answer_Llama-2-13b-chat'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_llama_13b_correct'].mean()
print("*** Llama-2-13b-chat ***")
print("整个数据集的准确率:", overall_accuracy)

# GPT-3.5-turbo
df_tf_pred['is_bma_correct'] = df_tf_pred['answer_gpt-3.5-turbo'] == df_tf_pred['answer_ground_truth']
overall_accuracy = df_tf_pred['is_bma_correct'].mean()
print("*** GPT-3.5 ***")
print("整个数据集的准确率:", overall_accuracy)

# Majority
most_common_answer_overall = df_tf_pred['answer_ground_truth'].value_counts(normalize=True).idxmax()
proportion_overall = df_tf_pred['answer_ground_truth'].value_counts(normalize=True).max()
print("*** Majority ***")
print("整个数据集的最高占比答案及占比:", most_common_answer_overall, proportion_overall, '\n\n')

# Random
random_seeds = [0, 1, 2, 3, 4]
answer_columns = ['answer_pythia-2.8b', 'answer_Llama-2-7b-chat', 'answer_Llama-2-13b-chat', 'answer_gpt-3.5-turbo']

# 初始化用于存储结果的字典
random_selection_counts = {col: [] for col in answer_columns}
accuracy_by_source = {}
overall_accuracy = []

# 对每个随机种子进行操作
for seed in random_seeds:
	random.seed(seed)
	# 随机选择列
	df_tf_pred['random_answer'] = df_tf_pred.apply(lambda row: row[random.choice(answer_columns)], axis=1)

	# 计算正确率
	df_tf_pred['is_correct'] = df_tf_pred['random_answer'] == df_tf_pred['answer_ground_truth']
	overall_accuracy.append(df_tf_pred['is_correct'].mean())

# # 计算随机选择的列的次数和占比
# for col in answer_columns:
# 	count = (df_mc_pred['random_answer'] == df_mc_pred[col]).sum()
# 	random_selection_counts[col].append((count, count / len(df_mc_pred)))

# 计算整体平均正确率和每个数据源的平均正确率
mean_overall_accuracy = np.mean(overall_accuracy)

# 输出结果
print("*** Random ***")
print("整个数据集的平均正确比例:", mean_overall_accuracy)
# print("\n随机选择统计:")
# for col in random_selection_counts:
# 	counts, proportions = zip(*random_selection_counts[col])
# 	mean_count = np.mean(counts)
# 	mean_proportion = np.mean(proportions)
# 	print(f"{col}: 平均次数 = {mean_count}, 平均占比 = {mean_proportion}")
