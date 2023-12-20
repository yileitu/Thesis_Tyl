# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
from util.util_func import set_seed

# # Merge data
# df_mc_orig = pd.read_csv(
# 	'/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/mc/unfiltered/mc_unfiltered_answers.csv'
# 	)
# df_mc_pred = pd.read_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/mc_test_data.csv')
# df_merged = pd.merge(df_mc_orig, df_mc_pred.drop(columns=['filled_template']), on='id', how='inner')
#
#
# def determine_predicted_answer(row):
# 	label_to_answer = {
# 		'<|Simple|>'    : row['answer_pythia-2.8b'],
# 		'<|Unsolvable|>': row['answer_pythia-2.8b'],
# 		'<|Middle|>'    : row['answer_Llama-2-7b-chat'],
# 		'<|Arduous|>'   : row['answer_Llama-2-13b-chat'],
# 		'<|Difficult|>' : row['answer_gpt-3.5-turbo'],
# 		}
# 	return label_to_answer.get(row['predicted_label'], None)
#
#
# df_merged['predicted_answer'] = df_merged.apply(determine_predicted_answer, axis=1)
# df_merged.to_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/mc_test_merged.csv', index=False)

# # Read merged data

# Routing
df_mc_pred = pd.read_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/predicted/temp2.0/mc_test_merged.csv')
df_mc_pred['is_routing_correct'] = df_mc_pred['predicted_answer'] == df_mc_pred['answer_ground_truth']
overall_accuracy = df_mc_pred['is_routing_correct'].mean()
print("*** Routing ***")
print("整个数据集的准确率:", overall_accuracy)
accuracy_by_source = df_mc_pred.groupby('data_source')['is_routing_correct'].mean()
print("每个数据来源的准确率:\n", accuracy_by_source, "\n\n")

# pythia
df_mc_pred['is_pythia_correct'] = df_mc_pred['answer_pythia-2.8b'] == df_mc_pred['answer_ground_truth']
overall_accuracy = df_mc_pred['is_pythia_correct'].mean()
print("*** Pythia ***")
print("整个数据集的准确率:", overall_accuracy)
accuracy_by_source = df_mc_pred.groupby('data_source')['is_pythia_correct'].mean()
print("每个数据来源的准确率:\n", accuracy_by_source, "\n\n")

# Llama-2-7b-chat
df_mc_pred['is_llama_7b_correct'] = df_mc_pred['answer_Llama-2-7b-chat'] == df_mc_pred['answer_ground_truth']
overall_accuracy = df_mc_pred['is_llama_7b_correct'].mean()
print("*** Llama-2-7b-chat ***")
print("整个数据集的准确率:", overall_accuracy)
accuracy_by_source = df_mc_pred.groupby('data_source')['is_llama_7b_correct'].mean()
print("每个数据来源的准确率:\n", accuracy_by_source, "\n\n")

# Llama-2-13b-chat
df_mc_pred['is_llama_13b_correct'] = df_mc_pred['answer_Llama-2-13b-chat'] == df_mc_pred['answer_ground_truth']
overall_accuracy = df_mc_pred['is_llama_13b_correct'].mean()
print("*** Llama-2-13b-chat ***")
print("整个数据集的准确率:", overall_accuracy)
accuracy_by_source = df_mc_pred.groupby('data_source')['is_llama_13b_correct'].mean()
print("每个数据来源的准确率:\n", accuracy_by_source, "\n\n")

# GPT-3.5-turbo (BMA)
df_mc_pred['is_bma_correct'] = df_mc_pred['answer_gpt-3.5-turbo'] == df_mc_pred['answer_ground_truth']
overall_accuracy = df_mc_pred['is_bma_correct'].mean()
print("*** GPT-3.5 ***")
print("整个数据集的准确率:", overall_accuracy)
accuracy_by_source = df_mc_pred.groupby('data_source')['is_bma_correct'].mean()
print("每个数据来源的准确率:\n", accuracy_by_source, "\n\n")


# Majority
most_common_answers_by_source = df_mc_pred.groupby('data_source')['answer_ground_truth'].apply(
	lambda x: x.value_counts(normalize=True).idxmax()
	)
proportions_by_source = df_mc_pred.groupby('data_source')['answer_ground_truth'].apply(
	lambda x: x.value_counts(normalize=True).max()
	)
# 整个数据集中answer_ground_truth的最高占比答案及占比
most_common_answer_overall = df_mc_pred['answer_ground_truth'].value_counts(normalize=True).idxmax()
proportion_overall = df_mc_pred['answer_ground_truth'].value_counts(normalize=True).max()
print("*** Majority ***")
print("每个数据来源的最高占比答案及占比:\n", most_common_answers_by_source, "\n", proportions_by_source)
print("\n整个数据集的最高占比答案及占比:", most_common_answer_overall, proportion_overall, '\n\n')


# Random
random_seeds = [0, 1, 2, 3, 4]
answer_columns = ['answer_pythia-2.8b', 'answer_Llama-2-7b-chat', 'answer_Llama-2-13b-chat', 'answer_gpt-3.5-turbo']

# 初始化用于存储结果的字典
random_selection_counts = {col: [] for col in answer_columns}
accuracy_by_source = {}
overall_accuracy = []

# 对每个随机种子进行操作
for seed in random_seeds:
	set_seed(seed)
	# 随机选择列
	df_mc_pred['random_answer'] = df_mc_pred.apply(lambda row: row[random.choice(answer_columns)], axis=1)

	# 计算正确率
	df_mc_pred['is_correct'] = df_mc_pred['random_answer'] == df_mc_pred['answer_ground_truth']
	overall_accuracy.append(df_mc_pred['is_correct'].mean())
	accuracy_by_source[seed] = df_mc_pred.groupby('data_source')['is_correct'].mean()

	# # 计算随机选择的列的次数和占比
	# for col in answer_columns:
	# 	count = (df_mc_pred['random_answer'] == df_mc_pred[col]).sum()
	# 	random_selection_counts[col].append((count, count / len(df_mc_pred)))

# 计算整体平均正确率和每个数据源的平均正确率
mean_overall_accuracy = np.mean(overall_accuracy)
mean_accuracy_by_source = pd.DataFrame(accuracy_by_source).mean(axis=1)

# 输出结果
print("*** Random ***")
print("整个数据集的平均正确比例:", mean_overall_accuracy)
print("每个数据源的平均正确比例:\n", mean_accuracy_by_source)
# print("\n随机选择统计:")
# for col in random_selection_counts:
# 	counts, proportions = zip(*random_selection_counts[col])
# 	mean_count = np.mean(counts)
# 	mean_proportion = np.mean(proportions)
# 	print(f"{col}: 平均次数 = {mean_count}, 平均占比 = {mean_proportion}")