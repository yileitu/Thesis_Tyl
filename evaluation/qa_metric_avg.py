# -*- coding: utf-8 -*-
import pandas as pd

DATA_PATH = '../data/predicted/temp2.0/qa_test_merged.csv'
df = pd.read_csv(DATA_PATH, index_col='id')
results = pd.DataFrame(columns=['Column', 'Mean', 'Standard Deviation'])

for column in df.columns:
	# 跳过'id'列
	if column == 'id':
		continue

	# 检查列是否只包含数值
	if pd.api.types.is_numeric_dtype(df[column]):
		# 如果列名不包含“bart”，则将数值乘以100
		if 'bart' not in column:
			df[column] *= 100

		# 计算平均值和标准差
		mean = df[column].mean()
		std_dev = df[column].std()

		# 将结果添加到新的DataFrame
		new_row = pd.DataFrame({'Column': [column], 'Mean': [mean], 'Standard Deviation': [std_dev]})
		results = pd.concat([results, new_row], ignore_index=True)

results.to_csv('../data/predicted/temp2.0/qa_metric_avg_std.csv', index=False)
