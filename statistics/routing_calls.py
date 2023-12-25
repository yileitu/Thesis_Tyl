# -*- coding: utf-8 -*-
import pandas as pd

DATA_PATH = '../data/predicted/temp2.0/mc_test_merged.csv'
df = pd.read_csv(DATA_PATH, index_col='id')
# 计算'label'列的值的个数及占比
label_counts = df['label'].value_counts()
label_percentages = df['label'].value_counts(normalize=True) * 100

# 计算'predicted_label'列的值的个数及占比
predicted_label_counts = df['predicted_label'].value_counts()
predicted_label_percentages = df['predicted_label'].value_counts(normalize=True) * 100

# 打印结果
print("Label Counts:\n", label_counts)
print("Label Percentages:\n", label_percentages)
print("Predicted Label Counts:\n", predicted_label_counts)
print("Predicted Label Percentages:\n", predicted_label_percentages)
