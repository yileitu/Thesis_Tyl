# -*- coding: utf-8 -*-
from datasets import load_dataset, Dataset

# 加载数据集
dataset = load_dataset('gsm8k', 'main')

# 随机打乱训练集
dataset = dataset['train'].shuffle(seed=42)

# 抽取前100个样本
subset = dataset[:100]

subset_dataset = Dataset.from_dict(subset)
subset_dataset.to_csv('gsm8k.csv')
