# -*- coding: utf-8 -*-
import pandas as pd
from datasets import DatasetDict, load_dataset

SEED: int = 21946520

dataset: DatasetDict = load_dataset('tatsu-lab/alpaca_farm', 'alpaca_human_preference')
df = pd.DataFrame(dataset['preference'])

# 新增response列
df['response'] = df.apply(lambda row: row['output_1'] if row['preference'] == 1 else row['output_2'], axis=1)

# 合并instruction和input列
df['input'] = df.apply(
	lambda row: row['instruction'] if row['input'] == "" else row['instruction'] + "\n\n" + row['input'], axis=1
	)

# 删除无关的列
df = df.drop(columns=['instruction', 'output_1', 'output_2', 'preference', 'raw_preference'])

print(df.head())
df.to_csv('../../data/sampled/qa/alpacafarm.csv', index=False)
