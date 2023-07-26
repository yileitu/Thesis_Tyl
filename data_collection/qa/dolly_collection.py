# -*- coding: utf-8 -*-
import pandas as pd
from datasets import DatasetDict, load_dataset

dataset: DatasetDict = load_dataset('databricks/databricks-dolly-15k')
df = pd.DataFrame(dataset['train'])
df['input'] = df.apply(
	lambda row: row['instruction'] if row['context'] == "" else row['instruction'] + "\n\n" + row['context'], axis=1
	)
df = df.drop(columns=['instruction', 'context'])

df.to_csv('../../data/sampled/qa/dolly.csv', index=False)

source_counts = df['category'].value_counts()
for source, count in source_counts.items():
	print(f"Classification: {source}, Count: {count}")
