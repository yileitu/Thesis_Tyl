# -*- coding: utf-8 -*-

import pandas as pd
from datasets import DatasetDict, load_dataset

dataset: DatasetDict = load_dataset('boolq')
df = pd.DataFrame(dataset['train'])
df = df[['passage', 'question', 'answer']]
df.to_csv("../../data/sampled/exam/boolq.csv", index=False)
