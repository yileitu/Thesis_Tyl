# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd
from datasets import DatasetDict, load_dataset

ANSWER_MAP: Dict[int, str] = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

dataset: DatasetDict = load_dataset('openbookqa', 'main')
df = pd.DataFrame(dataset['train'])

df = df.rename(columns={'answerKey': 'answer', 'question_stem': 'question'})
unexpected_values = df.loc[~df['answer'].isin(['A', 'B', 'C', 'D']), 'answer'].unique()
if len(unexpected_values) > 0:
	print("... Found unexpected values in 'answer' column:", unexpected_values)
else:
	print("... No unexpected values found in 'answer' column.")

df['subject'] = None
df['passage'] = None
df[['option_A', 'option_B', 'option_C', 'option_D']] = pd.DataFrame(
	df['choices'].map(lambda x: x['text']).to_list(), index=df.index
	)
df = df[['passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'answer', 'subject']]
df.to_csv("../../data/sampled/exam/openbookqa.csv", index=False)
