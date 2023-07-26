# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Optional

import pandas as pd
from datasets import DatasetDict, load_dataset

ANSWER_MAP: Dict[int, str] = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
SUBJECTS: List[str] = [
	'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en', 'aqua-rat', 'sat-en-without-passage',
	'gaokao-english'
	]
# SUBJECTS: List[str] = ['lsat-ar', 'lsat-lr']


def remove_option_label(s: str) -> str:
	"""
	Remove option label (e.g. `(A)`) from the beginning of a string.
	:param s: original string
	:return: cleaned string
	"""
	if isinstance(s, str):
		return re.sub(r'^\([ABCDE]\)\s*', '', s)
	return s


def pad_options(option_list: List[str]) -> List[Optional[str]]:
	"""
	Pad option list to length 5.
	:param option_list: original option list
	:return: padded option list
	"""
	return option_list + [None] * (5 - len(option_list))


df = pd.DataFrame()
for subject in SUBJECTS:
	dataset: DatasetDict = load_dataset('v-xchen-v/agieval_eng_qa', subject)
	sub_df = pd.DataFrame(dataset['validation'])
	sub_df['subject'] = subject
	df = pd.concat([df, sub_df], ignore_index=True)

df.rename(columns={'label': 'answer'}, inplace=True)
unexpected_values = df.loc[~df['answer'].isin(list(ANSWER_MAP.keys())), 'answer'].unique()
if len(unexpected_values) > 0:
	print("... Found unexpected values in 'answer' column:", unexpected_values)
else:
	print("... No unexpected values found in 'answer' column.")

df['options'] = df['options'].apply(pad_options)
df[['option_A', 'option_B', 'option_C', 'option_D', 'option_E']] = pd.DataFrame(df.options.tolist(), index=df.index)
df['answer'] = df['answer'].map(ANSWER_MAP)
for col in ['option_A', 'option_B', 'option_C', 'option_D', 'option_E']:
	df[col] = df[col].apply(remove_option_label)
df = df[['passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'option_E', 'answer', 'subject']]
df.to_csv("../../data/sampled/exam/agieval.csv", index=False)
