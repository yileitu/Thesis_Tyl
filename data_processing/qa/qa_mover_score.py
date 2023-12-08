# -*- coding: utf-8 -*-
import os
import sys

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

import pandas as pd

from moverscore.examples.example import ref_score

# Load the dataset
file_dir = '../../data/processed/qa/without_chatgpt/sample/'
df = pd.read_csv(os.path.join(file_dir, 'qa_combined_nonempty.csv'))

# Calculate the scores
for response_column in ['response_pythia-2.8b', 'response_Llama-2-7b-chat', 'response_Llama-2-13b-chat',
                        'response_Llama-2-70b-chat']:
	scores = []
	for index, row in df.iterrows():
		sys = row['response_ground_truth']
		refs = [row[response_column]]
		score = ref_score(sys, refs)
		scores.append(score[0])  # Assuming each ref_score returns a list with a single score
	df[f'{response_column}_score'] = scores

df.to_csv(os.path.join(file_dir, 'qa_combined_nonempty_moverscore.csv'), index=False)
