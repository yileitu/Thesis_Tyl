# -*- coding: utf-8 -*-
import os

import pandas as pd
from tqdm.auto import tqdm

from moverscore_expand import ref_score

#
# script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
# parent_dir = os.path.dirname(script_dir)  # 获取父目录
# sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

# Load the dataset
file_dir = '../data/processed/qa/without_chatgpt/sample/'
df = pd.read_csv(os.path.join(file_dir, 'qa_combined_nonempty.csv'))

# Calculate the scores
# for response_column in ['response_pythia-2.8b', 'response_Llama-2-7b-chat', 'response_Llama-2-13b-chat',
#                         'response_Llama-2-70b-chat']:
# 	scores = []
# 	for index, row in df.iterrows():
# 		sys = row['response_ground_truth']
# 		refs = [row[response_column]]
# 		score = ref_score(sys, refs)
# 		scores.append(score[0])  # Assuming each ref_score returns a list with a single score
# 	df[f'{response_column}_score'] = scores

tqdm.pandas(desc="Calculating Moverscores")
response_columns = ['response_pythia-2.8b', 'response_Llama-2-7b-chat', 'response_Llama-2-13b-chat',
                    'response_Llama-2-70b-chat']
df_scores = df.progress_apply(
	lambda row: ref_score(row['response_ground_truth'], [row[response_col] for response_col in response_columns]),
	axis=1,
	result_type='expand'
	)

# Assign the scores back to the main dataframe
for i, response_column in enumerate(response_columns):
	df[f'{response_column}_moverscore'] = df_scores[i]

df.to_csv(os.path.join(file_dir, 'qa_combined_nonempty_moverscore.csv'), index=False)
