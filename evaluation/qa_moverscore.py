# -*- coding: utf-8 -*-
import pandas as pd

DATA_PATH = '../data/predicted/temp2.0/qa_test_merged.csv'
df = pd.read_csv(DATA_PATH)

response_cols = ['response_pythia-2.8b', 'response_Llama-2-7b-chat', 'response_Llama-2-13b-chat',
                 'response_Llama-2-70b-chat']
moverscore_cols = ['response_pythia-2.8b_moverscore', 'response_Llama-2-7b-chat_moverscore',
                   'response_Llama-2-13b-chat_moverscore', 'response_Llama-2-70b-chat_moverscore']
df['random_moverscore'] = None

for index, row in df.iterrows():
	for i, response_col in enumerate(response_cols):
		# 检查 random_response 是否来自当前列
		if row['random_response'].strip() == row[response_col].strip():
			# 从对应的 moverscore 列获取值
			df.at[index, 'random_moverscore'] = row[moverscore_cols[i]]
			break

df.to_csv(DATA_PATH, index=False)
