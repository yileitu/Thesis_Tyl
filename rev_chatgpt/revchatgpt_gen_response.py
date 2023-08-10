# -*- coding: utf-8 -*-
"""
Generate responses for multiple choice questions using revChatGPT.
"""
# # Make sure to run this script from the root directory correctly
# import os
# import sys
#
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import os

import pandas as pd
from tqdm import tqdm

from rev_chatgpt.my_chatbot import MyChatbot
from util.constants import Task
from util.util_func import MCOptions, find_first_unprocessed, gen_mc_templated_prompt, gen_qa_templated_prompt, \
	setup_signal_handlers

TASK = Task.QA

# Load revChatGPT config
REV_CHATGPT_CONFIG_PATH1 = 'rev_chatgpt_config_account_ETH.json'
REV_CHATGPT_CONFIG_PATH2 = 'rev_chatgpt_config_account_Google.json'
with open(REV_CHATGPT_CONFIG_PATH1) as f:
	rev_chatgpt_config = json.load(f)
LLM_NAME: str = rev_chatgpt_config['model']

# Load the dataset
if TASK == Task.MC:
	DF_PATH: str = "/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/mc.csv"
	RESPONSE_PATH: str = f"/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/mc_response_{LLM_NAME}.csv"
elif TASK == Task.QA:
	DF_PATH: str = "/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/qa.csv"
	RESPONSE_PATH: str = f"/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/qa_response_{LLM_NAME}.csv"
else:
	raise ValueError("Invalid task type")

df = pd.read_csv(DF_PATH)
num_rows = df.shape[0]
output_col_name = f'response_{LLM_NAME}'

# Check if the file exists, create it if not
if not os.path.exists(RESPONSE_PATH):
	response_df = pd.DataFrame(index=df.index, columns=[output_col_name])
	response_df.to_csv(RESPONSE_PATH, index=False)
	print(f"... Empty DataFrame saved to {RESPONSE_PATH}")
else:
	print(f"... File already exists: {RESPONSE_PATH}")
	response_df = pd.read_csv(RESPONSE_PATH)
	dummy_df = pd.DataFrame(index=df.index, columns=[output_col_name])
	dummy_df.update(response_df)
	dummy_df = dummy_df.where(pd.notnull(dummy_df), None)
	response_df = dummy_df

setup_signal_handlers(df_to_save=response_df, save_path=RESPONSE_PATH)

# Find the first row that has not been processed
start_index = find_first_unprocessed(df=response_df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

# Create revChatGPT chatbot
rev_chatgpt = MyChatbot(config=rev_chatgpt_config)

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	if TASK == Task.MC:
		options = MCOptions(
			A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
			)
		input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
	elif TASK == Task.QA:
		input_text = gen_qa_templated_prompt(row['input'])
	else:
		raise ValueError(f"... Invalid task: {TASK}")
	output_text = rev_chatgpt.get_response(input_text)
	response_df.loc[idx, output_col_name] = output_text
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
