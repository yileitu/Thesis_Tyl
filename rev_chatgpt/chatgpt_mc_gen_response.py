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

import pandas as pd
from tqdm import tqdm

from my_chatbot import MyChatbot
from util.util_func import MCOptions, find_first_unprocessed, gen_mc_templated_prompt

# Load revChatGPT config
REV_CHATGPT_CONFIG_PATH = 'rev_chatgpt_config.json'
with open(REV_CHATGPT_CONFIG_PATH, 'r') as f:
	rev_chatgpt_config = json.load(f)
LLM_NAME: str = rev_chatgpt_config['model']
SAVE_INTERVAL: int = 20

# Load the dataset
# DF_PATH: str = "data/output/toy_mc.csv"
DF_PATH: str = "/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/toy_mc.csv"
df = pd.read_csv(DF_PATH)
output_col_name = f'response_{LLM_NAME}'
if output_col_name not in df.columns:
	df[output_col_name] = None

# Find the first row that has not been processed
start_index = find_first_unprocessed(df=df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

# Create revChatGPT chatbot
rev_chatgpt = MyChatbot(config=rev_chatgpt_config)

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	options = MCOptions(
		A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
		)
	input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
	output_text = rev_chatgpt.get_response(input_text)
	df.loc[idx, output_col_name] = output_text

	# Save the dataframe every SAVE_INTERVAL rows and clear memory
	if (idx + 1) % SAVE_INTERVAL == 0:
		df.to_csv(DF_PATH, index=False)

df.to_csv(DF_PATH, index=False)
