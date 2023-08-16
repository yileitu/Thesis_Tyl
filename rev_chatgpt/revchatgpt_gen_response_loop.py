# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from rev_chatgpt.my_chatbot import MyChatbot
from util.constants import Task
from util.util_func import MCOptions, find_first_unprocessed, gen_mc_templated_prompt, gen_qa_templated_prompt, \
	setup_signal_handlers

TASK = Task.QA

# Load revChatGPT config
# REV_CHATGPT_CONFIG_PATH1 = 'rev_chatgpt_config_account_ETH.json'
REV_CHATGPT_CONFIG_PATH2 = 'rev_chatgpt_config_account_Google.json'
with open(REV_CHATGPT_CONFIG_PATH2) as tmp_f:
	tmp_rev_chatgpt_config = json.load(tmp_f)
LLM_NAME: str = tmp_rev_chatgpt_config['model']

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

# Initialize revChatGPT chatbot
MAX_ATTEMPTS = 100  # 总尝试次数，比如每个配置尝试2次，则总次数为4
WAITING_MINUTES = 60  # config_paths全部失败后等待的分钟数
config_paths = [REV_CHATGPT_CONFIG_PATH2]
current_config_index = 1
attempts = 0


def switch_config() -> Optional[MyChatbot]:
	global current_config_index
	current_config_index = (current_config_index + 1) % len(config_paths)
	print(f"... Switching to config: {config_paths[current_config_index]}")
	with open(config_paths[current_config_index]) as f:
		rev_chatgpt_config = json.load(f)
	# 尝试初始化MyChatbot并捕获任何可能的异常
	try:
		return MyChatbot(config=rev_chatgpt_config)
	except Exception as e:
		print(f"Error while switching to config {current_config_index}: {e}")
		return None


rev_chatgpt = switch_config()  # 初始配置

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

	while True:  # 循环，直到获得答案或所有配置都失败
		try:
			output_text = rev_chatgpt.get_response(input_text)
			response_df.loc[idx, output_col_name] = output_text
			response_df.to_csv(RESPONSE_PATH, index=False)
			break  # 成功，跳出循环
		except Exception as e:
			attempts += 1
			if attempts >= MAX_ATTEMPTS:
				print("... All attempts have been exhausted. Exiting...")
				exit(0)
			elif attempts % len(config_paths) == 0:
				# 如果每个配置都失败了一次，那么休息45分钟
				print(
					f"... Attempt {attempts} - Error with configuration {current_config_index}: {e}. Switching to next configuration..."
					)
				print(f"... All configurations have failed. Waiting for {WAITING_MINUTES} minutes before retrying...")
				time.sleep(WAITING_MINUTES * 60)  # 45分钟
				rev_chatgpt = switch_config()  # 切换配置
			else:
				print(
					f"... Attempt {attempts} - Error with configuration {current_config_index}: {e}. Switching to next configuration..."
					)
				rev_chatgpt = switch_config()  # 切换配置

			# 如果返回的是None，即尝试初始化MyChatbot失败，则直接进入下一轮循环
			if rev_chatgpt is None:
				continue

response_df.to_csv(RESPONSE_PATH, index=False)
