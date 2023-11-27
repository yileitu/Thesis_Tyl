# -*- coding: utf-8 -*-
import os
from typing import List

import pandas as pd

from util.struct import Task
from util.util_func import save_df_to_csv, truncate_after_first_all_nan

TASK: Task = Task.QA
TASK_NAME: str = 'qa'
TASK_DATA_PATH: str = f'../../data/output/{TASK_NAME}/{TASK_NAME}.csv'

include_chatgpt: bool = True
if include_chatgpt:
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat',
	                        'text-davinci-002-render-sha']
	save_dir = f"../../data/processed/{TASK_NAME}/with_chatgpt"
else:
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat']
	save_dir = f"../../data/processed/{TASK_NAME}/without_chatgpt"

# Read the task data
df_task = pd.read_csv(TASK_DATA_PATH)
if 'id' not in df_task.columns:
	df_task.reset_index(level=0, inplace=True)
	df_task.rename(columns={'index': 'id'}, inplace=True)
df_task.rename(columns={'response': 'response_ground_truth'}, inplace=True)
df_task.drop(columns=['response_Vicuna-13b'], inplace=True)

# Merge the responses of LLMs with the task data
df_merged = df_task.copy()
for llm_name in llm_names:
	llm_response_path = f'../../data/output/{TASK_NAME}/{TASK_NAME}_response_{llm_name}.csv'
	df_llm_response = pd.read_csv(llm_response_path)
	df_llm_response = truncate_after_first_all_nan(df_llm_response)
	print(f"Shape of {llm_name}: {df_llm_response.shape}")
	df_merged = pd.concat([df_merged, df_llm_response], axis=1)
	min_length = min(len(df_task), len(df_llm_response))
	df_merged = df_merged.iloc[:min_length]
	print(f"Shape of merged dataframe: {df_merged.shape}")

df_merged_name = f'{TASK_NAME}_combined.csv'
df_merged_path = os.path.join(save_dir, df_merged_name)
save_df_to_csv(df=df_merged, path=df_merged_path, index=False)

# Get NaN info and drop NaN
response_col_names = [f'response_{llm_name}' for llm_name in llm_names]
missing_values_info = ""
for col in response_col_names:
	# Check and record missing values for each LLM column
	missing_values = df_merged[col].isna()
	missing_indices = missing_values.where(missing_values).dropna().index.tolist()
	missing_values_info += f"Column {col} has {len(missing_indices)} missing values at indices {missing_indices}\n"

txt_name = f"{TASK_NAME}_missing_values_info.txt"
txt_path = os.path.join(save_dir, txt_name)
with open(txt_path, "w") as text_file:
	# Write missing values info to a text file
	text_file.write(missing_values_info)

df_merged_nonempty = df_merged.dropna(subset=response_col_names)
df_merged_nonempty_name = f'{TASK_NAME}_combined_nonempty.csv'
df_merged_nonempty_path = os.path.join(save_dir, df_merged_nonempty_name)
save_df_to_csv(df=df_merged_nonempty, path=df_merged_nonempty_path, index=False)
