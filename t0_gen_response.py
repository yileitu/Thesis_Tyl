# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging

from util.constants import GEN_CONFIG_FOR_EXAM, Task
from util.util_func import MCOptions, find_first_unprocessed, gen_mc_templated_prompt, gen_qa_templated_prompt, \
	gen_tf_templated_prompt, set_mtec_env, set_seed, setup_signal_handlers

# Constant Initialization
TASK = Task.QA
SAVE_INTERVAL: int = 20
NUM_GPU: int = 1
LLM_NAME: str = "T0"

# Set environments
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
if TASK == Task.MC:
	DF_PATH: str = "data/output/mc.csv"
	RESPONSE_PATH: str = f"data/output/mc_response_{LLM_NAME}.csv"
elif TASK == Task.TF:
	DF_PATH: str = "data/output/boolq.csv"
	RESPONSE_PATH: str = f"data/output/tf_response_{LLM_NAME}.csv"
elif TASK == Task.QA:
	DF_PATH: str = "data/output/qa.csv"
	RESPONSE_PATH: str = f"data/output/qa_response_{LLM_NAME}.csv"
else:
	raise ValueError("Invalid task type")

df = pd.read_csv(DF_PATH)
df = df.replace({np.nan: None})  # NaN is the default value when reading from CSV, replace it with None
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

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", torch_dtype=torch.bfloat16, trust_remote_code=True)
print(f"... Loaded {LLM_NAME}")
model.to(device)
model.eval()

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	if TASK == Task.MC:
		options = MCOptions(
			A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
			)
		input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
	elif TASK == Task.TF:
		input_text = gen_tf_templated_prompt(passage=row['passage'], question=row['question'])
	elif TASK == Task.QA:
		input_text = gen_qa_templated_prompt(row['input'])
	else:
		raise ValueError(f"... Invalid task: {TASK}")

	# Generate response
	# Use autocast() to generate responses faster
	with autocast():
		input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
		with torch.no_grad():
			output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_EXAM)
	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	response_df.loc[idx, output_col_name] = output_text
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
