# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging

from util.constants import GEN_CONFIG_FOR_EXAM, GEN_CONFIG_FOR_QA
from util.util_func import find_first_unprocessed, gen_mc_templated_prompt, gen_qa_templated_prompt, gen_response_file, gen_tf_templated_prompt, set_gpu_env, set_seed, setup_signal_handlers, get_task_df_path
from util.struct import MCOptions, Task

# Constant Initialization
TASK = Task.QA
LLM_NAME: str = "T0"
LLM_HF_PATH: str = f"bigscience/{LLM_NAME}"
NUM_GPU: int = 1

# Set environments
set_seed()
device = set_gpu_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()


DF_PATH, RESPONSE_PATH = get_task_df_path(task=TASK, llm_name=LLM_NAME)
df = pd.read_csv(DF_PATH)
df = df.replace({np.nan: None})  # NaN is the default value when reading from CSV, replace it with None
output_col_name = f'response_{LLM_NAME}'
response_df = gen_response_file(response_df_path=RESPONSE_PATH, task_df=df, col_name=output_col_name)
setup_signal_handlers(df_to_save=response_df, save_path=RESPONSE_PATH)

# Find the first row that has not been processed
start_index = find_first_unprocessed(df=response_df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_HF_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_HF_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
print(f"... Loaded {LLM_NAME}")
model.to(device)
model.eval()

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	if TASK == Task.MC or TASK == Task.TOY_MC:
		options = MCOptions(
			A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
			)
		input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
	elif TASK == Task.TF:
		input_text = gen_tf_templated_prompt(passage=row['passage'], question=row['question'])
	elif TASK == Task.QA:
		input_text = gen_qa_templated_prompt(input_text=row['input'])
	else:
		raise ValueError(f"... Invalid task: {TASK}")

	# Generate response
	input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
	with torch.no_grad():
		if TASK == Task.QA:
			output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_QA)
		else:
			output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_EXAM)

	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	response_df.loc[idx, output_col_name] = output_text
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
