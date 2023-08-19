# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, logging

from util.constants import GEN_CONFIG_FOR_EXAM, GEN_CONFIG_FOR_QA
from util.struct import MCOptions, Task
from util.util_func import find_first_unprocessed, gen_clean_output, gen_input_with_split, gen_mc_templated_prompt, \
	gen_qa_templated_prompt, gen_response_file, gen_tf_templated_prompt, get_task_df_path, set_mtec_env, set_seed, \
	setup_signal_handlers

# Constant Initialization
TASK = Task.MC
LLM_NAME: str = "Llama-2-13b-chat"
LLM_PATH: str = f"meta-llama/{LLM_NAME}-hf"
NUM_GPU: int = 1

# Set environments
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
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
tokenizer = LlamaTokenizer.from_pretrained(LLM_PATH, use_auth_token=True)
model = LlamaForCausalLM.from_pretrained(LLM_PATH, torch_dtype=torch.bfloat16, use_auth_token=True)
print(f"... Loaded {LLM_NAME}")

# Follow up HF tips https://huggingface.co/docs/transformers/model_doc/llama2
padding_token = "<pad>"
pad_token_id = tokenizer.add_special_tokens({"pad_token": padding_token})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = pad_token_id
model.config.pretraining_tp = 100

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
	input_text = gen_input_with_split(text=input_text)

	# Generate response
	with autocast():
		input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
		with torch.no_grad():
			if TASK == Task.QA:
				output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_QA)
			else:
				output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_EXAM)

	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	clean_output = gen_clean_output(output_text)
	response_df.loc[idx, output_col_name] = clean_output
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
