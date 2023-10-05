# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, logging

from util.constants import MAX_LEN_QA, TEMPERATURE, TOP_P
from util.struct import MCOptions, Task
from util.util_func import find_first_unprocessed, gen_clean_output, gen_input_with_split, gen_mc_templated_prompt, \
	gen_qa_templated_prompt, gen_response_file, gen_tf_templated_prompt, get_task_df_path, set_gpu_env, \
	set_llama_config, set_seed, setup_signal_handlers

# Constant Initialization
TASK = Task.QA
LLM_PARAM: int = 7  # Choose from [7, 13, 70]
LLM_NAME: str = f"Llama-2-{LLM_PARAM}b-chat"
# LLM_NAME: str = f"Llama-2-{LLM_PARAM}b"
NUM_GPU: int = 1

# Set environments
set_seed()
device = set_gpu_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
DF_PATH, RESPONSE_PATH = get_task_df_path(task=TASK, llm_name=LLM_NAME)
df = pd.read_csv(DF_PATH)
# df = read_csv_with_bad_lines(DF_PATH)
df = df.replace({np.nan: None})  # NaN is the default value when reading from CSV, replace it with None
output_col_name = f'response_{LLM_NAME}'
response_df = gen_response_file(response_df_path=RESPONSE_PATH, task_df=df, col_name=output_col_name)
setup_signal_handlers(df_to_save=response_df, save_path=RESPONSE_PATH)

# Find the first row that has not been processed
start_index = find_first_unprocessed(df=response_df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

# Load LLM
if LLM_PARAM == 70:
	LLM_HF_PATH: str = "TheBloke/Llama-2-70B-chat-GPTQ"
	REVISION = "gptq-4bit-128g-actorder_True"
	tokenizer = AutoTokenizer.from_pretrained(LLM_HF_PATH, use_fast=True)
elif LLM_PARAM == 13 or LLM_PARAM == 7:
	LLM_HF_PATH: str = f"meta-llama/{LLM_NAME}-hf"
	tokenizer = LlamaTokenizer.from_pretrained(LLM_HF_PATH)
else:
	raise ValueError(f"... Invalid number of parameters of Llama: {LLM_PARAM}")
tokenizer.pad_token_id = tokenizer.eos_token_id  # for open-ended generation

if LLM_PARAM == 70:
	# Use 4-bit quantization for Llama-2-70b
	model = AutoModelForCausalLM.from_pretrained(
		LLM_HF_PATH,
		revision=REVISION,
		trust_remote_code=False,
		torch_dtype=torch.float16,  # Do not support bfloat16
		device_map="auto",
		)
elif LLM_PARAM == 13 or LLM_PARAM == 7:
	model = LlamaForCausalLM.from_pretrained(
		LLM_HF_PATH,
		torch_dtype=torch.bfloat16,
		)
else:
	raise ValueError(f"... Invalid number of parameters of Llama: {LLM_PARAM}")

print(f"... Loaded {LLM_NAME}")
if LLM_PARAM == 70:
	print(f"... Revision: {REVISION}")

if LLM_PARAM == 7 or LLM_PARAM == 13:
	gen_config = set_llama_config(model=model, tokenizer=tokenizer, device=device, task=TASK)
	print(gen_config)

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
	input_text = gen_input_with_split(text=input_text, task=TASK, llm_name=LLM_NAME)

	# Generate response
	if LLM_PARAM == 70:
		input_ids = tokenizer(input_text, return_tensors='pt', truncation=True).input_ids.to(device)
		output_ids = model.generate(
			inputs=input_ids,
			max_new_tokens=MAX_LEN_QA,
			temperature=TEMPERATURE,
			do_sample=True,
			top_p=TOP_P
			)
	elif LLM_PARAM == 13 or LLM_PARAM == 7:
		input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
		with torch.no_grad():
			output_ids = model.generate(input_ids, generation_config=gen_config)
	else:
		raise ValueError(f"... Invalid number of parameters of Llama: {LLM_PARAM}")

	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	clean_output = gen_clean_output(output_text=output_text, task=TASK, llm_name=LLM_NAME)
	response_df.loc[idx, output_col_name] = clean_output
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
