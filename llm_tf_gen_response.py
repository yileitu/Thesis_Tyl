# -*- coding: utf-8 -*-
"""
Asks all LLMs to generate responses for the AlpacaEval dataset and calculates the BERTScore for each response.
"""
import gc

import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from util.constants import GEN_CONFIG_FOR_ALL_LLM
from util.util_func import find_first_unprocessed, gen_clean_output, gen_tf_templated_prompt, \
	get_llm_names_and_hf_paths, \
	set_mtec_env, set_seed

SAVE_INTERVAL: int = 10

# Set environments
NUM_GPU: int = 1
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
DF_PATH: str = "data/output/boolq.csv"
df = pd.read_csv(DF_PATH)
llm_name2hf_path, _, _, _ = get_llm_names_and_hf_paths()

for llm_name, llm_hf_path in tqdm(llm_name2hf_path.items()):
	output_col_name = f'response_{llm_name}'
	if output_col_name not in df.columns:
		df[output_col_name] = None

	# Load LLM
	tokenizer = AutoTokenizer.from_pretrained(llm_hf_path)
	model = AutoModelForCausalLM.from_pretrained(llm_hf_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
	print(f"... Loaded {llm_name}")
	model.to(device)
	model.eval()

	# Find the first row that has not been processed
	start_index = find_first_unprocessed(df=df, target_col_name=output_col_name)
	print(f"... Starting from index {start_index}")

	# Iterate through the rows and generate responses
	for idx, row in tqdm(df.iloc[start_index:].iterrows()):
		input_text = gen_tf_templated_prompt(passage=row['passage'], question=row['question'])

		# Generate response
		# Use autocast() to generate responses faster
		with autocast():
			input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
			with torch.no_grad():
				output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_ALL_LLM)
		output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		clean_output = gen_clean_output(output_text)
		df.loc[idx, output_col_name] = clean_output

		# Save the dataframe every SAVE_INTERVAL rows and clear memory
		if (idx + 1) % SAVE_INTERVAL == 0:
			df.to_csv(DF_PATH, index=False)
			del input_ids, output_ids, input_text, output_text, clean_output
			gc.collect()
			torch.cuda.empty_cache()

	# Save the remaining rows
	df.to_csv(DF_PATH, index=False)

	# Clear memory
	del model
	del tokenizer
	torch.cuda.empty_cache()
	gc.collect()

df.to_csv(DF_PATH, index=False)
