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

from util.constants import GEN_CONFIG_FOR_QA, RESPONSE_SPLIT
from util.util_func import find_first_unprocessed, gen_clean_output, gen_qa_templated_prompt, \
	get_llm_names_and_hf_paths, set_gpu_env, set_seed

SAVE_INTERVAL: int = 20

# Set environments
NUM_GPU: int = 1
set_seed()
device = set_gpu_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
DF_PATH: str = "data/output/qa.csv"
df = pd.read_csv(DF_PATH)
llm_name2hf_path, _, _, _ = get_llm_names_and_hf_paths()

for llm_name, llm_hf_path in tqdm(llm_name2hf_path.items()):
	output_col_name = f'response_{llm_name}'
	# bert_score_col_name = f'BertScore_{llm_name}'
	if output_col_name not in df.columns:
		df[output_col_name] = None
	# if bert_score_col_name not in df.columns:
	# 	df[bert_score_col_name] = None

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
		# # Check the condition: if category is "pacovaldez/stackoverflow-questions" and data_source is "GPT4All"
		# category = row['category']
		# data_source = row['data_source']
		# if category == "pacovaldez/stackoverflow-questions" and data_source == "GPT4All":
		# 	# Skip the current row and continue to the next iteration
		# 	continue

		input_text = gen_qa_templated_prompt(row['input'])
		input_text += "\n\n" + RESPONSE_SPLIT

		# Generate response
		# Use autocast() to generate responses faster
		with autocast():
			input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
			with torch.no_grad():
				output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_QA)
		output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		clean_output = gen_clean_output(output_text)
		df.loc[idx, output_col_name] = clean_output

		# # Calculate BERTScore
		# _, _, F1 = score([clean_output], [row['response']], lang='en')
		# df.loc[idx, bert_score_col_name] = F1.item()

		# Save the dataframe every SAVE_INTERVAL rows and clear memory
		if (idx + 1) % SAVE_INTERVAL == 0:
			torch.cuda.empty_cache()
			del input_ids, output_ids, output_text, clean_output
			gc.collect()
			df.to_csv(DF_PATH, index=False)

	# Save the remaining rows
	df.to_csv(DF_PATH, index=False)
