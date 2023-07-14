# -*- coding: utf-8 -*-
"""
Asks all LLMs to generate responses for the AlpacaEval dataset and calculates the BERTScore for each response.
"""
import gc

import pandas as pd
import torch
from bert_score import score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from util.constants import GEN_CONFIG_FOR_ALL_LLM
from util.util_func import gen_templated_prompt, get_llm_names_and_hf_paths, set_mtec_env, set_seed, gen_clean_output

# Set environments
NUM_GPU: int = 4
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

# Load the dataset
DF_PATH: str = "data/AlpacaEval.csv"
df = pd.read_csv(DF_PATH)
llm_name2hf_path, _, _, _ = get_llm_names_and_hf_paths()

for llm_name, llm_hf_path in tqdm(llm_name2hf_path.items()):
	output_col_name = f'output-{llm_name}'
	bert_score_col_name = f'bert_score-{llm_name}'
	df[output_col_name] = None
	df[bert_score_col_name] = None

	# Load LLM
	tokenizer = AutoTokenizer.from_pretrained(llm_hf_path)
	model = AutoModelForCausalLM.from_pretrained(llm_hf_path, torch_dtype=torch.float16)
	print(f"Loaded {llm_name}")
	model.to(device)
	model.eval()

	# Find the first row that has not been processed
	start_index = df[bert_score_col_name].isna().idxmax()

	# Iterate through the rows and generate responses
	for idx, row in tqdm(df.iloc[start_index:].iterrows()):
		input_text = gen_templated_prompt(row['input'])

		# Generate response
		input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
		with torch.no_grad():
			output_ids = model.generate(
				input_ids,
				generation_config=GEN_CONFIG_FOR_ALL_LLM,
				)
		output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		clean_output = gen_clean_output(output_text)
		df.loc[idx, output_col_name] = output_text

		# Calculate BERTScore
		_, _, F1 = score([output_text], [row['output-text_davinci_003']], lang='en')
		df.loc[idx, bert_score_col_name] = F1.item()

		# Save the dataframe every 10 rows
		if idx % 10 == 0:
			df.to_csv(DF_PATH, index=False)

	df.to_csv(DF_PATH, index=False)

	# Clear memory
	del model
	del tokenizer
	gc.collect()
	torch.cuda.empty_cache()

df.to_csv(DF_PATH, index=False)
