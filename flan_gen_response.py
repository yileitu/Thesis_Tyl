# -*- coding: utf-8 -*-
import gc

import pandas as pd
import torch
from bert_score import score
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, logging

from util.constants import GEN_CONFIG_FOR_ALL_LLM
from util.util_func import gen_clean_output_flan, gen_templated_prompt, set_mtec_env, set_seed

# Set environments
NUM_GPU: int = 1
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

llm_name = 'FLAN-T5-large'
llm_hf_path = 'google/flan-t5-large'

# Load the dataset
DF_PATH: str = "data/output/qa.csv"
df = pd.read_csv(DF_PATH)

tokenizer = T5Tokenizer.from_pretrained(llm_hf_path)
model = T5ForConditionalGeneration.from_pretrained(llm_hf_path, torch_dtype=torch.bfloat16)
print(f"Loaded {llm_name}")
model.to(device)
model.eval()

output_col_name = f'response_{llm_name}'
bert_score_col_name = f'BertScore_{llm_name}'
if output_col_name not in df.columns:
	df[output_col_name] = None
if bert_score_col_name not in df.columns:
	df[bert_score_col_name] = None

# Find the first row that has not been processed
start_index = df[bert_score_col_name].isna().idxmax()
print(f"... Starting from index {start_index}")

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	input_text = gen_templated_prompt(row['input'])

	# Generate response
	input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
	with torch.no_grad():
		output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_ALL_LLM)
	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	df.loc[idx, output_col_name] = output_text

	# Calculate BERTScore
	_, _, F1 = score([output_text], [row['response']], lang='en')
	df.loc[idx, bert_score_col_name] = F1.item()

	if idx % 100 == 0:
		df.to_csv(DF_PATH, index=False)

df.to_csv(DF_PATH, index=False)

# Clear memory
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
