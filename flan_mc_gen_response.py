# -*- coding: utf-8 -*-
import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, logging

from util.constants import GEN_CONFIG_FOR_ALL_LLM
from util.util_func import MCOptions, find_first_unprocessed, gen_mc_templated_prompt, set_mtec_env, set_seed

SAVE_INTERVAL: int = 100

# Set environments
NUM_GPU: int = 1
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

llm_name = 'FLAN-T5-large'
llm_hf_path = 'google/flan-t5-large'

# Load the dataset
DF_PATH: str = "data/output/mc.csv"
df = pd.read_csv(DF_PATH)
df = df.replace({np.nan: None})  # NaN is the default value when reading from CSV, replace it with None

tokenizer = T5Tokenizer.from_pretrained(llm_hf_path)
model = T5ForConditionalGeneration.from_pretrained(llm_hf_path, torch_dtype=torch.bfloat16)
print(f"... Loaded {llm_name}")
model.to(device)
model.eval()

output_col_name = f'response_{llm_name}'
if output_col_name not in df.columns:
	df[output_col_name] = None

start_index = find_first_unprocessed(df=df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

# Iterate through the rows and generate responses
for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	options = MCOptions(
		A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
		)
	input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)

	# Generate response
	input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
	with torch.no_grad():
		output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_ALL_LLM)
	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	df.loc[idx, output_col_name] = output_text

	# Save the dataframe every SAVE_INTERVAL rows and clear memory
	if (idx + 1) % SAVE_INTERVAL == 0:
		df.to_csv(DF_PATH, index=False)
		del input_ids, output_ids, input_text, output_text
		gc.collect()
		torch.cuda.empty_cache()

# Save the remaining rows
df.to_csv(DF_PATH, index=False)

# Clear memory
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
