# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging

from util.constants import GEN_CONFIG_FOR_EXAM
from util.util_func import find_first_unprocessed, gen_mc_templated_prompt, gen_response_file, set_gpu_env, \
	set_seed, setup_signal_handlers
from util.struct import MCOptions

LLM_NAME: str = "T0_3B"
LLM_PATH = "bigscience/T0_3B"
NUM_GPU = 1
DF_PATH: str = "data/output/toy_mc.csv"
RESPONSE_PATH: str = f"data/output/toy_mc_response_{LLM_NAME}.csv"

df = pd.read_csv(DF_PATH)
df = df.replace({np.nan: None})  # NaN is the default value when reading from CSV, replace it with None
output_col_name = f'response_{LLM_NAME}'

response_df = gen_response_file(response_df_path=RESPONSE_PATH, task_df=df, col_name=output_col_name)
setup_signal_handlers(df_to_save=response_df, save_path=RESPONSE_PATH)

set_seed()
device = set_gpu_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_PATH)
print(f"... Loaded {LLM_NAME}")
model.to(device)
model.eval()

start_index = find_first_unprocessed(df=response_df, target_col_name=output_col_name)
print(f"... Starting from index {start_index}")

for idx, row in tqdm(df.iloc[start_index:].iterrows()):
	options = MCOptions(
		A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
		)
	input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
	input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
	with torch.no_grad():
		output_ids = model.generate(input_ids, generation_config=GEN_CONFIG_FOR_EXAM)
	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	response_df.loc[idx, output_col_name] = output_text
	response_df.to_csv(RESPONSE_PATH, index=False)

response_df.to_csv(RESPONSE_PATH, index=False)
