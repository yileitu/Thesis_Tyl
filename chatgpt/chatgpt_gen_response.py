# -*- coding: utf-8 -*-
"""
Use OpenAI ChatGPT API to generate responses for datasets of different tasks.
"""
import os

import openai
import pandas as pd
from tqdm import tqdm

from util.constants import CHATGPT_SYS_PROMPT, MAX_LEN_EXAM, TEMPERATURE, TOP_P
from util.util_func import find_first_unprocessed, gen_mc_templated_prompt, gen_tf_templated_prompt
from util.struct import MCOptions, Task

# ChatGPT
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_NAME: str = "gpt-3.5-turbo"
TASK = Task.MC

# Load the dataset
if TASK == Task.MC:
	DF_PATH: str = "/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/mc.csv"
	RESPONSE_PATH: str = f"/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/mc_response_{LLM_NAME}.csv"
elif TASK == Task.TF:
	DF_PATH: str = "/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/boolq.csv"
else:
	raise ValueError("Invalid task type")

df = pd.read_csv(DF_PATH)
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


def process_chatgpt_response() -> None:
	"""
	Get response using ChatGPT API and save the responses to the dataset.
	"""
	# Find the first row that has not been processed
	start_index = find_first_unprocessed(df=response_df, target_col_name=output_col_name)
	print(f"... Starting from index {start_index}")

	# Iterate through the rows and generate responses
	for idx, row in tqdm(df.iloc[start_index:].iterrows()):
		if TASK == Task.MC:
			options = MCOptions(
				A=row['option_A'], B=row['option_B'], C=row['option_C'], D=row['option_D'], E=row['option_E']
				)
			input_text = gen_mc_templated_prompt(passage=row['passage'], question=row['question'], options=options)
		elif TASK == Task.TF:
			input_text = gen_tf_templated_prompt(passage=row['passage'], question=row['question'])
		else:
			raise ValueError("Invalid task type")

		try:
			response = openai.ChatCompletion.create(
				model=LLM_NAME,
				messages=[
					{"role": "system", "content": CHATGPT_SYS_PROMPT},
					{"role": "user", "content": input_text},
					],
				temperature=TEMPERATURE,
				top_p=TOP_P,
				max_tokens=MAX_LEN_EXAM,
				)
			output_text = response["choices"][0]["message"]["content"]
			response_df.loc[idx, output_col_name] = output_text
			response_df.to_csv(RESPONSE_PATH, index=False)
		except openai.error.APIError as e:
			response_df.to_csv(RESPONSE_PATH, index=False)
			print(f"... Error: {e}")
			print(f"... Stopping at index {idx}")
			print("... Restarting from the beginning.")
			process_chatgpt_response()


process_chatgpt_response()
response_df.to_csv(RESPONSE_PATH, index=False)
