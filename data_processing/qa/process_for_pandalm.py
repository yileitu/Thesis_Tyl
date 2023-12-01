# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, List

import pandas as pd

include_chatgpt: bool = True
if include_chatgpt:
	qa_data_dir: str = '../../data/processed/qa/with_chatgpt/Sample1000'
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'text-davinci-002-render-sha']
else:
	qa_data_dir: str = '../data/processed/qa/without_chatgpt'
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat']

qa_data_path: str = os.path.join(qa_data_dir, 'qa_combined_nonempty.csv')
df_qa: pd.DataFrame = pd.read_csv(qa_data_path)

# Process the input for PandaLM
qa_pandalm_input_data: List[Dict[str, str]] = []  # Create an empty list to store the formatted data
for index, row in df_qa.iterrows():
	# Only fill the key 'instruction'. Leave the key 'input' empty. In accordance with the format of PandaLM.
	data_dict: Dict[str, str] = {"instruction": row['input'], "input": ""}
	qa_pandalm_input_data.append(data_dict)

qa_pandalm_input_data_json: str = json.dumps(qa_pandalm_input_data, ensure_ascii=False)
with open(os.path.join(qa_data_dir, "pandalm_qa_input.json"), 'w', encoding='utf-8') as f:
	f.write(qa_pandalm_input_data_json)


def df_column_to_json(df: pd.DataFrame, llm_name: str) -> None:
	"""
	Convert LLM responses from a dataframe column to a json file.

	:param df: QA input & responses dataframe
	:param llm_name: Name of the LLM
	"""
	col_name: str = f"response_{llm_name}"
	col_data = df[col_name].tolist()
	col_data_json: str = json.dumps(col_data, ensure_ascii=False)
	with open(os.path.join(qa_data_dir, f"pandalm_{llm_name}.json"), 'w', encoding='utf-8') as json_file:
		json_file.write(col_data_json)


# Process the LLM responses from PandaLM
for llm in llm_names:
	df_column_to_json(df=df_qa, llm_name=llm)
