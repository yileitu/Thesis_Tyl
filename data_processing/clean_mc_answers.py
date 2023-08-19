# -*- coding: utf-8 -*-
import re
from typing import List, Optional

import pandas as pd


def extract_letter(response: str, options: str, option_texts: List[str]) -> Optional[str]:
	"""
	Extract the answer letter from the response.
	:param response: response from the LLM
	:param options: options for the multiple choice question, 'ABCDE' or 'ABCD'
	:param option_texts: list of option texts
	:return:
	"""
	if response is None or pd.isna(response):
		return None
	# First search for the option letter
	match = re.search(fr'\b[{options}]\b', response, re.IGNORECASE)
	if match:
		return match.group(0).upper()
	else:
		# If no option letter is found, search for the option text
		for option, option_text in zip(options, option_texts):
			if pd.isna(option_text):
				continue
			if option_text in response:
				return option
		return None


COLS_TO_CLEAN:List[str] = ['response_FLAN-T5-large', 'response_Vicuna-13b', 'response_gpt-3.5-turbo']

df = pd.read_csv('../data/output/mc.csv')
df.reset_index(level=0, inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)
df.to_csv('../data/output/mc.csv', index=False)

# Extract the answers and form the new DataFrame
df_answers = df[['id', 'answer']].copy()

# rename the column
df_answers.rename(columns={'answer': 'answer_ground_truth'}, inplace=True)

# Generate new answer columns for each LLM model
for col in ['response_FLAN-T5-large', 'response_Vicuna-13b', 'response_gpt-3.5-turbo']:
	df_answers[f'{col.replace("response_", "answer_")}'] = df.apply(
		lambda row: extract_letter(
			response=row[col],
			options='ABCDE' if pd.notna(row['option_E']) else 'ABCD',
			option_texts=[row['option_A'], row['option_B'], row['option_C'], row['option_D'],
			              row['option_E'] if pd.notna(row['option_E']) else None]
			),
		axis=1
		)

missing_values_info = ""
# Check and record missing values for each LLM column
for col in ['answer_FLAN-T5-large', 'answer_Vicuna-13b', 'answer_gpt-3.5-turbo']:
	missing_values = df_answers[col].isna()
	missing_indices = missing_values.where(missing_values).dropna().index.tolist()
	missing_values_info += f"Column {col} has {len(missing_indices)} missing values at indices {missing_indices}\n"

# Write missing values info to a text file
with open("../data/processed/mc/mc_missing_values_info.txt", "w") as text_file:
	text_file.write(missing_values_info)

# Drop rows with any missing values
df_clean = df_answers.dropna()

# Save the cleaned dataframe to a csv file
df_clean.to_csv('../data/processed/mc/mc_answers.csv', index=False)
