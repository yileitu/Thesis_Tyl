# -*- coding: utf-8 -*-
import random
import re
from typing import List, Optional

import pandas as pd
from fuzzywuzzy import process
from transformers import BertForSequenceClassification, BertTokenizer
import torch

from constants_for_clean import LLM_NAMES
from util.struct import Task
from util.constants import NULL_VALUES
from util.util_func import set_seed

TASK: Task = Task.TF
NEGATIVES = ["not", "no", "neither", "nor", "never", "none", "without", "hardly", "scarcely", "barely"]
set_seed()

if TASK == Task.TF:
	sentimental_tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
	sentimental_model = BertForSequenceClassification.from_pretrained(
		'nlptown/bert-base-multilingual-uncased-sentiment'
		)


def extract_letter(response: str, options: str, option_texts: List[str], fuzziness_threshold: int = 80) \
		-> Optional[str]:
	"""
	Extract the answer letter from the response.
	:param response: response from the LLM
	:param options: options for the multiple choice question, 'ABCDE' or 'ABCD'
	:param option_texts: list of option texts
	:param fuzziness_threshold: threshold for fuzzy matching
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
		best_match = process.extractOne(response, option_texts, score_cutoff=fuzziness_threshold)
		if best_match:
			index = option_texts.index(best_match[0])
			return options[index]
		return None


def extract_tf(response: Optional[str]) -> Optional[str]:
	"""
	Extracts TRUE/FALSE from a response based on various conditions
	:param response: response from the LLM
	:return: TRUE, FALSE, or None
	"""
	response = str(response).strip()

	# Check for null or special values
	if response in NULL_VALUES or all(c in '?!., ' for c in response) or response == "Thank you!":
		return None

	# Check for the special case "YES or NO"
	if response.lower() == "yes or no":
		return random.choice(["TRUE", "FALSE"])

	# Replace 'True' with 'yes' and 'False' with 'no', case insensitive
	response = re.sub(r'\btrue\b', 'yes', response, flags=re.IGNORECASE)
	response = re.sub(r'\bfalse\b', 'no', response, flags=re.IGNORECASE)

	# Count 'yes' and 'no' occurrences
	count_yes = len(re.findall(r'\byes\b', response, re.IGNORECASE))
	count_no = len(re.findall(r'\bno\b', response, re.IGNORECASE))

	if count_yes > count_no:
		return "TRUE"
	elif count_no > count_yes:
		return "FALSE"

	# Check for negative words
	if any(re.search(r'\b' + word + r'\b', response, re.IGNORECASE) for word in NEGATIVES):
		return "FALSE"

	# If count_yes and count_no are equal and greater than 0
	if count_yes == count_no and count_yes > 0:
		return random.choice(["TRUE", "FALSE"])

	# If none of the above conditions are met, use the sentimental model for classification
	inputs = sentimental_tokenizer.encode_plus(response, return_tensors="pt", max_length=512, truncation=True)

	with torch.no_grad():
		outputs = sentimental_model(**inputs)
		logits = outputs[0]
		predicted_class = torch.argmax(logits, dim=1).item()

	if predicted_class in [1, 2]:
		return "FALSE"
	else:
		return "TRUE"


if TASK == Task.MC:
	TASK_NAME = 'mc'
elif TASK == Task.TF:
	TASK_NAME = 'tf'
else:
	raise NotImplementedError(f"... Task {TASK} not implemented yet!")
TASK_DATA_PATH: str = f'../data/output/{TASK_NAME}/{TASK_NAME}.csv'
FUZZY_THRESHOLD: int = 50  # Threshold for fuzzy matching

df_task = pd.read_csv(TASK_DATA_PATH)
if 'id' not in df_task.columns:
	df_task.reset_index(level=0, inplace=True)
	df_task.rename(columns={'index': 'id'}, inplace=True)

if TASK == Task.MC:
	df_answers = df_task[
		['id', 'answer', 'data_source', 'subject']].copy()  # Extract the answers and form the new DataFrame
elif TASK == Task.TF:
	df_answers = df_task[['id', 'answer']].copy()
df_answers.rename(columns={'answer': 'answer_ground_truth'}, inplace=True)

dfs_to_merge = [df_task]
for llm_name in LLM_NAMES:
	if llm_name == 'Vicuna-13b' or llm_name == 'gpt-3.5-turbo':
		continue
	llm_response_path = f'../data/output/{TASK_NAME}/{TASK_NAME}_response_{llm_name}.csv'
	df_llm_response = pd.read_csv(llm_response_path)
	# Check if row count is same as that of df_task.csv
	if df_llm_response.shape[0] != df_task.shape[0]:
		raise ValueError(f"... Row count of {df_llm_response} does not match with task dataframe!")
	dfs_to_merge.append(df_llm_response)
df_combined = pd.concat(dfs_to_merge, axis=1)  # Concatenate all dataframes column-wise

cols_to_clean = [f'response_{llm_name}' for llm_name in LLM_NAMES]
# Generate new answer columns for each LLM model
print(f"... Doing data cleaning for {TASK_NAME} task")
for col in cols_to_clean:
	print(f"... Cleaning column {col}")
	answer_col_name = f'{col.replace("response_", "answer_")}'
	if TASK == Task.TF:
		df_answers[answer_col_name] = df_combined.apply(lambda row: extract_tf(response=row[col]), axis=1)
	elif TASK == Task.MC:
		df_answers[answer_col_name] = df_combined.apply(
			lambda row: extract_letter(
				response=row[col],
				options='ABCDE' if pd.notna(row['option_E']) else 'ABCD',
				option_texts=[row['option_A'], row['option_B'], row['option_C'], row['option_D'],
				              row['option_E'] if pd.notna(row['option_E']) else None],
				fuzziness_threshold=FUZZY_THRESHOLD
				),
			axis=1
			)
	else:
		raise NotImplementedError(f"... Task {TASK} not implemented yet!")

answer_col_names = [f'answer_{llm_name}' for llm_name in LLM_NAMES]
missing_values_info = ""
# Check and record missing values for each LLM column
for col in answer_col_names:
	missing_values = df_answers[col].isna()
	missing_indices = missing_values.where(missing_values).dropna().index.tolist()
	missing_values_info += f"Column {col} has {len(missing_indices)} missing values at indices {missing_indices}\n"

# Write missing values info to a text file
with open(f"../data/processed/{TASK_NAME}/{TASK_NAME}_missing_values_info.txt", "w") as text_file:
	text_file.write(missing_values_info)

df_answers.to_csv(f'../data/processed/{TASK_NAME}/{TASK_NAME}_answers.csv', index=False)
