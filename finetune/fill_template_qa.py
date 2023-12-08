# -*- coding: utf-8 -*-
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from util.constants import DIFFICULTY_LABELS, LABEL_TOK
from util.util_func import connect_word_list_to_str_with_or, read_txt, save_df_to_csv

# Constants
SEED = 42
TEST_RATIO = 0.2

# Define file paths
save_dir = '../data/templated/qa'
df_labeled_path = '../data/processed/qa/without_chatgpt/sample/labeled/qa_labeled.csv'

# Template
template_path = 'prompt_template/qa_template.txt'
template = read_txt(template_path)

df_labeled = pd.read_csv(df_labeled_path)
difficulty_labels_text = connect_word_list_to_str_with_or(DIFFICULTY_LABELS[:-1])


def fill_template(row: pd.DataFrame) -> str:
	"""
	Fill the template with the data in the row

	:param row: datapoint
	:return: filled template text
	"""
	return template.format(
		input=row.get('input', ''),
		difficulty_labels=difficulty_labels_text
		)


def preprocess_data(df):
	# Combine the 'filled_template' and 'label' into a single string
	df['input_target'] = df['filled_template'] + f" {LABEL_TOK} " + df['label'] + " "
	return df['input_target'].tolist()


df_labeled['filled_template'] = df_labeled.apply(fill_template, axis=1)
fine_tune_df = df_labeled[['id', 'filled_template', 'label']]

# Split the data into train+dev and test sets
train_df, test_df = train_test_split(fine_tune_df, test_size=TEST_RATIO, random_state=SEED)

# Save these splits to new CSV files
save_df_to_csv(df=train_df, path=os.path.join(save_dir, 'train_data.csv'), index=False)
save_df_to_csv(df=test_df, path=os.path.join(save_dir, 'test_data.csv'), index=False)

train_texts = preprocess_data(train_df)
test_texts = preprocess_data(test_df)

# Save texts to files
with open(os.path.join(save_dir, 'train_texts.txt'), 'w') as f:
	f.write("<|endoftext|>\n\n\n".join(train_texts))
with open(os.path.join(save_dir, 'test_texts.txt'), 'w') as f:
	f.write("<|endoftext|>\n\n\n".join(test_texts))
