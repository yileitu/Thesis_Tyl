# -*- coding: utf-8 -*-
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from util.util_func import read_txt, save_df_to_csv, connect_word_list_to_str_with_or
from util.constants import LABEL_TOK, DIFFICULTY_LABELS

# Constants
SEED = 42
TEST_RATIO = 0.2
DEV_RATIO = TEST_RATIO / (1 - TEST_RATIO)  # 0.25 x 0.8 = 0.2

# Define file paths
save_dir = '../data/templated/mc'
df_labeled_path = '../data/labeled/mc_labeled.csv'
# Define file paths for each template
template_with_passage_and_option_E_path = 'prompt_template/mc_template_with_passage_and_option_e.txt'
template_with_passage_without_option_E_path = 'prompt_template/mc_template_with_passage_without_option_e.txt'
template_without_passage_with_option_E_path = 'prompt_template/mc_template_without_passage_with_option_e.txt'
template_without_passage_without_option_E_path = 'prompt_template/mc_template_without_passage_without_option_e.txt'

# Read each template
template_with_passage_and_option_E = read_txt(template_with_passage_and_option_E_path)
template_with_passage_without_option_E = read_txt(template_with_passage_without_option_E_path)
template_without_passage_with_option_E = read_txt(template_without_passage_with_option_E_path)
template_without_passage_without_option_E = read_txt(template_without_passage_without_option_E_path)

df = pd.read_csv(df_labeled_path)
difficulty_labels_text = connect_word_list_to_str_with_or(DIFFICULTY_LABELS)

def fill_template(row: pd.DataFrame) -> str:
	"""
	Fill the template with the data in the row

	:param row: datapoint
	:return: filled template text
	"""
	# Check the presence of 'passage' and 'option_E' in the row
	has_passage = pd.notna(row['passage'])
	has_option_E = pd.notna(row['option_E'])

	# Select the appropriate template based on the data available
	if has_passage and has_option_E:
		template = template_with_passage_and_option_E
	elif has_passage and not has_option_E:
		template = template_with_passage_without_option_E
	elif not has_passage and has_option_E:
		template = template_without_passage_with_option_E
	else:  # Neither passage nor option_E
		template = template_without_passage_without_option_E

	return template.format(
		passage=row.get('passage', ''),
		question=row.get('question', ''),
		option_A=row.get('option_A', ''),
		option_B=row.get('option_B', ''),
		option_C=row.get('option_C', ''),
		option_D=row.get('option_D', ''),
		option_E=row.get('option_E', ''),
		difficulty_labels=difficulty_labels_text
		)


def preprocess_data(df):
	# Combine the 'filled_template' and 'label' into a single string
	df['input_target'] = df['filled_template'] + f" {LABEL_TOK} " + df['label']
	return df['input_target'].tolist()


df['filled_template'] = df.apply(fill_template, axis=1)
fine_tune_df = df[['id', 'filled_template', 'label']]

# Split the data into train+dev and test sets
train_dev_df, test_df = train_test_split(fine_tune_df, test_size=TEST_RATIO, random_state=SEED)

# Split the train+dev set into train and dev sets
train_df, dev_df = train_test_split(train_dev_df, test_size=DEV_RATIO, random_state=SEED)  # 0.25 x 0.8 = 0.2

# Save these splits to new CSV files
save_df_to_csv(df=train_df, path=os.path.join(save_dir, 'train_data.csv'), index=False)
save_df_to_csv(df=dev_df, path=os.path.join(save_dir, 'dev_data.csv'), index=False)
save_df_to_csv(df=test_df, path=os.path.join(save_dir, 'test_data.csv'), index=False)

# train_texts = preprocess_data(train_df)
# dev_texts = preprocess_data(dev_df)
# test_texts = preprocess_data(test_df)
#
# # Save texts to files
# with open(os.path.join(save_dir, 'train_texts.txt'), 'w') as f:
# 	f.write("<|endoftext|>\n".join(train_texts))
# with open(os.path.join(save_dir, 'dev_texts.txt'), 'w') as f:
# 	f.write("<|endoftext|>\n".join(dev_texts))
# with open(os.path.join(save_dir, 'test_texts.txt'), 'w') as f:
# 	f.write("<|endoftext|>\n".join(test_texts))
