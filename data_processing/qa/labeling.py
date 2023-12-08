# -*- coding: utf-8 -*-
import os

import pandas as pd

from util.constants import DIFFICULTY_LABELS
from util.util_func import save_df_to_csv


def determine_label(row) -> str:
	"""
	Determine the "difficulty" label for each question-answer pair.

	:param row: data point
	:return: Difficulty label
	"""
	scores = [
		row['response_pythia-2.8b_moverscore'],
		row['response_Llama-2-7b-chat_moverscore'],
		row['response_Llama-2-13b-chat_moverscore'],
		row['response_Llama-2-70b-chat_moverscore']
		]

	max_score_index = scores.index(max(scores))
	return DIFFICULTY_LABELS[max_score_index]


# Read processed QA with moverscores
parent_dir = '../../data/processed/qa/without_chatgpt/sample'
df_qa: pd.DataFrame = pd.read_csv(os.path.join(parent_dir, 'qa_combined_nonempty_moverscore.csv'), index_col='id')

# Label each MC/TF question-answer pair
df_qa['label'] = df_qa.apply(determine_label, axis=1)

labeled_dir_path = os.path.join(parent_dir, 'labeled')
df_labeled_path = os.path.join(labeled_dir_path, f'qa_labeled.csv')
print(f"Saving labeled data to {df_labeled_path}")
save_df_to_csv(df=df_qa, path=df_labeled_path, index=True)
