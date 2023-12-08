# -*- coding: utf-8 -*-
import os.path
import pickle
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Tuple, TypeAlias

import pandas as pd

from util.constants import DIFFICULTY_LABELS
from util.util_func import set_seed

set_seed()
PandaEvalResult: TypeAlias = Dict[Tuple[str, str], List[int]]


class Outcome(IntEnum):
	"""
	Enum for the possible outcomes of a round of the LLM competition
	"""
	DRAW: int = 0
	WIN_FIRST: int = 1  # first model wins
	WIN_SECOND: int = 2  # second model wins


# Define constants for scoring
DRAW_SCORE = 1
WIN_SCORE = 3


def process_eval_results(element):
	"""
	Replace the model paths with their names

	:param element:
	:return:
	"""
	if 'text-davinci-002-render-sha' in element:
		return 'text-davinci-002-render-sha'
	elif 'Llama-2-7b-chat' in element:
		return 'Llama-2-7b-chat'
	elif 'Llama-2-13b-chat' in element:
		return 'Llama-2-13b-chat'
	elif 'pythia-2.8b' in element:
		return 'pythia-2.8b'
	else:
		raise ValueError(f"Unknown model path: {element}")


# Read pickled results
parent_dir = "../data/processed/qa/with_chatgpt/Sample1000"
with open(os.path.join(parent_dir, 'pandalm_eval_results_parsed.pickle'), 'rb') as f:
	eval_results: PandaEvalResult = pickle.load(f)
processed_eval_results = {}
for key, value in eval_results.items():
	new_key = tuple(process_eval_results(elem) for elem in key)
	processed_eval_results[new_key] = value


def calculate_scores_per_round(results):
	"""
	Calculate the scores for each round of the LLM competition

	:param results: dictionary with keys being the model pairs and values being the results of each round (each data point)
	:return: list of the winners of each round
	"""
	rounds = zip(*results.values())  # Transpose results to get each round's results
	round_winners: List[str] = []

	for round_results in rounds:
		scores: Dict[str, int] = defaultdict(int)  # Counter for calculating each model's scores
		for outcome, pair in zip(round_results, results.keys()):
			if outcome == Outcome.WIN_FIRST:
				scores[pair[0]] += WIN_SCORE
			elif outcome == Outcome.WIN_SECOND:
				scores[pair[1]] += WIN_SCORE
			elif outcome == Outcome.DRAW:
				scores[pair[0]] += DRAW_SCORE
				scores[pair[1]] += DRAW_SCORE

		# Pick the winner of the round
		max_score = max(scores.values())  # highest score
		# find the models with the highest score, if there are multiple, choose the least powerful one
		candidates = [model for model, score in scores.items() if score == max_score]
		if len(candidates) > 1:
			print(f"Multiple models with the highest score: {candidates}")
		round_winner = candidates[0]
		round_winners.append(round_winner)

	return round_winners


# results = {
# 	('huggyllama/llama-7b', 'bigscience/bloom-7b1'): [1, 2, 0, 1, 2, 1, 1, 2, 2, 1],
# 	('huggyllama/llama-7b', 'facebook/opt-6.7b')   : [1, 1, 0, 1, 2, 1, 0, 2, 0, 0],
# 	('bigscience/bloom-7b1', 'facebook/opt-6.7b')  : [1, 1, 0, 2, 2, 1, 2, 2, 0, 0]
# 	}

winners = calculate_scores_per_round(processed_eval_results)
print(winners)

# Save winners to sampled data
df_qa_sampled = pd.read_csv(os.path.join(parent_dir, 'qa_combined_nonempty.csv'))
df_qa_sampled['winner'] = winners


# Label difficulty level
def determine_difficulty_label(row) -> str:
	"""
	Determine the "difficulty" label for each question-answer pair.

	:param row: data point
	:return: Difficulty label
	"""
	if row['winner'] == 'text-davinci-002-render-sha':
		return DIFFICULTY_LABELS[3]
	elif row['winner'] == 'Llama-2-13b-chat':
		return DIFFICULTY_LABELS[2]
	elif row['winner'] == 'Llama-2-7b-chat':
		return DIFFICULTY_LABELS[1]
	elif row['winner'] == 'pythia-2.8b':
		return DIFFICULTY_LABELS[0]
	else:
		return DIFFICULTY_LABELS[4]


df_qa_sampled['label'] = df_qa_sampled.apply(determine_difficulty_label, axis=1)
df_qa_sampled.to_csv(os.path.join(parent_dir, 'qa_combined_nonempty.csv'), index=False)
