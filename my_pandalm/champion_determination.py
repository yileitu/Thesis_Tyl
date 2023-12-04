# -*- coding: utf-8 -*-
import pickle
import random
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Tuple, TypeAlias

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
with open('../data/processed/qa/with_chatgpt/Sample1000/pandalm_eval_results_parsed.pickle', 'rb') as f:
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

		# print(scores)
		max_score = max(scores.values())  # highest score
		# find the models with the highest score, if there are multiple, choose one randomly
		round_winner = random.choice([model for model, score in scores.items() if score == max_score])
		round_winners.append(round_winner)

	return round_winners


# results = {
# 	('huggyllama/llama-7b', 'bigscience/bloom-7b1'): [1, 2, 0, 1, 2, 1, 1, 2, 2, 1],
# 	('huggyllama/llama-7b', 'facebook/opt-6.7b')   : [1, 1, 0, 1, 2, 1, 0, 2, 0, 0],
# 	('bigscience/bloom-7b1', 'facebook/opt-6.7b')  : [1, 1, 0, 2, 2, 1, 2, 2, 0, 0]
# 	}

winners = calculate_scores_per_round(processed_eval_results)

print(winners)
