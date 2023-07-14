# -*- coding: utf-8 -*-
import json
import pickle
from typing import Dict, List, Tuple

from typing_extensions import TypeAlias

from pandalm import EvaluationPipeline
from util.util_func import set_mtec_env, set_seed

set_mtec_env(num_gpus=1)
set_seed()
PandaEval: TypeAlias = Dict[Tuple[str, str], List[int]]

with open('../util/llm_hf_paths.json', 'r') as f:
	llm_name2paths: Dict[str, str] = json.load(f)
llm_hf_paths: List[str] = list(llm_name2paths.values())

pipeline = EvaluationPipeline(
	candidate_paths=llm_hf_paths,
	input_data_path="../data/AlpacaEval_for_pandalm.json",
	output_data_path="../data/output/alpaca_eval_for_pandalm_output.json",
	)
eval_results: PandaEval = pipeline.evaluate()

# Construct a reverse dictionary to find the key from a value
reverse_dict: Dict[str, str] = {v: k for k, v in llm_name2paths.items()}
# Initialize a new empty dictionary to store the transformed data
new_eval_results: PandaEval = {}
# Iterate through the original dictionary
for key, value in eval_results.items():
	# Transform the original key (a tuple of two elements) into a new key
	new_key = (reverse_dict[key[0]], reverse_dict[key[1]])
	# Store the new key and its corresponding value into the new dictionary
	new_eval_results[new_key] = value

with open('alpaca_eval_pandalm_results.pkl', 'wb') as f:
	pickle.dump(new_eval_results, f)
