# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict, List, Tuple

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

from typing_extensions import TypeAlias

from evaluation_pipeline import EvaluationPipeline
from util.util_func import set_gpu_env, set_seed

include_chatgpt: bool = True

PandaEvalResult: TypeAlias = Dict[Tuple[str, str], List[int]]
# device = set_gpu_env(num_gpus=1)
set_seed()
set_gpu_env(num_gpus=2)

# Paths
# pandalm_hf_path = "WeOpenML/PandaLM-Alpaca-7B-v1"
if include_chatgpt:
	qa_data_dir: str = '../data/processed/qa/with_chatgpt/Sample1000'
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'text-davinci-002-render-sha']
else:
	qa_data_dir: str = '../data/processed/qa/without_chatgpt'
	llm_names: List[str] = ['pythia-2.8b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat']
llm_response_paths: List[str] = [os.path.join(qa_data_dir, f"pandalm_{llm}.json") for llm in llm_names]
qa_input_path: str = os.path.join(qa_data_dir, "pandalm_qa_input.json")
pandalm_eval_output_path: str = os.path.join(qa_data_dir, "pandalm_eval_output.pickle")

pipeline = EvaluationPipeline(
	candidate_paths=llm_response_paths,
	# pandalm_path=pandalm_hf_path,
	input_data_path=qa_input_path,
	output_data_path=qa_data_dir,
	)
eval_results: PandaEvalResult = pipeline.evaluate()

# # Initialize a new empty dictionary to store the transformed data
# new_eval_results: PandaEvalResult = {}
# # Iterate through the original dictionary
# for key, value in eval_results.items():
# 	# Transform the original key (a tuple of two elements) into a new key
# 	new_key = (reverse_dict[key[0]], reverse_dict[key[1]])
# 	# Store the new key and its corresponding value into the new dictionary
# 	new_eval_results[new_key] = value
#
# with open('alpaca_eval_pandalm_results.pkl', 'wb') as f:
# 	pickle.dump(new_eval_results, f)
