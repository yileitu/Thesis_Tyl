# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pandas import DataFrame

from util.constants import NULL_VALUES, SEED


# Structures
@dataclass
class MCOptions:
	"""
	Option texts for multiple choice questions
	"""
	A: str = ''
	B: str = ''
	C: str = ''
	D: str = ''
	E: str = ''


def set_seed(seed: int = SEED) -> None:
	"""
	Set all seeds to make results reproducible
	:param seed: seed number
	:return: None
	"""
	import torch
	import random
	import numpy as np

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)


def get_total_gpus() -> int:
	"""
	Get total number of GPUs in the server
	:return: number of GPUs
	"""
	import subprocess

	sp = subprocess.Popen(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out_str = sp.communicate()
	out_list = out_str[0].decode("utf-8").split('\n')
	# Subtract one as the last line is empty
	num_gpus = len(out_list) - 1
	print(f"... {num_gpus} GPUs found")
	return num_gpus


def get_idle_gpus(num_gpus: int = 2) -> List[int]:
	"""
	Get idle GPUs in the server
	:param num_gpus: requested number of GPUs
	:return: list of idle GPU IDs
	"""
	import operator
	import subprocess

	total_gpus = get_total_gpus()
	if num_gpus > total_gpus:
		raise ValueError(f'Requested number of GPUs ({num_gpus}) exceeds available GPUs ({total_gpus})')

	sp = subprocess.Popen(
		['nvidia-smi', '--format=csv', '--query-gpu=utilization.gpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
		)
	out_str = sp.communicate()
	out_list = out_str[0].decode("utf-8").split('\n')
	gpu_utilization = []
	for i, gpu in enumerate(out_list[1:-1]):
		utilization = int(gpu.split(' ')[0])
		gpu_utilization.append((i, utilization))
	sorted_gpus = sorted(gpu_utilization, key=operator.itemgetter(1))
	idle_gpus = [gpu[0] for gpu in sorted_gpus[:num_gpus]]
	return idle_gpus


def set_mtec_env(num_gpus: int = 2):
	"""
	Set environments for MTEC server
	:param num_gpus: number of GPUs to use
	:return: PyTorch device
	"""
	import os
	import torch

	# Check if number of GPUs is valid
	if num_gpus < 1 or num_gpus > 8:
		raise ValueError(f"GPU count should be between 1 and 8. Your input is {num_gpus}.")

	idle_gpus = get_idle_gpus(num_gpus)
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, idle_gpus))
	print(f"... setting up GPUs {idle_gpus}")
	# list available GPUs
	gpu_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
	print(f"... {len(gpu_list)} visible 'logical' GPUs: {gpu_list}")
	# Set up GPUs for multi-GPU training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"... using {device}")

	return device


def get_llm_names_and_hf_paths() -> Tuple[Dict[str, str], Dict[str, str], List[str], List[str]]:
	"""
	Get LLM names and their HF paths from a JSON file
	:return: key-value pairs of LLM names and their HF paths
	"""
	import json
	from util.constants import LLM_HF_PATHS_DIR

	with open(LLM_HF_PATHS_DIR, 'r') as f:
		llm_name2hf_path: Dict[str, str] = json.load(f)
	llm_hf_path2name: Dict[str, str] = {v: k for k, v in llm_name2hf_path.items()}
	llm_names: List[str] = list(llm_name2hf_path.keys())
	llm_hf_paths: List[str] = list(llm_name2hf_path.values())
	print(f"... {len(llm_names)} LLMs found: {llm_names}")

	return llm_name2hf_path, llm_hf_path2name, llm_names, llm_hf_paths


def gen_qa_templated_prompt(input_text: str) -> str:
	"""
	Generate a prompt for a given input text and a template
	:param input_text: input text
	:return: templated prompt
	"""
	from util.constants import QA_PROMPT_TEMPLATE

	return QA_PROMPT_TEMPLATE.format(input_text=input_text)


def gen_tf_templated_prompt(passage: str, question: str) -> str:
	"""
	Generate a prompt for a given passage and a question for True/False questions
	:param passage: passage text
	:param question: question text
	:return: templated prompt for Ture/False questions
	"""
	from util.constants import TF_PROMPT_TEMPLATE

	return TF_PROMPT_TEMPLATE.format(passage=passage, question=question)


def gen_mc_templated_prompt(passage: str, question: str, options: MCOptions) -> str:
	"""
	Generate a prompt for a given passage, question and options for multiple choice questions
	:param passage: passage text
	:param question: question text
	:param options: options for the question, should be an instance of MCOptions
	:return: templated prompt for multiple choice questions
	"""
	from util.constants import MC_PROMPT_TEMPLATE_A2E, MC_PROMPT_TEMPLATE_A2D, MC_PROMPT_TEMPLATE_A2E_NO_PASSAGE, \
		MC_PROMPT_TEMPLATE_A2D_NO_PASSAGE

	if passage not in NULL_VALUES:
		if options.E not in NULL_VALUES:
			return MC_PROMPT_TEMPLATE_A2E.format(
				passage=passage, question=question, option_A=options.A, option_B=options.B, option_C=options.C,
				option_D=options.D, option_E=options.E
				)
		else:
			return MC_PROMPT_TEMPLATE_A2D.format(
				passage=passage, question=question, option_A=options.A, option_B=options.B, option_C=options.C,
				option_D=options.D
				)
	else:
		if options.E not in NULL_VALUES:
			return MC_PROMPT_TEMPLATE_A2E_NO_PASSAGE.format(
				question=question, option_A=options.A, option_B=options.B, option_C=options.C, option_D=options.D,
				option_E=options.E
				)
		else:
			return MC_PROMPT_TEMPLATE_A2D_NO_PASSAGE.format(
				question=question, option_A=options.A, option_B=options.B, option_C=options.C, option_D=options.D
				)


def gen_clean_output(output_text: str) -> str:
	"""
	Generate a clean output from the raw output

	:param output_text: raw output text
	:return: clean output text
	"""
	import re
	from util.constants import RESPONSE_SPLIT

	pattern = re.compile(
		r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
		)
	clean_output = output_text.split(RESPONSE_SPLIT)[1].strip()
	clean_output = pattern.sub("", clean_output.strip()).strip()

	return clean_output


def find_first_unprocessed(df: DataFrame, target_col_name: str) -> int:
	"""
	Find the first unprocessed row in a dataframe
	:param df: pandas dataframe
	:param target_col_name: column to check
	:return: first unprocessed row index
	"""
	# Get mask of the data where True for non-NaN (processed) and False for NaN (unprocessed)
	mask = df[target_col_name].notna()

	# Check if all rows are unprocessed
	if not mask.any():
		# All rows are unprocessed, return the first index
		return df.index[0]

	# Find indices where there are changes from non-NaN to NaN
	changes = mask.ne(mask.shift())

	# Identify indices where it changes from True (non-NaN) to False (NaN)
	boundaries = changes.loc[mask.shift() & ~mask].index

	if boundaries.empty:
		print(f"... All rows are processed. Exiting...")
		exit(0)
	else:
		# As the indices are the locations of NaNs, we add 1 to point to the next unprocessed row
		start_index = boundaries.max()
		return start_index
