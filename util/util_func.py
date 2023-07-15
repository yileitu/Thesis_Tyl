# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

from util.constants import SEED


def set_seed(seed: int = SEED) -> None:
	"""
	Set all seeds to make results reproducible

	:param seed: seed number
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
	:return: PyTorch device
	"""
	import os
	import torch

	# Check if number of GPUs is valid
	if num_gpus < 1 or num_gpus > 8:
		raise ValueError(f"GPU count should be between 1 and 8. Your input is {num_gpus}.")

	# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# gpu_indices = ",".join(str(i) for i in range(num_gpus))
	# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
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


def gen_templated_prompt(input_text: str) -> str:
	"""
	Generate a prompt for a given input text and a template

	:param input_text: input text
	:return: templated prompt
	"""
	from util.constants import PROMPT_TEMPLATE

	return PROMPT_TEMPLATE.format(input_text=input_text)


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