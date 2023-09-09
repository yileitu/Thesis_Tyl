# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import pandas as pd
from pandas import DataFrame

from util.constants import NULL_VALUES, PADDING_TOKEN, RESPONSE_SPLIT_FOR_EXAM, RESPONSE_SPLIT_FOR_QA, SEED
from util.struct import MCOptions, Task


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


def set_gpu_env(num_gpus: int = 1):
	"""
	Set GPU environments in the server
	:param num_gpus: number of GPUs to use
	:return: PyTorch device
	"""
	import os
	import torch

	idle_gpus = get_idle_gpus(num_gpus)
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, idle_gpus))
	print(f"... Available GPUs {idle_gpus}")
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


def gen_input_with_split(text: str, task: Task) -> str:
	"""
	Generate input text with the response split
	:param text: original input text
	:param task: task type
	:return: input text with the response split
	"""
	if task == Task.QA:
		return text + "\n\n" + RESPONSE_SPLIT_FOR_QA
	else:
		return text + "\n\n" + RESPONSE_SPLIT_FOR_EXAM


def gen_clean_output(output_text: str, task: Task) -> str:
	"""
	Generate a clean output from the raw output
	:param output_text: raw output text
	:param task: task type
	:return: clean output text
	"""
	import re

	pattern = re.compile(
		r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
		)
	if task == Task.QA:
		clean_output = output_text.split(RESPONSE_SPLIT_FOR_QA)[1].strip()
	else:
		clean_output = output_text.split(RESPONSE_SPLIT_FOR_EXAM)[1].strip()
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


def setup_signal_handlers(df_to_save, save_path):
	"""
	Setup signal handlers to save data when program is terminated manually by user
	:param df_to_save: dataframe to save
	:param save_path: path to save the dataframe
	"""
	import signal
	import sys

	def save_data():
		df_to_save.to_csv(save_path, index=False)
		print(f"\n ... Program terminated by user. Data {save_path} saved successfully!")

	def signal_handler(sig, frame):
		save_data()
		sys.exit(0)

	# 设置信号处理器捕捉 SIGINT 和 SIGTERM
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)


def gen_response_file(response_df_path: str, task_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
	"""
	Generate a response file for a given task dataframe if it does not exist. If exists, update its index to match the task_df
	:param response_df_path: Path to the response dataframe
	:param task_df: Task dataframe
	:param col_name: Column name for the response
	:return: Response dataframe with the same index as the task dataframe
	"""
	import os
	# Check if the file exists, create it if not
	if not os.path.exists(response_df_path):
		response_df = pd.DataFrame(index=task_df.index, columns=[col_name])
		response_df.to_csv(response_df_path, index=False)
		print(f"... Empty DataFrame saved to {response_df_path}")
	else:
		print(f"... File already exists: {response_df_path}")
		response_df = pd.read_csv(response_df_path)
		dummy_df = pd.DataFrame(index=task_df.index, columns=[col_name])
		dummy_df.update(response_df)
		dummy_df = dummy_df.where(pd.notnull(dummy_df), None)
		response_df = dummy_df

	return response_df


def get_task_df_path(task: Task, llm_name: str) -> Tuple[str, str]:
	"""
	Get the path to the task dataframe and the response dataframe
	:param task: Task type
	:param llm_name: Name of the
	:return:
	"""
	if task == Task.MC:
		task_name = 'mc'
	elif task == Task.TF:
		task_name = 'tf'
	elif task == Task.QA:
		task_name = 'qa'
	elif task == Task.TOY_MC:
		task_name = 'toy_mc'
	else:
		raise ValueError(f"Invalid task type {task}")

	df_path: str = f"data/output/{task_name}/{task_name}.csv"
	response_path: str = f"data/output/{task_name}/{task_name}_response_{llm_name}.csv"

	return df_path, response_path


def set_llm_config(model, tokenizer, device, task: Task):
	"""
	Set LLM configurations. Padding token is the same as EOS token
	:param model: LLM model
	:param tokenizer: LLM tokenizer
	:param device: PyTorch device
	:param task: Task type
	:return: None
	"""
	from util.constants import MAX_LEN_QA, MAX_LEN_EXAM, TEMPERATURE, TOP_P, EARLY_STOPPING, DO_SAMPLE
	from transformers import GenerationConfig

	eos_token_id = tokenizer.eos_token_id
	tokenizer.pad_token = tokenizer.eos_token
	model.config.pad_token_id = eos_token_id
	model.to(device)
	model.eval()

	gen_config = GenerationConfig(
		# max_length=MAX_LEN_QA if task == Task.QA else MAX_LEN_EXAM,
		max_new_tokens=MAX_LEN_QA if task == Task.QA else MAX_LEN_EXAM,
		temperature=TEMPERATURE,
		top_p=TOP_P,
		early_stopping=EARLY_STOPPING,
		do_sample=DO_SAMPLE,
		pad_token_id=eos_token_id,
		)
	return gen_config


def set_llama_config(model, tokenizer, device, task: Task):
	"""
	Set Llama configurations. Follow up HF tips https://huggingface.co/docs/transformers/model_doc/llama2
	:param model: LLM model
	:param tokenizer: LLM tokenizer
	:param device: PyTorch device
	:param task: Task type
	:return: None
	"""
	from util.constants import MAX_LEN_QA, MAX_LEN_EXAM, TEMPERATURE, TOP_P, EARLY_STOPPING, DO_SAMPLE
	from transformers import GenerationConfig

	pad_token_id = tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})
	model.resize_token_embeddings(len(tokenizer))
	model.config.pad_token_id = pad_token_id
	model.config.pretraining_tp = 100
	model.to(device)
	model.eval()

	gen_config = GenerationConfig(
		# max_length=MAX_LEN_QA if task == Task.QA else MAX_LEN_EXAM,
		max_new_tokens=MAX_LEN_QA if task == Task.QA else MAX_LEN_EXAM,
		temperature=TEMPERATURE,
		top_p=TOP_P,
		early_stopping=EARLY_STOPPING,
		do_sample=DO_SAMPLE,
		pad_token_id=pad_token_id,
		)
	return gen_config
