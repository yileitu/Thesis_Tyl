# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Final

import numpy as np
from transformers import GenerationConfig

# Numbers
SEED: Final[int] = 21946520

# Strings
"""
QA = Question Answering
TF = True/False
MC = Multiple Choice
EXAM = MC + TF
A2E = 5 options (A, B, C, D, E)
A2D = 4 options (A, B, C, D)
"""
RESPONSE_SPLIT_FOR_QA: Final[str] = "### Response:"
RESPONSE_SPLIT_FOR_EXAM: Final[str] = "You answer:"
QA_PROMPT_TEMPLATE: str = "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\n### Input:\n{input_text}"
TF_PROMPT_TEMPLATE: str = "Based on the information in: \"{passage}\", please respond to the following question with either 'TRUE' or 'FALSE': \"{question}\". You don't have to provide reasons for your answer."
MC_PROMPT_TEMPLATE_A2E: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nPlease provide me with the best answer in the form of a single letter (A, B, C, D or E). You don't have to provide reasons for your answer."
MC_PROMPT_TEMPLATE_A2D: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nPlease provide me with the best answer in the form of a single letter (A, B, C, or D). You don't have to provide reasons for your answer."
MC_PROMPT_TEMPLATE_A2E_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nPlease provide me with the best answer based on your own knowledge in the form of a single letter (A, B, C, D or E). You don't have to provide reasons for your answer."
MC_PROMPT_TEMPLATE_A2D_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nPlease provide me with the best answer based on your own knowledge in the form of a single letter (A, B, C, or D). You don't have to provide reasons for your answer."

current_date = datetime.now().strftime("%Y-%m-%d")
CHATGPT_SYS_PROMPT = f"You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. \n\
Knowledge cutoff: 2021-09 \n\
Current date: {current_date}"

# Paths
LLM_HF_PATHS_DIR: Final[str] = "util/llm_name2hf_path.json"

# Configs
# No need: top_k, num_beams
MAX_LEN_QA: Final[int] = 2000
MAX_LEN_EXAM: Final[int] = 50
TEMPERATURE: Final[float] = 0.1
TOP_P: Final[float] = 0.9
EARLY_STOPPING: Final[bool] = False
DO_SAMPLE: Final[bool] = True

GEN_CONFIG_FOR_QA: Final[GenerationConfig] = GenerationConfig(
	# max_length=MAX_LEN_QA,
	max_new_tokens=MAX_LEN_QA,
	temperature=TEMPERATURE,
	# top_p=TOP_P,
	early_stopping=EARLY_STOPPING,
	do_sample=DO_SAMPLE
	)

GEN_CONFIG_FOR_EXAM: Final[GenerationConfig] = GenerationConfig(
	# max_length=MAX_LEN_EXAM,
	max_new_tokens=MAX_LEN_EXAM,
	temperature=TEMPERATURE,
	# top_p=TOP_P,
	early_stopping=EARLY_STOPPING,
	do_sample=DO_SAMPLE
	)

# Others
NULL_VALUES = ['', None, np.nan]
PADDING_TOKEN = "<pad>"
