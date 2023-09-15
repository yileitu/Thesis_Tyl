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
RESPONSE_SPLIT_FOR_QA: Final[str] = "Response:"
RESPONSE_SPLIT_FOR_EXAM: Final[str] = "So your answer is:"
RESPONSE_SPLIT_FOR_EXAM_LLAMA: Final[str] = RESPONSE_SPLIT_FOR_EXAM
# EXAM_PROMPT_ENDING: Final[str] = "You don't have to provide reasons."
EXAM_PROMPT_ENDING: Final[str] = ""
QA_PROMPT_TEMPLATE: str = "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\nInput:\n{input_text}"
TF_PROMPT_TEMPLATE: str = "Based on the information in: \"{passage}\", please respond to the following question with either 'TRUE' or 'FALSE': \"{question}\". " + EXAM_PROMPT_ENDING
MC_PROMPT_TEMPLATE_A2E: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nPlease provide me with the best option to the question in the form of a single letter (A, B, C, D or E). " + EXAM_PROMPT_ENDING
MC_PROMPT_TEMPLATE_A2D: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nPlease provide me with the best option to the question in the form of a single letter (A, B, C, or D). " + EXAM_PROMPT_ENDING
MC_PROMPT_TEMPLATE_A2E_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nPlease provide me with the best option to the question based on your own knowledge in the form of a single letter (A, B, C, D or E). " + EXAM_PROMPT_ENDING
MC_PROMPT_TEMPLATE_A2D_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nPlease provide me with the best option to the question based on your own knowledge in the form of a single letter (A, B, C, or D). " + EXAM_PROMPT_ENDING

current_date = datetime.now().strftime("%Y-%m-%d")
CHATGPT_SYS_PROMPT = f"You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. \n\
Knowledge cutoff: 2021-09 \n\
Current date: {current_date}"

# Paths
LLM_HF_PATHS_DIR: Final[str] = "util/llm_name2hf_path.json"

# Configs
MAX_LEN_QA: Final[int] = 1000
MAX_LEN_EXAM: Final[int] = 50
TEMPERATURE: Final[float] = 0.1
TOP_P: Final[float] = 0.9
NUM_BEAMS: Final[int] = 5
EARLY_STOPPING: Final[bool] = True
DO_SAMPLE: Final[bool] = False

GEN_CONFIG_FOR_QA: Final[GenerationConfig] = GenerationConfig(
	max_new_tokens=MAX_LEN_QA,
	temperature=TEMPERATURE,
	top_p=TOP_P,
	early_stopping=EARLY_STOPPING,
	do_sample=True
	)

GEN_CONFIG_FOR_EXAM: Final[GenerationConfig] = GenerationConfig(
	max_new_tokens=MAX_LEN_EXAM,
	temperature=TEMPERATURE,
	# top_p=TOP_P,
	early_stopping=EARLY_STOPPING,
	do_sample=DO_SAMPLE
	)

# Others
NULL_VALUES = ['', None, np.nan]
PADDING_TOKEN = "<pad>"
