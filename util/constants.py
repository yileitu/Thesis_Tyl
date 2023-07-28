# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Final

from transformers import GenerationConfig

# Numbers
SEED: Final[int] = 21946520

# Strings
"""
QA = Question Answering
TF = True/False
MC = Multiple Choice
A2E = 5 options (A, B, C, D, E)
A2D = 4 options (A, B, C, D)
"""
RESPONSE_SPLIT: Final[str] = "### Response:"
QA_PROMPT_TEMPLATE: str = "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\n### Input:\n{input_text}\n\n### Response:\n"
TF_PROMPT_TEMPLATE: str = "Based on the information in: \"{passage}\", please respond to the following question with either 'TRUE' or 'FALSE': \"{question}\"\n\n### Response:"
MC_PROMPT_TEMPLATE_A2E: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nCould you please provide me with the best answer from the options A, B, C, D, and E?\n\n### Response:"
MC_PROMPT_TEMPLATE_A2D: str = "Please consider the following scenario: \n\n\"{passage}\"\n\nNow, I have a question based on this scenario: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nCould you please provide me with the best answer from the options A, B, C, and D?\n\n### Response:"
MC_PROMPT_TEMPLATE_A2E_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your five options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\nE: {option_E}\n\nCould you please provide me with the best answer from the options A, B, C, D, and E?\n\n### Response:"
MC_PROMPT_TEMPLATE_A2D_NO_PASSAGE: str = "Here is a question for you: \n\n\"{question}\"\n\nHere are your four options: \nA: {option_A}\nB: {option_B}\nC: {option_C}\nD: {option_D}\n\nCould you please provide me with the best answer from the options A, B, C, and D?\n\n### Response:"

# Paths
LLM_HF_PATHS_DIR: Final[str] = "util/llm_name2hf_path.json"

# Classes
GEN_CONFIG_FOR_ALL_LLM: Final[GenerationConfig] = GenerationConfig(
	max_length=5000,
	max_new_tokens=5000,
	temperature=0.1,
	top_p=1.0,
	top_k=10,
	num_beams=5,
	repetition_penalty=1.2,
	early_stopping=True,
	do_sample=False,
	)