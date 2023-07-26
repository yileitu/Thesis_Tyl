# -*- coding: utf-8 -*-
from typing import Final

from transformers import GenerationConfig

# Numbers
SEED: Final[int] = 21946520

# Strings
"""
QA = Question Answering
TF = True/False
MC = Multiple Choice
"""
QA_PROMPT_TEMPLATE: str = "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\n### Input:\n{input_text}\n\n### Response:\n"
QA_RESPONSE_SPLIT: Final[str] = "### Response:"
TF_PROMPT_TEMPLATE: str = "Based on the information in: \"{passage}\", please respond to the following question with either 'TRUE' or 'FALSE': \"{question}\""

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
