# -*- coding: utf-8 -*-
from typing import Final

from transformers import GenerationConfig

SEED: Final[int] = 21946520

PROMPT_TEMPLATE: str = "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\n### Input:\n{input_text}\n\n### Response:\n"
RESPONSE_SPLIT: Final[str] = "### Response:"

GEN_CONFIG_FOR_ALL_LLM: Final[GenerationConfig] = GenerationConfig(
	max_length=10000,
	max_new_tokens=10000,
	temperature=0.1,
	top_p=0.9,
	top_k=10,
	num_beams=5,
	repetition_penalty=1.2,
	early_stopping=True,
	do_sample=False,
	)
