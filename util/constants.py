# -*- coding: utf-8 -*-
from typing import Final

from transformers import GenerationConfig

SEED: Final[int] = 21946520

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


# TODO: 使之成为通用模板
prompt_templates = {
	"alpaca": {
		"description"    : "Template used by Alpaca-LoRA.",
		"prompt_input"   : "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
		"prompt_no_input": "Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.\n\n### Input:\n{instruction}\n\n### Response:\n",
		"response_split" : "### Response:",
		}
	}
