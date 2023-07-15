import argparse
import json
import logging
import os
import random
import re
import sys
from typing import Dict

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.constants import GEN_CONFIG_FOR_ALL_LLM, RESPONSE_SPLIT, PROMPT_TEMPLATE


def seed_everything(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)


# prompt_templates = {
#     "alpaca": {
#         "description": "Template used by Alpaca-LoRA.",
#         "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
#         "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
#         "response_split": "### Response:",
#     }
# }

prompt_templates = {
	"prompt_input"   : "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
	"prompt_no_input": PROMPT_TEMPLATE,
	"response_split" : RESPONSE_SPLIT,
	}


class CandidateBatchInferenceProvider(object):
	"""
	Batch inference provider for candidate generation models.
	"""

	def __init__(self, model_path) -> None:
		super().__init__()
		self.template = prompt_templates
		try:
			tokenizer = AutoTokenizer.from_pretrained(model_path)
		except:
			tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
		model = AutoModelForCausalLM.from_pretrained(
			model_path,
			torch_dtype=torch.bfloat16,
			device_map="auto",
			)
		if tokenizer.pad_token is None:
			self.smart_tokenizer_and_embedding_resize(
				special_tokens_dict=dict(pad_token="[PAD]"),
				tokenizer=tokenizer,
				model=model,
				)
		if "llama" in model_path:
			tokenizer.add_special_tokens(
				{
					"eos_token": "</s>",
					"bos_token": "</s>",
					"unk_token": "</s>",
					}
				)
		self.tokenizer = tokenizer

		model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
		model.config.bos_token_id = 1
		model.config.eos_token_id = 2
		model.eval()

		if torch.__version__ >= "2" and sys.platform != "win32":
			model = torch.compile(model)

		self.model = model
		self.prepared = []
		self.pattern = re.compile(
			r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
			)

	def generate_prompt(self, instruction, input=None, label=None):
		if input:
			res = self.template["prompt_input"].format(
				instruction=instruction, input=input
				)
		else:
			res = self.template["prompt_no_input"].format(instruction=instruction)
		if label:
			res = f"{res}{label}"
		return res

	def post_process_output(self, text):
		text = text.split(self.template["response_split"])[1].strip()
		text = self.pattern.sub("", text.strip()).strip()
		return text

	def smart_tokenizer_and_embedding_resize(
			self,
			special_tokens_dict: Dict,
			tokenizer: transformers.PreTrainedTokenizer,
			model: transformers.PreTrainedModel,
			):
		"""Resize tokenizer and embedding.

		Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
		"""
		num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
		model.resize_token_embeddings(len(tokenizer))

		if num_new_tokens > 0:
			input_embeddings = model.get_input_embeddings().weight.data
			output_embeddings = model.get_output_embeddings().weight.data

			input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
				dim=0, keepdim=True
				)
			output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
				dim=0, keepdim=True
				)

			input_embeddings[-num_new_tokens:] = input_embeddings_avg
			output_embeddings[-num_new_tokens:] = output_embeddings_avg

	def preprocess_input(self, instruction, input):
		prompt = self.generate_prompt(instruction, input)
		# self.prepared.append(self.tokenizer(prompt, return_tensors="pt", padding=True))
		self.prepared.append(prompt)

	def filter_special_token(self, text):
		return self.pattern.sub("", text.strip()).strip()

	def inference(self):
		generated = []
		for idx in tqdm(range(len(self.prepared))):
			inputs = self.tokenizer(self.prepared[idx], return_tensors="pt")
			input_ids = inputs["input_ids"].to(self.model.device)
			with torch.no_grad():
				generation_output = self.model.generate(
					input_ids=input_ids,
					generation_config=GEN_CONFIG_FOR_ALL_LLM,
					return_dict_in_generate=True,
					output_scores=True,
					)

			for j in range(len(generation_output.sequences)):
				s = generation_output.sequences[j]
				output = self.tokenizer.decode(s)
				resp = self.post_process_output(output)
				resp = self.filter_special_token(resp)
				generated.append(resp)
		return generated


if __name__ == "__main__":
	logging.basicConfig(
		format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
		)

	parser = argparse.ArgumentParser(
		description="Candidate model batch inference script"
		)
	parser.add_argument("-s", "--seed", type=int, default=2023)
	parser.add_argument(
		"-m", "--model_name", default="/ssdwork/wyd/test/llm/output/llama-7b/"
		)
	parser.add_argument(
		"-i",
		"--input_path",
		default="/home/yzh/llm/PandaLM/data/testset-inference-v1.json",
		)
	parser.add_argument("-o", "--output_path", default=None)

	parser.add_argument("-r", "--data_root", default=None)

	args = parser.parse_args()
	if args.data_root is None:
		args.data_root = os.path.join(args.model_name, "batch_eval_outputs")

	logging.info(args)

	seed_everything(args.seed)

	logging.info(f"Loading model from {args.model_name}")
	handler = CandidateBatchInferenceProvider(
		model_path=args.model_name,
		)
	logging.info(f"Loading inference data from{args.input_path}")
	with open(args.input_path) as f:
		input_data = json.load(f)
	logging.info(f"Loaded {len(input_data)} instance(s) to inference")
	results = []
	for item in tqdm(input_data):
		output = handler.preprocess_input(
			instruction=item["instruction"],
			input=item["input"],
			)
	generated = handler.inference()
	for idx, item in enumerate(input_data):
		results.append([item, generated[idx]])

	if args.output_path:
		with open(args.output_path, "w") as f:
			json.dump(results, f)
	else:
		print(results)
