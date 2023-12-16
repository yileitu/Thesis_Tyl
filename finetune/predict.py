# -*- coding: utf-8 -*-
import os
import sys

import torch

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

import pandas as pd
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList

from util.constants import DIFFICULTY_LABELS, LABEL_TOK, TOP_P, TEMPERATURE
from util.util_func import set_seed

MODEL_DIR: str = "results/qa/20231215034707"
# MODEL_DIR: str = "results/qa/20231214233104"
DATA_PATH: str = "../data/templated/mc/test_data.csv"

# Read finetuned model
set_seed()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
config = GPT2Config.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, config=config)

# tokenizer.pad_token = tokenizer.eos_token
# special_tokens_dict = {'additional_special_tokens': DIFFICULTY_LABELS + [LABEL_TOK]}
# tokenizer.add_special_tokens(special_tokens_dict)
# tokenizer.pad_token = tokenizer.eos_token
# model.resize_token_embeddings(len(tokenizer))

# 检查 "<label>" 是否在词汇表中
# token_id = tokenizer.convert_tokens_to_ids(LABEL_TOK)
# print(f"Token '{LABEL_TOK}' has ID: {token_id}")
# token_exists_by_id = token_id != tokenizer.unk_token_id  # 确保它不是未知词汇的ID
# print(f"Token '<label>' has a valid ID (not unknown): {token_exists_by_id}")

df_test = pd.read_csv(DATA_PATH)


class MyLogitsProcessor(LogitsProcessor):
	def __init__(self, tokenizer, difficulty_labels):
		self.tokenizer = tokenizer
		self.difficulty_label_ids = [tokenizer.convert_tokens_to_ids(label) for label in difficulty_labels]

	def __call__(self, input_ids, scores):
		for token_id in range(scores.size(-1)):
			if token_id not in self.difficulty_label_ids:
				scores[:, token_id] = -float('Inf')
		return scores


logits_processor = MyLogitsProcessor(tokenizer, DIFFICULTY_LABELS)
logits_processor_list = LogitsProcessorList([logits_processor])


def predict_difficulty_label(input_text):
	input_text = input_text.strip()
	input_ids = tokenizer.encode(input_text)

	max_length = 1023  # GPT2 supports sequences of up to 1024 tokens
	if len(input_ids) > max_length:
		input_ids = input_ids[:max_length]
	label_tok_id = tokenizer.encode(LABEL_TOK)[0]
	input_ids.append(label_tok_id)
	input_ids = torch.tensor([input_ids])
	attention_mask = torch.ones(input_ids.shape)

	try:
		output = model.generate(
			input_ids,
			attention_mask=attention_mask,
			max_new_tokens=1,
			temperature=2.0,
			do_sample=True,
			top_p=TOP_P,
			logits_processor=logits_processor_list,
			pad_token_id=tokenizer.eos_token_id,
			)
	except Exception as e:
		print(f"An error occurred: {e}")
		return ""
	output_text = tokenizer.decode(output[0])
	clean_output = output_text.split(LABEL_TOK, 1)[-1].strip()
	return clean_output


tqdm.pandas(desc="Predicting difficulty labels")
df_test['predicted_label'] = df_test['filled_template'].progress_apply(predict_difficulty_label)
df_test.to_csv(DATA_PATH, index=False)
