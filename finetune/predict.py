# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from util.constants import LABEL_TOK


MODEL_DIR: str = "results/qa/20231208234239"
DATA_PATH: str = "../data/templated/toy/test_data.csv"

# Read finetuned model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
config = GPT2Config.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, config=config)

tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'additional_special_tokens': [LABEL_TOK]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))


# 检查 "<label>" 是否在词汇表中
token_id = tokenizer.convert_tokens_to_ids("<label>")
print(f"Token '<label>' has ID: {token_id}")
# token_exists_by_id = token_id != tokenizer.unk_token_id  # 确保它不是未知词汇的ID
# print(f"Token '<label>' has a valid ID (not unknown): {token_exists_by_id}")

df_test = pd.read_csv(DATA_PATH)


# def predict_difficulty_label(row):
# 	input_text = row['filled_template']
# 	input_ids = tokenizer.encode(input_text, return_tensors='pt')
# 	output = model.generate(input_ids, max_length=50)
# 	predicted_text = tokenizer.decode(output[0], skip_special_tokens=False)
# 	return predicted_text

def predict_difficulty_label(input_text):
	encoding = tokenizer(input_text, return_tensors='pt')
	input_ids = encoding['input_ids']
	attention_mask = encoding['attention_mask']

	output = model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.shape[1] + 10)
	output_text = tokenizer.decode(output[0], skip_special_tokens=False)
	# label_index = output_text.find(LABEL_TOK)
	# if label_index != -1:
	# 	output_text = output_text[label_index + len("<label>"):]
	# print(f"Output text: {output_text}")
	# output_text = output_text.split()[0] if output_text else ''
	return output_text.strip()


tqdm.pandas(desc="Predicting difficulty labels")
df_test['predicted_label'] = df_test['filled_template'].progress_apply(predict_difficulty_label)
df_test.to_csv(DATA_PATH, index=False)
