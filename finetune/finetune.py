# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from util.constants import LABEL_TOK

# Load the GPT-2 tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add the new tokens to the vocabulary.
special_tokens_dict = {'additional_special_tokens': [LABEL_TOK]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load labeled data
data_dir = '../data/templated/mc'
train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))


# dev_df = pd.read_csv('dev_data.csv')
# test_df = pd.read_csv('test_data.csv')


# Function to concatenate input text and label
def prepare_data(row):
	return f"{row['filled_template']} {LABEL_TOK} {row['label']}"


def tokenize_texts(texts):
	return tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")


class MyDataset(Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

	def __len__(self):
		return len(self.encodings.input_ids)


# Prepare the datasets
train_texts = train_df.apply(prepare_data, axis=1).tolist()
# dev_texts = dev_df.apply(prepare_data, axis=1).tolist()
# test_texts = test_df.apply(prepare_data, axis=1).tolist()

train_encodings = tokenize_texts(train_texts)
# dev_encodings = tokenize_texts(dev_texts)
# test_encodings = tokenize_texts(test_texts)

train_dataset = MyDataset(train_encodings)
# dev_dataset = MyDataset(dev_encodings)
# test_dataset = MyDataset(test_encodings)

sample_index = 0  # You can choose any valid index

# Retrieve the sample
sample = train_dataset[sample_index]

# Decode the sample to text
input_ids = sample['input_ids'].numpy()  # Convert to numpy array if it's a tensor
decoded_text = tokenizer.decode(input_ids)

print("Decoded text from the dataset:", decoded_text)
