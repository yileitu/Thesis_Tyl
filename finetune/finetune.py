# -*- coding: utf-8 -*-
import os

import pandas as pd
from transformers import DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer, TextDataset, Trainer, \
	TrainingArguments

from util.constants import LABEL_TOK


# Load the GPT-2 tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Dataset
data_dir = '../data/templated/mc'
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(data_dir, 'train_data.csv'),
    block_size=128  # This can be adjusted, often set to 512 or 1024 for larger models
)



# Special tokens to add (if any)
special_tokens_dict = {'additional_special_tokens': [LABEL_TOK]}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Prepare datasets for training
train_dataset = TextDataset(tokenizer=tokenizer, file_path='train_texts.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
	output_dir='./results',
	num_train_epochs=3,
	per_device_train_batch_size=2,
	save_steps=10_000,
	save_total_limit=2,
	)

# Initialize Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=train_dataset,
	)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./finetuned_model')
