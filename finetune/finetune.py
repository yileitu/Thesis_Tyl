# -*- coding: utf-8 -*-
import pandas as pd
from transformers import DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer, TextDataset, Trainer, \
	TrainingArguments


# Load and preprocess the data
def preprocess_data(df):
	# Combine the 'filled_template' and 'label' into a single string
	df['input_target'] = df['filled_template'] + " <label> " + df['label']
	return df['input_target'].tolist()


# Prepare datasets
train_df = pd.read_csv('../data/templated/mc/train_data.csv')
dev_df = pd.read_csv('../data/templated/mc/dev_data.csv')
test_df = pd.read_csv('../data/templated/mc/test_data.csv')

train_texts = preprocess_data(train_df)
dev_texts = preprocess_data(dev_df)
test_texts = preprocess_data(test_df)

# Save preprocessed texts to files (required by TextDataset)
with open('train_texts.txt', 'w') as f:
	f.write("\n".join(train_texts))

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Special tokens to add (if any)
special_tokens_dict = {'additional_special_tokens': ['<label>']}
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
