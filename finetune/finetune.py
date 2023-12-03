# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

import torch

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

from dataclasses import asdict

import wandb
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer, \
	GPTNeoXForCausalLM, HfArgumentParser, TextDataset, Trainer, TrainingArguments

from arguments import MyArguments
from util.constants import LABEL_TOK
from util.util_func import set_gpu_env, set_seed, setup_logger, transform_dict

# Parse arguments
parser = HfArgumentParser((MyArguments, TrainingArguments))
my_args, training_args = parser.parse_args_into_dataclasses()
my_args: MyArguments
training_args: TrainingArguments

# Setup wandb
wandb_proj_name = f"LMRouter-{my_args.model_name.upper()}-{my_args.task.upper()}"
serial = f"Epoch{int(training_args.num_train_epochs)}-LR{training_args.learning_rate}-Seed{training_args.seed}"
os.environ["WANDB_PROJECT"] = wandb_proj_name
wandb.init(
	project=wandb_proj_name,
	name=serial,
	)

# Setup training args
my_args.data_dir = os.path.join(my_args.data_dir, my_args.task)
training_args.output_dir = os.path.join(training_args.output_dir, my_args.task)
training_args.report_to = ["wandb"]
training_args.logging_steps = 50
training_args.run_name = serial
training_args.save_total_limit = 1
training_args.save_strategy = "no"
wandb.log(transform_dict(asdict(my_args)))

# Misc Setup
set_seed(training_args.seed)
logger = setup_logger(training_args)
device = set_gpu_env(num_gpus=my_args.n_gpu)

# Load tokenizer and model
if my_args.model_name == 'gpt2':
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	logger.info(f"Loaded GPT2 model.")
elif my_args.model_name == 'pythia':
	pythia_hf_path = "EleutherAI/pythia-2.8b-deduped"
	tokenizer = AutoTokenizer.from_pretrained(pythia_hf_path)
	model = GPTNeoXForCausalLM.from_pretrained(pythia_hf_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
elif my_args.model_name == 'finetuned':
	if my_args.model_path is None:
		raise ValueError(f"Model checkpoint path must be specified for finetuned model.")
	tokenizer = GPT2Tokenizer.from_pretrained(my_args.model_path)
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	logger.info(f"Loaded finetuned model from {my_args.model_path}")
else:
	raise ValueError(f"Unsupported model name: {my_args.model_name}")
model.to(device)

# Add the new tokens to the vocabulary.
special_tokens_dict = {'additional_special_tokens': [LABEL_TOK]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load labeled data
train_dataset = TextDataset(
	tokenizer=tokenizer,
	file_path=os.path.join(my_args.data_dir, 'train_data.csv'),
	block_size=my_args.max_length
	)
dev_dataset = TextDataset(
	tokenizer=tokenizer,
	file_path=os.path.join(my_args.data_dir, 'dev_data.csv'),
	block_size=my_args.max_length
	)

data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False,
	)

trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=train_dataset,
	eval_dataset=dev_dataset
	)

train_result = trainer.train()
current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
timestamped_dir = os.path.join(training_args.output_dir, current_timestamp)
trainer.save_model(output_dir=timestamped_dir)
# model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(save_directory=training_args.output_dir)
metrics = train_result.metrics
logger.info(f"*** Train Metrics *** \n{metrics}")

# logger.info("*** Test ***")
# metrics = trainer.evaluate(eval_dataset=test_dataset)
# logger.info(f"*** Test Metrics *** \n{metrics}")


# # Load labeled data
# train_df = pd.read_csv(os.path.join(my_args.data_dir, 'train_data.csv'))
# dev_df = pd.read_csv(os.path.join(my_args.data_dir, 'dev_data.csv'))
# test_df = pd.read_csv(os.path.join(my_args.data_dir, 'test_data.csv'))
#

# # Function to concatenate input text and label
# def prepare_data(row):
# 	return f"{row['filled_template']} {LABEL_TOK} {row['label']}"
#
#
# def tokenize_texts(texts):
# 	return tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
#
#
#
# class MyDataset(Dataset):
# 	def __init__(self, encodings):
# 		self.encodings = encodings
#
# 	def __getitem__(self, idx):
# 		return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#
# 	def __len__(self):
# 		return len(self.encodings.input_ids)
#
#
# # Prepare the datasets
# train_texts = train_df.apply(prepare_data, axis=1).tolist()
# # dev_texts = dev_df.apply(prepare_data, axis=1).tolist()
# # test_texts = test_df.apply(prepare_data, axis=1).tolist()
#
# train_encodings = tokenize_texts(train_texts)
# # dev_encodings = tokenize_texts(dev_texts)
# # test_encodings = tokenize_texts(test_texts)
#
# train_dataset = MyDataset(train_encodings)
# # dev_dataset = MyDataset(dev_encodings)
# # test_dataset = MyDataset(test_encodings)

# Fine-tune the model
