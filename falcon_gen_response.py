# -*- coding: utf-8 -*-
import torch
import transformers
from transformers import AutoTokenizer, logging

from util.util_func import set_mtec_env, set_seed

# Set environments
NUM_GPU: int = 1
set_seed()
device = set_mtec_env(num_gpus=NUM_GPU)
logging.set_verbosity_error()

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	torch_dtype=torch.bfloat16,
	trust_remote_code=True,
	device_map="auto",
	)
sequences = pipeline(
	"Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
	max_length=200,
	do_sample=True,
	top_k=10,
	num_return_sequences=1,
	eos_token_id=tokenizer.eos_token_id,
	)
for seq in sequences:
	print(f"Result: {seq['generated_text']}")
