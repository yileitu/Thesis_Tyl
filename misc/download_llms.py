# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# alpaca_7b_tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native")
# alpaca_7b = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native")

# alpaca_13b_tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-13b")
# alpaca_13b = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-13b")

# koala_7b_tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-7B-HF")
# koala_7b = AutoModelForCausalLM.from_pretrained("TheBloke/koala-7B-HF")

# koala_13b_tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-13B-HF")
# koala_13b = AutoModelForCausalLM.from_pretrained("TheBloke/koala-13B-HF")

# vicuna_7b_tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-7b-1.1")
# vicuna_7b = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-7b-1.1")

# vicuna_13b_tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-13b-1.1")
# vicuna_13b = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-13b-1.1")

# dolly_3b_tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
# dolly_3b = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", torch_dtype=torch.bfloat16)

# falcon_7b_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
# falcon_7b = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
