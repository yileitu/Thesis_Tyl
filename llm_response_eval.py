# -*- coding: utf-8 -*-
import warnings
from typing import List

import pandas as pd
import torch
from bert_score import score
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_seed

# Set environments
set_seed()
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DF_PATH: str = "data/alpaca_sample50.csv"
COLS_TO_ADD: List[str] = ['bert_score', 'alpaca_output']
DF_TYPE = {
	'instruction': str,
	'input'      : str,
	'output'     : str,
	}

df = pd.read_csv(DF_PATH, dtype=DF_TYPE)
# 对于列表中的每个列名，检查其是否存在，如果不存在则添加
for column in COLS_TO_ADD:
	if column not in df.columns:
		df[column] = None

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native")
model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", torch_dtype=torch.float16)
model.to(device)
model.eval()

# 查找第一个未处理的行索引
start_index = df['bert_score'].isna().idxmax()

with warnings.catch_warnings(record=True) as w:
	# 从未处理的最后一个位置开始处理DataFrame的每一行
	for idx, row in df.iloc[start_index:].iterrows():
		# 将instruction和input合并为一个字符串
		instruction_str = '' if pd.isnull(row['instruction']) else str(row['instruction'])
		input_str = '' if pd.isnull(row['input']) else str(row['input'])
		input_text = instruction_str + '\n' + input_str

		# 使用模型生成输出
		input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
		with torch.no_grad():
			output_ids = model.generate(
				input_ids, max_length=100000, do_sample=True, temperature=0.7, top_k=50,
				top_p=0.95, num_return_sequences=1
				)
		output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		df.loc[idx, 'alpaca_output'] = output_text

		# 使用BERTScore比较模型的输出和原始输出
		_, _, F1 = score([output_text], [row['output']], lang='en')
		df.loc[idx, 'bert_score'] = F1.item()

		# 保存DataFrame为csv文件
		df.to_csv(DF_PATH, index=False)

	print(w[-1].category)  # 这会打印最后一个警告的类型
