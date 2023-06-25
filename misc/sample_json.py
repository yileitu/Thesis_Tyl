# -*- coding: utf-8 -*-
import json
import random

import pandas as pd

# 读取json文件并解析为Python列表
with open('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/alpaca_data.json', 'r') as f:
    data = json.load(f)

# 随机抽样
sample_size = min(50, len(data))
sample = random.sample(data, sample_size)

# 将抽取的样本转换为DataFrame
df = pd.DataFrame(sample)

# 将DataFrame保存为csv文件
df.to_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/alpaca_sample50.csv', index=False)
