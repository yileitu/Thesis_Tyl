# -*- coding: utf-8 -*-

import pandas as pd

mc_data = pd.read_csv('../data/output/mc/mc.csv')

# 对数据按照'data_source'和'subject'进行分组
groups = mc_data.groupby(['data_source', 'subject'])

nonempty_option_e_list = []
empty_option_e_list = []
for name, group in groups:
	if group['option_E'].notna().any():
		nonempty_option_e_list.append(name)
	else:
		empty_option_e_list.append(name)

print(f'Nonempty Option E List: {nonempty_option_e_list}')
print(f'Empty Option E lList: {empty_option_e_list}')
