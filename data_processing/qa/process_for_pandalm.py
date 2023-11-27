# -*- coding: utf-8 -*-
import json
from typing import Dict, List

import pandas as pd

# Read the CSV file
df: pd.DataFrame = pd.read_csv('../data/alpaca_eval.csv')

# Create an empty list to store the data
data: List[Dict[str, str]] = []

# Loop over each row in the CSV file
for index, row in df.iterrows():
	# Add the value from the 'instruction' column to the dictionary
	data_dict: Dict[str, str] = {"instruction": row['instruction'], "input": ""}
	# Add the dictionary to the list
	data.append(data_dict)

# Convert the list into JSON format
json_data: str = json.dumps(data, ensure_ascii=False)

# Save the JSON data to a file
with open('../data/AlpacaEval_for_pandalm.json', 'w', encoding='utf-8') as f:
	f.write(json_data)
