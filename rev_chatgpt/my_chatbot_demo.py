# -*- coding: utf-8 -*-
import json
import os

from rev_chatgpt.my_chatbot import MyChatbot

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REV_CHATGPT_CONFIG_PATH = os.path.join(CUR_DIR, 'rev_chatgpt_config.json')

with open(REV_CHATGPT_CONFIG_PATH, 'r') as f:
	rev_chatgpt_config = json.load(f)

rev_chatgpt = MyChatbot(config=rev_chatgpt_config)
prompt = "how many beaches does portugal have?"
response = rev_chatgpt.get_response(prompt)
print(response)

# for data in rev_chatgpt.ask(prompt):
# 	print(data.keys())
