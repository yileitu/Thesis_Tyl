# -*- coding: utf-8 -*-
import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.Model.list())
# response = openai.ChatCompletion.create(
# 	model="gpt-3.5-turbo",
# 	messages=[
# 		{"role": "user", "content": "You are a annotator who knows the economics well."},
# 		]
# 	)

prompt = "You are a annotator who knows the economics well."
response = openai.Completion.create(
	engine="text-davinci-003",
	prompt=prompt,
	max_tokens=0,
	temperature=0,
	logprobs=0,
	echo=True,
	stop='\n',
	n=None,
	)
print(response.keys())
print(response)
