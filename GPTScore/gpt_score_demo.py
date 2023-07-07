# -*- coding: utf-8 -*-
import os

from GPTScore.gpt3_score import gpt3score

api_key = os.getenv("OPENAI_API_KEY")

gpt_score = gpt3score(
	input='Given a user query, find an answer to the query in an appropriate form. \n Query: What is the population of Japan?',
	output='According to the latest census, the population of Japan as of June 2020 is 126,476,461.',
	gpt3model='davinci003',
	api_key=api_key,
	)
print(gpt_score)
