# -*- coding: utf-8 -*-
import os
import sys

import bert_score
import pandas as pd
from bleurt import score
from rouge import Rouge

from bart_score import BARTScorer
from util.util_func import set_gpu_env, set_seed

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

# def determine_predicted_answer(row):
# 	label_to_answer = {
# 		'<|Simple|>'   : row['response_pythia-2.8b'],
# 		'<|Middle|>'   : row['response_Llama-2-7b-chat'],
# 		'<|Difficult|>': row['response_Llama-2-13b-chat'],
# 		'<|Arduous|>'  : row['response_Llama-2-70b-chat'],
# 		}
# 	return label_to_answer.get(row['predicted_label'], None)
#
#
# def determine_predicted_moversocre(row):
# 	label_to_answer = {
# 		'<|Simple|>'   : row['response_pythia-2.8b_moverscore'],
# 		'<|Middle|>'   : row['response_Llama-2-7b-chat_moverscore'],
# 		'<|Difficult|>': row['response_Llama-2-13b-chat_moverscore'],
# 		'<|Arduous|>'  : row['response_Llama-2-70b-chat_moverscore'],
# 		}
# 	return label_to_answer.get(row['predicted_label'], None)
#
#
# def determine_oracle_answer(row):
# 	label_to_answer = {
# 		'<|Simple|>'   : row['response_pythia-2.8b'],
# 		'<|Middle|>'   : row['response_Llama-2-7b-chat'],
# 		'<|Difficult|>': row['response_Llama-2-13b-chat'],
# 		'<|Arduous|>'  : row['response_Llama-2-70b-chat'],
# 		}
# 	return label_to_answer.get(row['label'], None)
#
#
# def determine_oracle_moversocre(row):
# 	label_to_answer = {
# 		'<|Simple|>'   : row['response_pythia-2.8b_moverscore'],
# 		'<|Middle|>'   : row['response_Llama-2-7b-chat_moverscore'],
# 		'<|Difficult|>': row['response_Llama-2-13b-chat_moverscore'],
# 		'<|Arduous|>'  : row['response_Llama-2-70b-chat_moverscore'],
# 		}
# 	return label_to_answer.get(row['label'], None)

# Read merged data
LLMS = [
	'pythia-2.8b',
	'Llama-2-7b-chat',
	'Llama-2-13b-chat',
	'Llama-2-70b-chat']
DATA_PATH = '../data/predicted/temp2.0/qa_test_merged.csv'
df_qa_pred = pd.read_csv(DATA_PATH)

# # Create oracle and routing answers
# df_qa_pred['predicted_answer'] = df_qa_pred.apply(determine_predicted_answer, axis=1)
# df_qa_pred['predicted_moverscore'] = df_qa_pred.apply(determine_predicted_moversocre, axis=1)
# df_qa_pred['oracle_answer'] = df_qa_pred.apply(determine_oracle_answer, axis=1)
# df_qa_pred['oracle_moverscore'] = df_qa_pred.apply(determine_oracle_moversocre, axis=1)
# df_qa_pred.to_csv(DATA_PATH, index=False)

# Create random answers
# set_seed()
# df_qa_pred['random_response'] = df_qa_pred.apply(
# 	lambda row: random.choice(
# 		[
# 			row['response_pythia-2.8b'],
# 			row['response_Llama-2-7b-chat'],
# 			row['response_Llama-2-13b-chat'],
# 			row['response_Llama-2-70b-chat']]
# 		),
# 	axis=1
# 	)

ground_truths = df_qa_pred['response_ground_truth'].tolist()
predicted_answers = df_qa_pred['oracle_answer'].tolist()
set_gpu_env()
set_seed()

BLEURT_CKPT_PATH = '/data-nfs/yilei/bleurt/BLEURT-20'
bleurt_scorer = score.BleurtScorer(checkpoint=BLEURT_CKPT_PATH)

# BERTScore
_, _, F1 = bert_score.score(predicted_answers, ground_truths, lang='en')
df_qa_pred['oracle_bertscore'] = F1.numpy()
df_qa_pred.to_csv(DATA_PATH, index=False)
print('BERTScore done')

# BART
bart_scorer = BARTScorer(device='cuda')
bart_scores = bart_scorer.score(predicted_answers, ground_truths)
df_qa_pred['oracle_bart'] = bart_scores
df_qa_pred.to_csv(DATA_PATH, index=False)
print('BARTScore done')

# ROUGE
rouge = Rouge()
sys.setrecursionlimit(1024 * 1024)
rouge_scores = [
	rouge.get_scores(
		pa, gt
		)[0] for pa, gt in zip(
		predicted_answers, ground_truths
		)]
df_qa_pred['oracle_rouge'] = [score['rouge-l']['f'] for score in rouge_scores]
df_qa_pred.to_csv(DATA_PATH, index=False)
print('ROUGE done')

# BLEURT
bleurt_scores = bleurt_scorer.score(
	references=ground_truths,
	candidates=predicted_answers,
	batch_size=128
	)
df_qa_pred['oracle_bleurt'] = bleurt_scores
df_qa_pred.to_csv(DATA_PATH, index=False)
print('BLEURT done')

# for llm in LLMS:
# 	predicted_answers = df_qa_pred[f'response_{llm}'].tolist()
# 	print("Processing: ", llm)
#
# 	# BERTScore
# 	_, _, F1 = score(predicted_answers, ground_truths, lang='en')
# 	df_qa_pred[f'{llm}_bertscore'] = F1.numpy()
# 	df_qa_pred.to_csv(DATA_PATH, index=False)
# 	print('BERTScore done')
#
# 	# BART
# 	bart_scorer = BARTScorer(device='cuda')
# 	bart_scores = bart_scorer.score(predicted_answers, ground_truths)
# 	df_qa_pred[f'{llm}_bart'] = bart_scores
# 	df_qa_pred.to_csv(DATA_PATH, index=False)
# 	print('BARTScore done')
#
# 	# ROUGE
# 	rouge = Rouge()
# 	sys.setrecursionlimit(1024 * 1024)
# 	rouge_scores = [rouge.get_scores(pa, gt)[0] for pa, gt in zip(predicted_answers, ground_truths)]
# 	df_qa_pred[f'{llm}_rouge'] = [score['rouge-l']['f'] for score in rouge_scores]
# 	df_qa_pred.to_csv(DATA_PATH, index=False)
# 	print('ROUGE done')
#
# 	# BLEURT
# 	bleurt_scores = bleurt_scorer.score(references=ground_truths, candidates=predicted_answers, batch_size=128)
# 	df_qa_pred[f'{llm}_bleurt'] = bleurt_scores
# 	df_qa_pred.to_csv(DATA_PATH, index=False)
# 	print('BLEURT done')
