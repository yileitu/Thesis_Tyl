# -*- coding: utf-8 -*-
import pandas as pd

from util.struct import Task
from util.util_func import save_df_to_csv

TASK: Task = Task.TF

if TASK == Task.MC:
	task_name = 'mc'
elif TASK == Task.TF:
	task_name = 'tf'
elif TASK == Task.QA:
	task_name = 'qa'
else:
	raise ValueError(f'Unknown task: {TASK}')

question_path = f'../data/output/{task_name}/{task_name}.csv'
label_path = f'../data/processed/{task_name}/unfiltered/labeled/{task_name}_labeled_answers.csv'
save_path = f'../data/labeled/{task_name}_labeled.csv'

df_question = pd.read_csv(question_path)
df_labeled = pd.read_csv(label_path)
labeled_selected_columns = ['id', 'label']
df_labeled = df_labeled[labeled_selected_columns]
merged_df = pd.merge(df_question, df_labeled, on='id')
if TASK == Task.MC:
	selected_columns = ['id', 'passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'option_E',
	                    'answer', 'subject', 'data_source', 'label']
elif TASK == Task.TF:
	selected_columns = ['id', 'passage', 'question', 'answer', 'label']
merged_df = merged_df[selected_columns]
save_df_to_csv(df=merged_df, path=save_path, index=False)
