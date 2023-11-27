# -*- coding: utf-8 -*-
import pandas as pd

from util.struct import Task

TASK: Task = Task.MC

if TASK == Task.MC:
	df_labeled_path = '../data/clean/mc_labeled.csv'
	template = '/prompt_template/mc_template_with_passage_and_option_e.txt'

df = pd.read_csv(df_labeled_path)


# Function to fill in the template with values from a row
def fill_template(row, template):
	return template.format(
		passage=row['passage'],
		question=row['question'],
		option_A=row['option_A'],
		option_B=row['option_B'],
		option_C=row['option_C'],
		option_D=row['option_D'],
		option_E=row['option_E']
		)


df['filled_template'] = df.apply(fill_template, axis=1, args=(template,))
