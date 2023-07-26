# -*- coding: utf-8 -*-
from typing import Dict, List

import pandas as pd
from datasets import DatasetDict, load_dataset

SEED: int = 21946520
ANSWER_MAP: Dict[int, str] = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
SUBJECTS: List[str] = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
                       'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
                       'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
                       'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
                       'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
                       'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
                       'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
                       'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                       'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
                       'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
                       'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
                       'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
                       'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
                       'sociology', 'us_foreign_policy', 'virology', 'world_religions']
# SUBJECTS: List[str] = ['abstract_algebra', 'anatomy']
NUM_SAMPLES_EACH_SUB: int = 500

df = pd.DataFrame()

for subject in SUBJECTS:
	dataset: DatasetDict = load_dataset('cais/mmlu', subject)
	sub_df = pd.DataFrame(dataset['auxiliary_train'])
	sampled_sub_df = sub_df.sample(n=NUM_SAMPLES_EACH_SUB, random_state=SEED)
	df = pd.concat([df, sampled_sub_df], ignore_index=True)

unexpected_values = df.loc[~df['answer'].isin(list(ANSWER_MAP.keys())), 'answer'].unique()
if len(unexpected_values) > 0:
	print("... Found unexpected values in 'answer' column:", unexpected_values)
else:
	print("... No unexpected values found in 'answer' column.")

df[['option_A', 'option_B', 'option_C', 'option_D']] = pd.DataFrame(df['choices'].tolist(), index=df.index)
df['answer'] = df['answer'].map(ANSWER_MAP)
df = df.drop('choices', axis=1)
df['passage'] = None
df = df[['passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'answer']]
df.to_csv("../../data/sampled/exam/mmlu.csv", index=False)
