# -*- coding: utf-8 -*-
import pandas as pd

question_path = '/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/output/mc/mc.csv'
label_path = '/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/mc/unfiltered/labeled/mc_labeled_answers.csv'
save_path = '/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/clean/mc_labeled.csv'

df_question = pd.read_csv(question_path)
df_labeled = pd.read_csv(label_path)
labeled_selected_columns = ['id', 'label']
df_labeled = df_labeled[labeled_selected_columns]
merged_df = pd.merge(df_question, df_labeled, on='id')
selected_columns = ['id', 'passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'option_E',
                    'answer', 'subject', 'data_source', 'label']
merged_df = merged_df[selected_columns]
merged_df.to_csv(save_path, index=False)
