# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("qa_combined_nonempty.csv")
df_gpt4all = df[df['data_source'] == 'GPT4All']
df_sampled = df_gpt4all.sample(n=1000, random_state=21946520)
df_sampled.drop('response_Llama-2-70b-chat', axis=1, inplace=True)
df_sampled.to_csv("qa_combined_nonempty.csv", index=False)
