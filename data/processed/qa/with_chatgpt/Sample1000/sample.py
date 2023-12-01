# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("qa_combined_nonempty.csv")
df = df.head(1000)
df.drop('response_Llama-2-70b-chat', axis=1, inplace=True)
df.to_csv("qa_combined_nonempty_toy.csv", index=False)