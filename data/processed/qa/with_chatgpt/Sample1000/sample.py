# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("qa_combined_nonempty.csv")
df = df.sample(n=3000, random_state=42)
df.drop('response_Llama-2-70b-chat', axis=1, inplace=True)
df.to_csv("qa_combined_nonempty_toy.csv", index=False)
