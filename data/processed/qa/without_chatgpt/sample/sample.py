# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("qa_combined_nonempty.csv")
df_sampled = df.sample(n=10, random_state=21946520)
df_sampled.to_csv("qa_combined_nonempty.csv", index=False)
