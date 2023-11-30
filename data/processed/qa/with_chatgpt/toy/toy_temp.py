# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("qa_combined_nonempty.csv")
df = df.head(3)
df.to_csv("qa_combined_nonempty_toy.csv", index=False)