# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("test_data.csv")
df_sampled = df.sample(n=10, random_state=21946520)
df_sampled.to_csv("test_data.csv", index=False)
