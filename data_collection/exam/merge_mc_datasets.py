# -*- coding: utf-8 -*-
import pandas as pd

df_agieval = pd.read_csv("../../data/sampled/exam/agieval.csv")
df_mmlu = pd.read_csv("../../data/sampled/exam/mmlu.csv")
df_openbookqa = pd.read_csv("../../data/sampled/exam/openbookqa.csv")

# Add a 'data_source' column to each DataFrame
df_agieval['data_source'] = 'AGIEval'
df_mmlu['data_source'] = 'MMLU'
df_openbookqa['data_source'] = 'OpenBookQA'

# Add missing 'option_E' column to df_mmlu and df_openbookqa
df_mmlu['option_E'] = None
df_openbookqa['option_E'] = None

# Ensure column order consistency
cols = df_agieval.columns.tolist()
df_mmlu = df_mmlu[cols]
df_openbookqa = df_openbookqa[cols]

# Combine the datasets
df_combined = pd.concat([df_agieval, df_mmlu, df_openbookqa], ignore_index=True)
df_combined.to_csv("../../data/archive/mc.csv", index=False)
