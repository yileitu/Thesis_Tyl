# -*- coding: utf-8 -*-
import pandas as pd

df_gpt4all = pd.read_csv("../../data/sampled/qa/gpt4all.csv")
df_dolly = pd.read_csv("../../data/sampled/qa/dolly.csv")
df_alpaca = pd.read_csv("../../data/sampled/qa/alpacafarm.csv")
df_baize = pd.read_csv("../../data/sampled/qa/baize.csv")
df_instinwild = pd.read_csv("../../data/sampled/qa/instinwild.csv")
df_fastchat = pd.read_csv("../../data/sampled/qa/fastchat.csv")


df_alpaca["category"] = None
df_baize["category"] = None
df_instinwild["category"] = None
df_fastchat["category"] = None

df_gpt4all = df_gpt4all[['input', 'response', 'category']]
df_dolly = df_dolly[['input', 'response', 'category']]
df_alpaca = df_alpaca[['input', 'response', 'category']]
df_baize = df_baize[['input', 'response', 'category']]
df_instinwild = df_instinwild[['input', 'response', 'category']]
df_fastchat = df_fastchat[['input', 'response', 'category']]

df_gpt4all['data_source'] = 'GPT4All'
df_dolly['data_source'] = 'Dolly'
df_alpaca['data_source'] = 'AlpacaFarm'
df_baize['data_source'] = 'Baize'
df_instinwild['data_source'] = 'InstructionWild'
df_fastchat['data_source'] = 'FastChat-Vicuna'

df = pd.concat([df_gpt4all, df_dolly, df_alpaca, df_baize, df_instinwild, df_fastchat], ignore_index=True)
df = df[['input', 'response', 'category', 'data_source']]
df.to_csv("../../data/qa.csv", index=False)
