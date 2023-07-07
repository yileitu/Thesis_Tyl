# -*- coding: utf-8 -*-
import pandas as pd

from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
from utils import set_mtec_env, set_seed

# Set environments
set_seed()
device = set_mtec_env()

df = pd.read_csv('data/alpaca_sample50.csv')
if 'mover_score' not in df.columns:
	df['mover_score'] = None

# MoverScore
idf_dict_ref = get_idf_dict(df['alpaca_output'])
refs = df['output'].tolist()
idf_dict_hyp = get_idf_dict(df['output'])
hyps = df['alpaca_output'].tolist()
scores = word_mover_score(refs=refs, hyps=hyps, idf_dict_ref=idf_dict_ref, idf_dict_hyp=idf_dict_hyp)
df['mover_score'] = scores
df.to_csv("data/alpaca_sample50.csv", index=False)
