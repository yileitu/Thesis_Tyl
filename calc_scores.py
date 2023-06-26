# -*- coding: utf-8 -*-
import warnings

import pandas as pd

from utils import set_mtec_env, set_seed

# Set environments
set_seed()
warnings.filterwarnings("ignore", category=UserWarning)
device = set_mtec_env()

df = pd.read_csv('data/alpaca_sample50.csv')
if 'mover_score' not in df.columns:
	df['mover_score'] = None

# # MoverScore
# idf_dict_ref = get_idf_dict(df['alpaca_output'])
# refs = df['output'].tolist()
# idf_dict_hyp = get_idf_dict(df['output'])
# hyps = df['alpaca_output'].tolist()
# scores = word_mover_score(refs=refs, hyps=hyps, idf_dict_ref=idf_dict_ref, idf_dict_hyp=idf_dict_hyp)
# print(scores)
