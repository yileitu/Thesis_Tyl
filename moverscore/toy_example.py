# -*- coding: utf-8 -*-
from examples.example import ref_score

refs = ['The dog bit the man.', 'The dog had bit the man.']
sys = 'The dog bit the man.'

moverscore = ref_score(sys, refs)
print(moverscore)