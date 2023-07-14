# -*- coding: utf-8 -*-
from datasets import load_dataset, Dataset, DatasetDict

dataset: DatasetDict = load_dataset('tatsu-lab/alpaca_eval', 'alpaca_eval')
dataset: Dataset = dataset['eval']
dataset.to_csv('../data/alpaca_eval.csv')
