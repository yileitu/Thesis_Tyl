# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MyArguments:
	"""
	Custom arguments for fine-tuning.
	"""
	max_length: Optional[int] = field(
		default=1024,
		metadata={"help": "The maximum input sequence length after tokenization."},
		)
	data_dir: str = field(
		default=None,
		metadata={"help": "Where to read labeled data."}
		)
	task: Optional[str] = field(
		default='ner',
		metadata={"help": "Tasks, one or more of {mc, tf, qa}."},
		)
	n_gpu: Optional[int] = field(
		default=1,
		metadata={"help": "Number of GPUs to use."},
		)
