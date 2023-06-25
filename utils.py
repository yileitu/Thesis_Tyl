# -*- coding: utf-8 -*-
import os
import random
from typing import Final

import numpy as np
import torch

SEED: Final[int] = 21946520


def set_seed(seed: int = SEED) -> None:
	"""
	Set all seeds to make results reproducible

	:param seed: seed number
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)


def set_mtec_env() -> torch.device:
	"""
	Set environments for MTEC server

	:return: PyTorch device
	"""
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
	gpu_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
	print(f"... {len(gpu_list)} visible 'logical' GPUs: {gpu_list}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"... using {device}")

	return device
