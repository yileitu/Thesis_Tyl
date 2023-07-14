# -*- coding: utf-8 -*-
import pickle

from pandalm import EvaluationPipeline
from util.util_func import set_mtec_env

set_mtec_env()

pipeline = EvaluationPipeline(
	candidate_paths=["huggyllama/llama-7b", "facebook/opt-6.7b"],
	input_data_path="data/pipeline_only_instruction.json",
	output_data_path="/data-nfs/yilei/thesis/repo/PandaLM/pipeline_only_instruction-output.json",
	)

eval_results = pipeline.evaluate()
with open('alpaca_eval_pandalm_results.pkl', 'wb') as f:
    pickle.dump(eval_results, f)
print(eval_results)
