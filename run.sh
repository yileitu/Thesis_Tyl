#!/bin/bash -l

#SBATCH -A es_sachan
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=v100:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10240

module load eth_proxy
module load gcc/8.2.0
conda activate thesis

python3 llm_response_eval.py
