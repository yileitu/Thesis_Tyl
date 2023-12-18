#!/bin/bash -l

#SBATCH -A es_sachan
#SBATCH -n 8
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
module load gcc/9.3.0
module load cuda/11.7.0
conda activate thesis

python3 toy_test.py