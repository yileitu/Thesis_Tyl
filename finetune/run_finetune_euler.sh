#!/bin/bash -l

#SBATCH -n 4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=a100_80gb:1

module load eth_proxy
module load gcc/9.3.0
module load cuda/11.7.0
conda activate thesis
wandb login

python3 finetune.py \
  --n_gpu 1 \
  --fp16 \
  --task mc \
  --data_dir ../data/templated/ \
  --output_dir results/ \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --save_strategy epoch \
  --evaluation_strategy epoch
