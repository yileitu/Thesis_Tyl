python3 finetune.py \
  --n_gpu 1 \
  --task mc \
  --data_dir ../data/templated/ \
  --output_dir results/ \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --save_strategy epoch \
  --evaluation_strategy epoch \

