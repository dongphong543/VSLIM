#!/bin/bash

# Example training script for VSLIM model
# Usage: bash run_vslim_train.sh

python main.py \
  --task vped \
  --model_dir ./outputs/vslim_model \
  --data_dir ./data \
  --intent_label_file intent_label.txt \
  --slot_label_file slot_label.txt \
  --model_type phobert \
  --multi_intent 1 \
  --tag_intent 1 \
  --cls_token_cat 1 \
  --intent_attn 1 \
  --num_mask 4 \
  --seed 42 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_len 128 \
  --learning_rate 5e-5 \
  --num_train_epochs 50 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --warmup_steps 0 \
  --dropout_rate 0.1 \
  --logging_steps 50 \
  --save_steps 200 \
  --do_train \
  --do_eval \
  --ignore_index -100 \
  --slot_loss_coef 2.0 \
  --tag_intent_coef 1.0

