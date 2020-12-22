#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python src/models/train.py \
    --ckpt "ckpt/attention1.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier.ckpt" \
    --train_data "src/data/train_val_test.h5" \
    --valid_data "src/data/test.h5" \
    --num_outputs 2 \
    --hidden_embedding 512 \
    --lr 1e-4 \
    --epoch 50 \
    --batch_size 8 \
    --use_pooling \
    --attention_ratio 1.0
