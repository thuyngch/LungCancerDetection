#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-10.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=1

# Multiple trainings
# python src/models/train.py \
#     --ckpt "ckpt/attention1.0_softmax_bs8_ep200_trainval/nodule3-classifier.ckpt" \
#     --train_data "src/data/train_val.h5" \
#     --valid_data "src/data/test.h5" \
#     --num_outputs 2 \
#     --hidden_embedding 512 \
#     --lr 1e-4 \
#     --epoch 200 \
#     --batch_size 8 \
#     --use_pooling \
#     --attention_ratio 1.0

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
