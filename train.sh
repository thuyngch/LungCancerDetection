#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-9.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Multiple trainings
python src/models/train.py \
    --ckpt "ckpt/attention0.25_softmax_bs8/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --lr 5e-4 \
    --epoch 50 \
    --batch_size 8 \
    --attention_ratio 0.25

python src/models/train.py \
    --ckpt "ckpt/attention0.75_softmax_bs8/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --lr 5e-4 \
    --epoch 50 \
    --batch_size 8 \
    --attention_ratio 0.75

python src/models/train.py \
    --ckpt "ckpt/attention1.0_softmax_bs8/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --lr 5e-4 \
    --epoch 50 \
    --batch_size 8 \
    --attention_ratio 1.0
