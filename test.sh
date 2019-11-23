#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-9.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Testings
python src/models/test.py \
    --ckpt "ckpt/attention0.25_softmax_bs8_ep100/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --attention_ratio 0.25

python src/models/test.py \
    --ckpt "ckpt/attention0.5_softmax_bs8_ep100/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --attention_ratio 0.5

python src/models/test.py \
    --ckpt "ckpt/attention0.75_softmax_bs8_ep100/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --attention_ratio 0.75

python src/models/test.py \
    --ckpt "ckpt/attention1.0_softmax_bs8_ep100/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --attention_ratio 1.0