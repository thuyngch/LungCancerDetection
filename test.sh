#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-10.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=1

# Testings

python src/models/test.py \
    --ckpt "ckpt/attention1.0_softmax_bs8_ep200/" \
    --num_outputs 2 \
    --hidden_embedding 512 \
    --use_pooling \
    --attention_ratio 1.0 #--tta

# python src/models/test.py \
#     --ckpt "ckpt/attention1.0_triplet128_bs128_ep200/" \
#     --num_outputs 128 \
#     --hidden_embedding 256 \
#     --use_pooling \
#     --use_triplet \
#     --attention_ratio 1.0
