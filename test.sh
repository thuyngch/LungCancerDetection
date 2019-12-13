#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-10.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Testings
python src/models/test.py \
    --ckpt "/home/cybercore/thuync/LungCancerDetection/ckpt/attention1.0_softmax_bs8_ep200_trainval/" \
    --num_outputs 2 \
    --hidden_embedding 512 \
    --use_pooling \
    --attention_ratio 1.0 #--tta
