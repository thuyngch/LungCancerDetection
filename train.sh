#!/bin/bash
set -e

# CUDA
CUDA_VERSION="cuda-10.0"
export PATH="/usr/local/${CUDA_VERSION}/bin:${PATH}"
export CUDADIR="/usr/local/${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Multiple trainings
python src/models/train.py \
    --ckpt "ckpt/attention1.0_focal_bs8_ep200/nodule3-classifier.ckpt" \
    --num_outputs 2 \
    --hidden_embedding 512 \
    --lr 1e-3 \
    --epoch 100 \
    --batch_size 8 \
    --use_pooling \
    --attention_ratio 1.0
    # --resume "ckpt/attention1.0_softmax_bs8_ep200/nodule3-classifier.ckpt9591" \
