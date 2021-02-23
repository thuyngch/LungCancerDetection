#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

#### OLD MODEL ####
# python src/models/test.py \
#     --ckpt "./ckpt/attention1.0_softmax_bs8_softsoft/" \
#     --test_data "src/data/test_s100_r50.h5" \
#     --num_outputs 2 \
#     --hidden_embedding 256 \
#     --use_pooling \
#     --attention_ratio 1.0


#### NEW MODEL ####
python src/models/test.py \
    --ckpt "/home/cybercore/tank/LungCancerDetection/ckpt3/attention0.0_softmax_bs8_ep50_trainvaltest/" \
    --test_data "src/data/test.h5" \
    --num_outputs 2 \
    --hidden_embedding 256 \
    --use_pooling \
    --attention_ratio 0.0
