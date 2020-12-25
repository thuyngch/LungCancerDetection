#!/bin/bash
set -e

# Testings
# python src/models/test.py \
#     --ckpt "./ckpt/triplet512all0.1_ep30/" \
#     --num_outputs 512 \
#     --hidden_embedding 512 \
#     --use_pooling \
#     --use_triplet \
#     --attention_ratio 0.1

python src/models/test.py \
    --ckpt "./ckpt/attention1.0_softmax_bs8_softsoft/" \
    --num_outputs 2 \
    --hidden_embedding 256 \
    --use_pooling \
    --attention_ratio 1.0

# python src/models/test.py \
#     --ckpt "./ckpt/attention1.0_softmax_bs8_ep200_triplettest/" \
#     --num_outputs 2 \
#     --hidden_embedding 512 \
#     --use_pooling \
#     --attention_ratio 1.0

# python src/models/test.py \
#     --ckpt "./ckpt/attention1.0_softmax_bs8_ep50_trainvaltest/" \
#     --num_outputs 2 \
#     --hidden_embedding 512 \
#     --use_pooling \
#     --attention_ratio 1.0
