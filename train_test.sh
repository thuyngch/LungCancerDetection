#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0


for lr in 0.5e-4 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3
do
    for hidden_embedding in 128 256 512 1024
    do 
        for batch_size in 4 8 16
        do
            for attention_ratio in 0 0.1 0.25 0.5 0.75 1.0
            do 
                # hidden_embedding=256
                epoch=50
                # batch_size=8
                # attention_ratio=1.0
                python src/models/train.py \
                    --ckpt "ckpt/attention1.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier.ckpt" \
                    --train_data "src/data/train_s100.h5" \
                    --valid_data "src/data/val_s100.h5" \
                    --num_outputs 2 \
                    --hidden_embedding $hidden_embedding \
                    --lr $lr \
                    --epoch $epoch \
                    --batch_size $batch_size \
                    --use_pooling \
                    --attention_ratio $attention_ratio


                python src/models/test.py \
                    --ckpt "ckpt/attention1.0_softmax_bs8_ep50_trainvaltest/" \
                    --test_data "src/data/test_s100.h5" \
                    --num_outputs 2 \
                    --hidden_embedding $hidden_embedding \
                    --use_pooling \
                    --lr $lr \
                    --epoch $epoch \
                    --batch_size $batch_size \
                    --attention_ratio $attention_ratio
            done
        done
    done
done