#!/bin/bash

DATASET=$1
GPUID=$2

#DATASET=/data/pengyu/EUR-Lex
#GPUID=7

if [ "$DATASET" = "/data/pengyu/EUR-Lex" ]; then
    #eurlex
    python prompt_learning.py --gpuid $GPUID --data_dir $DATASET \
    --epoch 30

    python data_augmentation.py --gpuid $GPUID --data_dir $DATASET \
    --checkpoint /data/pengyu/$DATASET/model/t5p.pth \
    --aug_num 2

    python filtering.py --gpuid $GPUID --data_dir $DATASET \
    --percentage 80 \
    --aug_num 2

    python run_xml.py --gpuid $GPUID --data_dir $DATASET \
    --head_num 396 \
    --valid_size 200 \
    --epoch 30 --batch 32 --lr 1e-4 \
    --swa --swa_warmup 10 --swa_step 200

elif [ "$DATASET" = "/data/pengyu/Wiki10-31K" ]; then
    #wikis
    python prompt_learning.py --gpuid $GPUID --data_dir $DATASET \
    --epoch 30

    python data_augmentation.py --gpuid $GPUID --data_dir $DATASET \
    --checkpoint /data/pengyu/$DATASET/model/t5p.pth \
    --aug_num 4

    python filtering.py --gpuid $GPUID --data_dir $DATASET \
    --percentage 80 \
    --aug_num 4

    python run_xml.py --gpuid $GPUID --data_dir $DATASET \
    --head_num 1135 \
    --valid_size 200 \
    --epoch 40 --batch 32 --lr 1e-4 \
    --swa --swa_warmup 10 --swa_step 300


elif [ "$DATASET" = "/data/pengyu/AmazonCat-13K" ]; then
    #amazoncat
    python prompt_learning.py --gpuid $GPUID --data_dir $DATASET \
    --epoch 5

    python data_augmentation.py --gpuid $GPUID --data_dir $DATASET \
    --checkpoint /data/pengyu/$DATASET/model/t5p.pth \
    --aug_num 2

    python filtering.py --gpuid $GPUID --data_dir $DATASET \
    --percentage 80 \
    --aug_num 2

    python run_xml.py --gpuid $GPUID --data_dir $DATASET \
    --head_num 3403 \
    --valid_size 200 \
    --epoch 5 --batch 16 --lr 1e-4 \
    --swa --swa_warmup 2 --swa_step 10000 --eval_step 20000
fi










