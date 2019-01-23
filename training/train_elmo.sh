#!/bin/bash

set -x
set -e

source ../paths.sh

SEED=1000
DATA_BIN_DIR=processed/bin

OUT_DIR=models/mlconv_elmo/model${SEED}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${FAIRSEQPY}/train.py \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a fconv_elmo \
    --clip-norm 0.1 --min-lr 1e-4 \
    --max-epoch 100 --max-tokens 1000 --max-sentences 10 \
    --no-progress-bar --seed ${SEED}

