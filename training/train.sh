#!/bin/bash

set -x
set -e

source ../paths.sh

SEED=1000
DATA_BIN_DIR=processed/bin

OUT_DIR=models/mlconv/model${SEED}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES="0" python ${FAIRSEQPY}/train.py \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a fconv \
    --encoder-embed-dim 500 \
    --decoder-embed-dim 500 --decoder-out-embed-dim 500 \
    --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 \
    --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' \
    --momentum 0.99 --max-epoch 100 \
    --max-tokens 1000 --max-sentences 10 \
    --no-progress-bar --seed ${SEED}

