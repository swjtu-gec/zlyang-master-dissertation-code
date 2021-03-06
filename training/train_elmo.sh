#!/bin/bash

set -x
set -e

source ../paths.sh

SEED=1000
DATA_BIN_DIR=processed/bin

OUT_DIR=models/fconv_elmo/model${SEED}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${FAIRSEQPY}/train.py \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a fconv_elmo \
    --clip-norm 0.1 --min-lr 1e-4 \
    --max-epoch 100 --max-tokens 5000 --max-sentences 50 \
    --no-progress-bar --seed ${SEED} \
    --use-other-embed --num-output-repr 1 \
    --merge-mode "concat" --token-embed-dim 500
