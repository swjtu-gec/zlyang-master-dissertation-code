#!/usr/bin/env bash

## This script used to train LSTM seq2seq NMT GEC model

set -e
set -x

source ../paths.sh

if [[ $# != 6 ]]; then
    echo "Usage: `basename $0` <dir to bin data> <GPU device id to use(e.g: 0)> <model_name(e.g: lstm_zh_char_random)> <random seed> <max tokens> <max sentences>"
    exit -1
fi

DATA_BIN_DIR=$1
gpu_to_use=$2
model_name=$3
SEED=$4
MAX_TOKENS=$5
MAX_SENS=$6

OUT_DIR=models/${model_name}/model${SEED}/
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES="${gpu_to_use}" python ${FAIRSEQPY}/train.py \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a lstm \
    --encoder-embed-dim=1000 --decoder-embed-dim=1000 \
    --encoder-hidden-size=500 --decoder-hidden-size=1000 \
    --encoder-bidirectional --encoder-layers=2 --decoder-layers=2 \
    --decoder-out-embed-dim=1000 \
    --lr-scheduler reduce_lr_on_plateau --clip-norm 0.1 \
    --num-workers=4 --skip-invalid-size-inputs-valid-test \
    --max-epoch 100 \
    --max-tokens ${MAX_TOKENS} --max-sentences ${MAX_SENS} \
    --no-progress-bar --seed ${SEED}

