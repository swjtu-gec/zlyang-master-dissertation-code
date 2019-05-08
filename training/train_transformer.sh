#!/usr/bin/env bash

## This script used to train Transformer NMT GEC model

set -e
set -x

source ../paths.sh

if [[ $# != 6 ]]; then
    echo "Usage: `basename $0` <dir to bin data> <GPU device id to use(e.g: 0)> <model_name(e.g: transformer_zh_char_random)> <random seed> <max tokens> <max sentences>"
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
    --arch transformer \
    --encoder-embed-dim=800 --decoder-embed-dim=800 \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --warmup-init-lr '1e-07' --warmup-updates 4000 \
    --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --num-workers=4 --skip-invalid-size-inputs-valid-test \
    --max-update 400000 \
    --max-tokens ${MAX_TOKENS} --max-sentences ${MAX_SENS} \
    --no-progress-bar --seed ${SEED}

