#!/bin/bash
## This script used to train fconv model

set -x
set -e

source ../paths.sh

if [[ $# != 6 ]]; then
    echo "Usage: `basename $0` <dir to bin data> <GPU device id to use(e.g: 0)> <model_name(e.g: fconv_zh_bpe)> <random seed> <max tokens> <max sentences>"
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
    -a fconv \
    --fp16 --fp16-init-scale=8 \
    --num-workers=4 --skip-invalid-size-inputs-valid-test \
    --encoder-embed-dim 500 \
    --decoder-embed-dim 500 --decoder-out-embed-dim 500 \
    --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' \
    --dropout='0.2' --clip-norm=0.1 \
    --optimizer nag --momentum 0.99 \
    --lr-scheduler=reduce_lr_on_plateau --lr=0.25 --lr-shrink=0.1 --min-lr=1e-4 \
    --max-epoch 100 \
    --max-tokens ${MAX_TOKENS} --max-sentences ${MAX_SENS} \
    --no-progress-bar --seed ${SEED}

