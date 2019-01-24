#!/bin/bash

set -x
set -e

source ../paths.sh

SEED=1000
DATA_BIN_DIR=toy-processed-bin

OUT_DIR=models/toy/model${SEED}/
test -e ${OUT_DIR} && rm -fr ${OUT_DIR}
test ! -d ${OUT_DIR} && mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${FAIRSEQPY}/train.py \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a fconv_elmo \
    --encoder-embed-dim 1124 \
    --decoder-embed-dim 1124 --decoder-out-embed-dim 100 \
    --encoder-layers '[(100,3)] * 1' --decoder-layers '[(100,3)] * 1' \
    --dropout 0.0 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --lr-shrink 0.2 \
    --max-epoch 100 --max-tokens 1000 --max-sentences 10 \
    --no-progress-bar --seed ${SEED} \
    --use-other-embed --num-output-repr 1 \
    --merge-mode "concat" --token-embed-dim 100

