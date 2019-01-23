#!/usr/bin/env bash

set -x

source ../paths.sh

PROCESSED_DIR=toy-processed-bin
test -e ${PROCESSED_DIR} && rm -fr ${PROCESSED_DIR}
test ! -d ${PROCESSED_DIR} && mkdir -p ${PROCESSED_DIR}

python ${FAIRSEQPY}/preprocess.py \
    --source-lang src --target-lang trg \
    --trainpref ${DATA_DIR}/toy/train \
    --validpref ${DATA_DIR}/toy/train \
    --testpref  ${DATA_DIR}/toy/train \
    --destdir ${PROCESSED_DIR}
