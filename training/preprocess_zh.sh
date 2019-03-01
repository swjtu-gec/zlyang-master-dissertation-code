#!/bin/bash

set -x
set -e

PROCESSED_DIR=zh-processed
mkdir -p ${PROCESSED_DIR}
use_bpe=true
remove_same=false
source ../paths.sh
NLPCC2018_DATA_DIR=${DATA_DIR}/zh-blcu-nlpcc2018
# path to sub-word nmt
SUBWORD_NMT=${SOFTWARE_DIR}/subword-nmt
BPE_MODEL_DIR=models/zh_bpe_model
BPE_MODEL=train.bpe.model
bpe_operations=30000

# paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=${NLPCC2018_DATA_DIR}/train.tok
dev_data_prefix=${NLPCC2018_DATA_DIR}/dev.tok

if [[ "${use_bpe}" == 'true' ]]; then
    # sub-word segmentation
    mkdir -p ${BPE_MODEL_DIR}
    cat ${train_data_prefix}.${trg_ext} | ${SUBWORD_NMT}/learn_bpe.py -s ${bpe_operations} > ${BPE_MODEL_DIR}/${BPE_MODEL}
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/${BPE_MODEL} < ${train_data_prefix}.${src_ext} > ${PROCESSED_DIR}/train.all.src
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/${BPE_MODEL} < ${train_data_prefix}.${trg_ext} > ${PROCESSED_DIR}/train.all.trg
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/${BPE_MODEL} < ${dev_data_prefix}.${src_ext} > ${PROCESSED_DIR}/dev.src
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/${BPE_MODEL} < ${dev_data_prefix}.${trg_ext} > ${PROCESSED_DIR}/dev.trg
else
    less ${train_data_prefix}.${src_ext} > ${PROCESSED_DIR}/train.all.src
    less ${train_data_prefix}.${trg_ext} > ${PROCESSED_DIR}/train.all.trg
    less ${dev_data_prefix}.${src_ext} > ${PROCESSED_DIR}/dev.src
    less ${dev_data_prefix}.${trg_ext} > ${PROCESSED_DIR}/dev.trg
fi

if [[ "${remove_same}" == 'true' ]]; then
    # getting annotated sentence pairs only
    python ${SCRIPTS_DIR}/get_diff.py  ${PROCESSED_DIR}/train.all src trg > ${PROCESSED_DIR}/train.annotated.src-trg
    cut -f1  ${PROCESSED_DIR}/train.annotated.src-trg > ${PROCESSED_DIR}/train.src
    cut -f2  ${PROCESSED_DIR}/train.annotated.src-trg > ${PROCESSED_DIR}/train.trg
else
    less ${PROCESSED_DIR}/train.all.src > ${PROCESSED_DIR}/train.src
    less ${PROCESSED_DIR}/train.all.trg > ${PROCESSED_DIR}/train.trg
fi


#########################
# pre-processing
python ${FAIRSEQPY}/preprocess.py \
    --source-lang src --target-lang trg \
    --trainpref ${PROCESSED_DIR}/train \
    --validpref ${PROCESSED_DIR}/dev \
    --testpref  ${PROCESSED_DIR}/dev \
    --nwordssrc 37000 --nwordstgt 37000 \
    --destdir ${PROCESSED_DIR}/bin

