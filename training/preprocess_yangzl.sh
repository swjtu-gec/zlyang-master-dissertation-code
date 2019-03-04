#!/bin/bash

set -x
set -e

source ../paths.sh

if [[ $# != 8 ]]; then
    echo "Usage: `basename $0` <dir to processed data(e.g: zh-bpe-processed-fusion)> <whether to use BPE(e.g: true)> <whether to remove same sen pairs(e.g: false)> <dir to split train and dev token-level data(e.g: zh-fusion)> <dir to BPE model(e.g: models/zh_bpe_model_fusion)> <BPE operations(e.g: 30000)> <src vocab size(e.g: 37000)> <tgt vocab size(e.g: 37000)>"
    exit -1
fi

PROCESSED_DIR=$1
mkdir -p ${PROCESSED_DIR}
use_bpe=$2
remove_same=$3
SPLIT_TOKEN_DATA_DIR=$4
BPE_MODEL_DIR=$5
bpe_operations=$6
src_vocab_size=$7
tgt_vocab_size=$8

# path to sub-word nmt
SUBWORD_NMT=${SOFTWARE_DIR}/subword-nmt
BPE_MODEL=train.bpe.model
# paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=${SPLIT_TOKEN_DATA_DIR}/train.tok
dev_data_prefix=${SPLIT_TOKEN_DATA_DIR}/dev.tok

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
    --nwordssrc ${src_vocab_size} --nwordstgt ${tgt_vocab_size} \
    --destdir ${PROCESSED_DIR}/bin

