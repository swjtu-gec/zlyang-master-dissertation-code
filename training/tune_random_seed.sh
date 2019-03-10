#!/usr/bin/env bash

## This script used to select best random num seed.

set -x
set -e

source ../paths.sh


full_process_sh=${BASE_DIR}/training/full_process.sh

if [[ $# -eq 13 ]]; then
    BIN_DATA_DIR=$1
    BPE_MODEL_DIR=$2
    EMBED_URL=$3
    test_input=$4
    gold_edit=$5
    m2scorer_url=$6
    GPUs_used_training=$7
    GPUs_used_inference=$8
    model_name=$9
    MAX_TOKENS=${10}
    MAX_SENS=${11}
    try_random_seed=${12}
    want_ensemble=${13}
else
    echo "Usage: `basename $0` <dir to bin data> <dir to BPE model> <embed_file_url> <test input> <url to gold edit> <url to m2scorer script> <GPU device id to use in training(e.g: '0, 1, 2')> <GPU device id used in test)> <model name(e.g: fconv_zh_bpe_embed_fusion)> <max tokens> <max sentences> <a set of random seed to try, space separate(e.g: '1 1000 2000 3000')> <whether to use entire model dir to ensemble decoding(e.g: true or false)>"
    exit -1
fi


for rdm_seed in ${try_random_seed}
do
    ${full_process_sh} ${BIN_DATA_DIR} ${BPE_MODEL_DIR} ${EMBED_URL} ${test_input} ${gold_edit} ${m2scorer_url} ${GPUs_used_training} ${GPUs_used_inference} ${model_name} ${MAX_TOKENS} ${MAX_SENS} ${rdm_seed} ${want_ensemble}
done

