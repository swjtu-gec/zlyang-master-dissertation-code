#!/usr/bin/env bash

## This script used to select best `long`, `low` and `high`.

set -x
set -e

source ../paths.sh

one_script_to_run_all_sh=${BASE_DIR}/training/one_script_to_run_all.sh

if [[ $# -eq 11 ]]; then
    model_level=$1
    params_file=$2
    src_vocab_size=$3
    trg_vocab_size=$4
    GPU_used_training=$5
    GPU_used_inference=$6
    MAX_TOKENS=$7
    MAX_SENS=$8
    random_seed=$9
    want_ensemble=${10}
    force_redo_remove_same_and_seg=${11}
else
    echo "Usage: `basename $0` <model level, e.g: bpe, char, word> <parameters file of short, long, low and high> <src_vocab_size> <trg_vocab_size> <GPU device id to use in training(e.g: 0)> <GPU device id used in test)> <max tokens> <max sentences> <random seed> <whether to use entire model dir to ensemble decoding(e.g: true or false)> <whether to force redo remove same and segmentation(e.g: false)>"
    exit -1
fi

if [[ ! -f ${params_file} ]]; then
    echo "not found parameters file, got $params_file"
    exit -2
fi

cat ${params_file} | while read line
do
    line=${line//[,\t]/' '}
    arr=(${line})
    short=${arr[0]}
    long=${arr[1]}
    low=${arr[2]}
    high=${arr[3]}
    echo "$short $long $low $high"
    ${one_script_to_run_all_sh} ${model_level} ${low} ${high} ${short} ${long} \
        ${src_vocab_size} ${trg_vocab_size} ${GPU_used_training} ${GPU_used_inference} \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${force_redo_remove_same_and_seg}
done

