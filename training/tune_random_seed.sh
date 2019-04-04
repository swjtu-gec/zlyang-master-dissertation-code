#!/usr/bin/env bash

## This script used to select best random num seed.

set -x
set -e

source ../paths.sh

one_script_to_run_all_sh=${BASE_DIR}/training/one_script_to_run_all.sh

if [[ $# -eq 17 ]]; then
    model_arch=$1
    model_level=$2
    which_pretrained_embed=$3
    fusion_mode=$4
    short=$5
    long=$6
    low=$7
    high=$8
    src_vocab_size=$9
    trg_vocab_size=${10}
    GPU_used_training=${11}
    GPU_used_inference=${12}
    MAX_TOKENS=${13}
    MAX_SENS=${14}
    try_random_seed=${15}
    want_ensemble=${16}
    force_redo_remove_same_and_seg=${17}
else
    echo "Usage: `basename $0` <model arch, e.g: fconv, lstm, transformer> <model level, e.g: bpe, char, word> <use which pre-trained token embeddings, e.g: random, word2vec, wang2vec, cw2vec> <fusion mode: 1: nlpcc_betterseg; 2: nlpcc_betterseg+HSK; 3: nlpcc_betterseg+HSK+BLCU> <short, e.g: 1> <long, e.g: 100> <low, e.g: 0.0> <high, e.g: 9.0> <src_vocab_size> <trg_vocab_size> <GPU device id to use in training(e.g: 0)> <GPU device id used in test)> <max tokens> <max sentences> <a set of random seed to try, space separate(e.g: '1 1000 2000 3000')> <whether to use entire model dir to ensemble decoding(e.g: false)> <whether to force redo remove same and segmentation(e.g: false)>"
    exit -1
fi


for rdm_seed in ${try_random_seed}
do
    ${one_script_to_run_all_sh} ${model_arch} ${model_level} ${which_pretrained_embed} ${fusion_mode} \
        ${short} ${long} ${low} ${high} \
        ${src_vocab_size} ${trg_vocab_size} ${GPU_used_training} ${GPU_used_inference} \
        ${MAX_TOKENS} ${MAX_SENS} ${rdm_seed} ${want_ensemble} ${force_redo_remove_same_and_seg}
done

