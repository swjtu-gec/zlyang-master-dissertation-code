#!/usr/bin/env bash

## This script will do train, inference and evaluation.

set -x
set -e

source ../paths.sh

training_dir=${BASE_DIR}/training
train_sh=${training_dir}/train_yangzl.sh
train_embed_sh=${training_dir}/train_embed_yangzl.sh
run_trained_model_sh=${training_dir}/run_trained_model.sh
remove_spac_pkunlp_segment_sh=${training_dir}/remove_spac_pkunlp_segment.sh


if [[ $# -ge 13 ]]; then
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
    random_seed=${12}
    want_ensemble=${13}
    if [[ $# -eq 15 ]]; then
        reranker_weights=${14}
        reranker_feats=${15}
    fi
else
    echo "Usage: `basename $0` <dir to bin data> <dir to BPE model> <embed_file_url> <test input> <url to gold edit> <url to m2scorer script> <GPU device id to use in training(e.g: '0, 1, 2')> <GPU device id used in test)> <model name(e.g: fconv_zh_bpe_embed_fusion)> <max tokens> <max sentences> <random seed> <whether to use entire model dir to ensemble decoding(e.g: true or false)> [optional args: <url-to-reranker-weights-file> <features(e.g: eo, eolm)]"
    exit -1
fi


if [[ ! -f "${EMBED_URL}" ]]; then
    ${train_sh} ${BIN_DATA_DIR} ${GPUs_used_training} ${model_name} ${random_seed} ${MAX_TOKENS} ${MAX_SENS}
else
    ${train_embed_sh} ${BIN_DATA_DIR} ${GPUs_used_training} ${model_name} ${EMBED_URL} ${random_seed} ${MAX_TOKENS} ${MAX_SENS}
fi


if [[ "${want_ensemble}" == true ]]; then
    model_path_used_inference=./models/${model_name}/model${random_seed}
    inference_output_dir=../eval/${model_name}_seed${random_seed}_ensmble
else
    model_path_used_inference=./models/${model_name}/model${random_seed}/checkpoint_best.pt
    inference_output_dir=../eval/${model_name}_seed${random_seed}_single
fi
gec_system_out=${inference_output_dir}/output.tok.txt


if [[ $# -eq 13 ]]; then
    ${run_trained_model_sh} ${test_input} ${inference_output_dir} ${GPUs_used_inference} ${model_path_used_inference} ${BIN_DATA_DIR} ${BPE_MODEL_DIR}
elif [[ $# -eq 15 ]]; then
    ${run_trained_model_sh} ${test_input} ${inference_output_dir} ${GPUs_used_inference} ${model_path_used_inference} ${BIN_DATA_DIR} ${BPE_MODEL_DIR} ${reranker_weights} ${reranker_feats}
else
    echo "the num of argument do not match!"
    exit -2
fi


${remove_spac_pkunlp_segment_sh} ${gec_system_out} ${m2scorer_url} ${gold_edit}

