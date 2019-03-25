#!/usr/bin/env bash

## This script will do train, inference and evaluation.

set -x
set -e

source ../paths.sh

run_trained_model_sh=${BASE_DIR}/training/run_trained_model.sh
remove_spac_pkunlp_segment_sh=${BASE_DIR}/training/remove_spac_pkunlp_segment.sh


if [[ $# -eq 15 ]]; then
    model_arch=$1
    BIN_DATA_DIR=$2
    BPE_MODEL_DIR=$3
    EMBED_URL=$4
    test_input=$5
    gold_edit=$6
    m2scorer_url=$7
    GPU_used_training=$8
    GPU_used_inference=$9
    model_name=${10}
    MAX_TOKENS=${11}
    MAX_SENS=${12}
    random_seed=${13}
    want_ensemble=${14}
    dev_data_dir=${15}
else
    echo "Usage: `basename $0` <model arch, e.g: lstm, fconv, transformer> <dir to bin data> <dir to BPE model> <embed_file_url> <test input> <url to gold edit> <url to m2scorer script> <GPU device id to use in training(e.g: 0)> <GPU device id used in test)> <model name(e.g: fconv_zh_bpe_embed_fusion)> <max tokens> <max sentences> <random seed> <whether to use entire model dir to ensemble decoding(e.g: true or false)> <dir to dev data>"
    exit -1
fi


if [[ ${model_arch} == 'fconv' ]]; then
    train_sh=${BASE_DIR}/training/train_fconv.sh
    train_embed_sh=${BASE_DIR}/training/train_fconv_embed.sh
elif [[ "$model_arch" == 'lstm' ]]; then
    train_sh=${BASE_DIR}/training/train_lstm.sh
    train_embed_sh=${BASE_DIR}/training/train_lstm_embed.sh
elif [[ "$model_arch" == 'transformer' ]]; then
    train_sh=${BASE_DIR}/training/train_transformer.sh
    train_embed_sh=${BASE_DIR}/training/train_transformer_embed.sh
else
    echo "illegal model architecture, got $model_arch"
    exit -2
fi


if [[ ! -f "${EMBED_URL}" ]]; then
    ${train_sh} ${BIN_DATA_DIR} "${GPU_used_training}" ${model_name} ${random_seed} ${MAX_TOKENS} ${MAX_SENS}
else
    ${train_embed_sh} ${BIN_DATA_DIR} "${GPU_used_training}" ${model_name} ${EMBED_URL} ${random_seed} ${MAX_TOKENS} ${MAX_SENS}
fi


if [[ "${want_ensemble}" == true ]]; then
    model_path_used_inference=./models/${model_name}/model${random_seed}
    inference_output_dir=../eval/${model_name}_seed${random_seed}_ensmble
else
    model_path_used_inference=./models/${model_name}/model${random_seed}/checkpoint_best.pt
    inference_output_dir=../eval/${model_name}_seed${random_seed}_single
fi
gec_system_out=${inference_output_dir}/output.tok.txt


echo "========= M^2 F_0.5 score on dev set ========="
${run_trained_model_sh} ${dev_data_dir}/dev.input.txt ${inference_output_dir} "${GPU_used_inference}" ${model_path_used_inference} ${BIN_DATA_DIR} ${BPE_MODEL_DIR}
${m2scorer_url} ${gec_system_out} ${dev_data_dir}/dev.m2


echo "========= M^2 F_0.5 score on test set ========="
${run_trained_model_sh} ${test_input} ${inference_output_dir} "${GPU_used_inference}" ${model_path_used_inference} ${BIN_DATA_DIR} ${BPE_MODEL_DIR}
${remove_spac_pkunlp_segment_sh} ${m2scorer_url} ${gec_system_out} ${gold_edit} 'false'

