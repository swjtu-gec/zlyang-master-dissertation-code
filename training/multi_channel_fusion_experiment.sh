#!/usr/bin/env bash

set -e
set -x

source ../paths.sh

if [[ $# -eq 7 ]]; then
    device=$1
    use_which_channels=$2
    channel_output_mode=$3
    force_redo_translation=$4
    char_model_path=$5
    bpe_model_path=$6
    reranker_feats=$7
else
    echo "Usage: `basename $0` <GPU device id to use(e.g: 0)> <use which channels, e.g: '1 1 1 0'> <channel output mode, e.g: 1-best or n-best> <whether to force redo translation(e.g: false)> <path to char level model file/dir> <path to bpe level model file/dir> <features, e.g: nmt-score, lm-score, lm-score-normalized, eo-feats, lm-feats, lm-feats-normalized, eo+lm>"
    exit -1
fi


#################
# fixed arguments
#################
train_multi_channel_fusion_reranker_sh=${BASE_DIR}/training/train_multi_channel_fusion_reranker.sh
multi_channel_fusion_sh=${BASE_DIR}/training/multi_channel_fusion.sh
remove_spac_pkunlp_segment_sh=${BASE_DIR}/training/remove_spac_pkunlp_segment.sh

char_dev_data_dir=zh-char-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/
moses_path=../../mosesdecoder/
CHAR_BIN_DATA_DIR=zh-char-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/bin/
BPE_BIN_DATA_DIR=zh-bpe-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/bin/
BPE_MODEL_DIR=models/zh_bpe_model_NLPCC_betterseg+HSK_remove_short1_long1000_low0.1_high200.0/
if [[ -d ${char_model_path} ]]; then
    inference_mode=4ens
else
    inference_mode=single
fi
multi_channel_fusion_train_reranker_output_dir=../eval/multi_channel_fusion_train_reranker_${inference_mode}_${channel_output_mode/-/}/
char_lm_url=models/lm/wiki_zh.char.5gram.binary.trie

test_input_char_file=../data/test/nlpcc2018-test/char.seg.txt
multi_channel_translation_output_dir=../eval/multi_channel_translation_${inference_mode}_${channel_output_mode/-/}/
reranker_weights=${multi_channel_fusion_train_reranker_output_dir}/${reranker_feats}/weights.${reranker_feats}.txt

gold_edit=${BASE_DIR}/data/test/nlpcc2018-test/gold.01
m2scorer_url=${BASE_DIR}/eval/m2scorer/scripts/m2scorer.py
gec_system_output=${multi_channel_translation_output_dir}/${reranker_feats}/output.char.merged.reranked.txt


##############################################################################
# training re-ranker component => multi channel fusion => evaluate performance
##############################################################################
if [[ "${reranker_feats}" == "eo-feats" || "${reranker_feats}" == "lm-feats" || \
        ${reranker_feats} == "lm-feats-normalized" || "${reranker_feats}" == "eo+lm" ]]; then
    ${train_multi_channel_fusion_reranker_sh} \
        ${char_dev_data_dir} ${multi_channel_fusion_train_reranker_output_dir} \
        ${device} "$use_which_channels" ${channel_output_mode} ${force_redo_translation} ${moses_path} \
        ${char_model_path} ${CHAR_BIN_DATA_DIR} ${bpe_model_path} ${BPE_BIN_DATA_DIR} ${BPE_MODEL_DIR} \
        ${reranker_feats} ${char_lm_url}
fi

${multi_channel_fusion_sh} ${test_input_char_file} ${multi_channel_translation_output_dir} \
    ${device} "$use_which_channels" ${channel_output_mode} ${force_redo_translation} \
    ${char_model_path} ${CHAR_BIN_DATA_DIR} ${bpe_model_path} ${BPE_BIN_DATA_DIR} ${BPE_MODEL_DIR} \
    ${reranker_feats} ${reranker_weights} ${char_lm_url}

${remove_spac_pkunlp_segment_sh} ${m2scorer_url} ${gec_system_output} ${gold_edit} false

