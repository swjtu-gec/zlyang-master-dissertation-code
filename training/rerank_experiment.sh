#!/usr/bin/env bash

set -e
set -x

source ../paths.sh


#################
# fixed arguments
#################
train_reranker_sh=${BASE_DIR}/training/train_reranker.sh
run_trained_model_sh=${BASE_DIR}/training/run_trained_model.sh
remove_spac_pkunlp_segment_sh=${BASE_DIR}/training/remove_spac_pkunlp_segment.sh

jieba_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/jieba.seg.txt
char_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/char.seg.txt

word_dev_data_dir=zh-bpe-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0
char_dev_data_dir=zh-char-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/
moses_path=../../mosesdecoder/

BPE_BIN_DATA_DIR=zh-bpe-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/bin/
CHAR_BIN_DATA_DIR=zh-char-processed-NLPCC_betterseg+HSK-remove-short1-long1000-low0.1-high200.0/bin/

char_lm_url=models/lm/wiki_zh.char.5gram.binary.trie
word_lm_url=${BASE_DIR}/training/models/lm/wiki_zh.word.5gram.binary.trie

gold_edit=${BASE_DIR}/data/test/nlpcc2018-test/gold.01
m2scorer_url=${BASE_DIR}/eval/m2scorer/scripts/m2scorer.py


############################
# modify following arguments
############################

# bpe level model
device=0
# eo, lm, eolm
reranker_feats=eolm
dev_data_dir=${word_dev_data_dir}
model_path=${BASE_DIR}/training/models/4_ens_fconv_bpe_word2vec_2_4_5_6
BPE_MODEL_DIR=models/zh_bpe_model_NLPCC_betterseg+HSK_remove_short1_long1000_low0.1_high200.0/
train_reranker_output_dir=../eval/${model_path/##*/}_train_reranker_${reranker_feats}
DATA_BIN_DIR=${BPE_BIN_DATA_DIR}
test_input=${jieba_seg_test_input}
run_models_output_dir=../eval/${model_path/##*/}_${reranker_feats}
reranker_weights=${train_reranker_output_dir}/weights.${reranker_feats}.txt
lm_url=${word_lm_url}
gec_system_output=${run_models_output_dir}/output.tok.txt


## char level model
#device=0
## eo, lm, eolm
#reranker_feats=eolm
#dev_data_dir=${char_dev_data_dir}
#model_path=${BASE_DIR}/training/models/4_ens_fconv_char_random_1_2_4_1000
#BPE_MODEL_DIR=None
#train_reranker_output_dir=../eval/${model_path/##*/}_train_reranker_${reranker_feats}
#DATA_BIN_DIR=${CHAR_BIN_DATA_DIR}
#test_input=${char_seg_test_input}
#run_models_output_dir=../eval/${model_path/##*/}_${reranker_feats}
#reranker_weights=${train_reranker_output_dir}/weights.${reranker_feats}.txt
#lm_url=${char_lm_url}
#gec_system_output=${run_models_output_dir}/output.tok.txt


################################################################################
# training re-ranker component => inference + re-ranking => evaluate performance
################################################################################
${train_reranker_sh} ${dev_data_dir} ${train_reranker_output_dir} ${device} \
    ${model_path} ${reranker_feats} ${moses_path} \
    ${DATA_BIN_DIR} ${BPE_MODEL_DIR} ${lm_url}

${run_trained_model_sh} ${test_input} ${run_models_output_dir} ${device} \
    ${model_path} ${DATA_BIN_DIR} ${BPE_MODEL_DIR} \
    ${reranker_weights} ${reranker_feats} ${lm_url}

${remove_spac_pkunlp_segment_sh} ${m2scorer_url} ${gec_system_output} ${gold_edit} false

