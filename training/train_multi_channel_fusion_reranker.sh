#!/usr/bin/env bash

set -e
set -x

source ../paths.sh

multi_channel_fusion_sh=${BASE_DIR}/training/multi_channel_fusion.sh

NBEST_RERANKER=${SOFTWARE_DIR}/nbest-reranker
threads=12


if [[ $# -ge 13 ]]; then
    char_dev_data_dir=$1
    output_dir=$2
    device=$3
    use_which_channels=$4
    channel_output_mode=$5
    force_redo_translation=$6
    moses_path=$7

    char_model_path=$8
    CHAR_BIN_DATA_DIR=$9
    bpe_model_path=${10}
    BPE_BIN_DATA_DIR=${11}
    BPE_MODEL_DIR=${12}

    reranker_feats=${13}  # eo-feats, lm-feats, lm-feats-normalized, eo+lm
    char_lm_url=${14}
    if [[ "${reranker_feats}" == "lm-feats" || "${reranker_feats}" == "lm-feats-normalized" || \
        "${reranker_feats}" == "eo+lm" ]]; then
        if [[ ! -f ${char_lm_url} ]]; then
            echo "char-level language model not found in ${char_lm_url}"
            exit -2
        fi
    fi
else
    echo "Usage: `basename $0` <dir to dev data in char level> <output_dir> <GPU device id to use(e.g: 0)> <use which channels, e.g: '1 1 0 0'> <channel output mode, e.g: 1-best or n-best> <whether to force redo translation(e.g: false)> <path-to-moses> <path to char level model file/dir> <dir to bin data of char level model> <path to bpe level model file/dir> <dir to bin data of bpe level model> <dir to BPE model> <features, e.g: eo-feats, lm-feats, lm-feats-normalized, eo+lm> [optional args: <trained char-level language model's url>]"
    exit -1
fi


echo "############################"
echo "training re-ranker component"
echo "############################"

rerank_dir=${output_dir}/${reranker_feats}
TRAIN_DIR=${output_dir}/${reranker_feats}/training/
mkdir -p ${TRAIN_DIR}

echo "[weight]" > ${TRAIN_DIR}/rerank_config.ini
echo "F0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
if [[ "${reranker_feats}" == "eo-feats" ]]; then
    echo "EditOps0= 0.2 0.2 0.2" >> ${TRAIN_DIR}/rerank_config.ini
elif [[ "${reranker_feats}" == "lm-feats" || "${reranker_feats}" == "lm-feats-normalized" ]]; then
    echo "LM0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
    echo "WordPenalty0= -1" >> ${TRAIN_DIR}/rerank_config.ini
elif [[ "${reranker_feats}" == "eo+lm" ]]; then
    echo "EditOps0= 0.2 0.2 0.2" >> ${TRAIN_DIR}/rerank_config.ini
    echo "LM0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
    echo "WordPenalty0= -1" >> ${TRAIN_DIR}/rerank_config.ini
else
    echo "unknown re-ranker features string, got ${reranker_feats}"
    exit -3
fi

if [[ "${reranker_feats}" == "eo-feats" ]]; then
    featstring="EditOps(name='EditOps0')"
elif [[ "${reranker_feats}" == "lm-feats" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
elif [[ "${reranker_feats}" == "lm-feats-normalized" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=True), WordPenalty(name='WordPenalty0')"
elif [[ "${reranker_feats}" == "eo+lm" ]]; then
    featstring="EditOps(name='EditOps0'), LM('LM0', '$char_lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
fi


train_reranker_starttime=$(date +%s)

${multi_channel_fusion_sh} ${char_dev_data_dir}/dev.input.txt ${output_dir} ${device} \
    "$use_which_channels" ${channel_output_mode} ${force_redo_translation} \
    ${char_model_path} ${CHAR_BIN_DATA_DIR} ${bpe_model_path} ${BPE_BIN_DATA_DIR} ${BPE_MODEL_DIR} \
    nmt-score None None

merged_file=${rerank_dir}/output.${channel_output_mode}.char.reformat.merged
less ${output_dir}/nmt-score/output.${channel_output_mode}.char.reformat.merged > ${merged_file}

# augmenting the dev nbest
${NBEST_RERANKER}/augmenter.py \
    -s ${char_dev_data_dir}/dev.input.txt \
    -i ${merged_file} \
    -o ${merged_file}.augmented \
    -f "$featstring"

# training the nbest to obtain the weights
${NBEST_RERANKER}/train.py \
    -i ${merged_file}.augmented \
    -r ${char_dev_data_dir}/dev.m2 \
    -c ${TRAIN_DIR}/rerank_config.ini \
    --threads ${threads} --tuning-metric m2 \
    --predictable-seed -o ${TRAIN_DIR} \
    --moses-dir ${moses_path} --no-add-weight

less ${TRAIN_DIR}/weights.txt > ${rerank_dir}/weights.${reranker_feats}.txt

train_reranker_endtime=$(date +%s)
cost=$((train_reranker_endtime - train_reranker_starttime))
echo "re-ranker training end. cost ${cost}s"

