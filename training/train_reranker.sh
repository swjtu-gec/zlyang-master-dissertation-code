#!/bin/bash

# This script used to train re-ranker.

set -e
set -x

source ../paths.sh

if [[ $# -ge 8 ]]; then
    dev_data_dir=$1
    output_dir=$2
    device=$3
    model_path=$4
    reranker_feats=$5
    moses_path=$6
    DATA_BIN_DIR=$7
    BPE_MODEL_DIR=$8
    if [[ "${reranker_feats}" == "eolm" || "${reranker_feats}" == "lm" ]]; then
        lm_url=$9
        if [[ ! -f ${lm_url} ]]; then
            echo "Language model not found: ${lm_url}"
            exit -2
        fi
    fi
else
    echo "Usage: `basename $0` <dir to dev data> <output_dir> <GPU device id to use(e.g: 0)> <path to model_file/dir> <features, e.g: 'eo' or 'lm' or 'eolm'> <path-to-moses> <dir to bin data> <dir to BPE model> [optional args: <trained language model's url>]"
    exit -1
fi


NBEST_RERANKER=${SOFTWARE_DIR}/nbest-reranker
beam=12
nbest=${beam}
threads=12

# setting model paths
if [[ -d "$model_path" ]]; then
    models=`ls ${model_path}/*pt | tr '\n' ' '`
    models=${models//pt /pt:}
    models=${models/%:/''}
    echo ${models}
elif [[ -f "$model_path" ]]; then
    models=${model_path}
elif [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
    exit -3
fi


###############
# training
###############

TRAIN_DIR=${output_dir}/training/
mkdir -p ${TRAIN_DIR}
echo "[weight]" > ${TRAIN_DIR}/rerank_config.ini
echo "F0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
if [[ "${reranker_feats}" == "eo" ]]; then
    echo "EditOps0= 0.2 0.2 0.2" >> ${TRAIN_DIR}/rerank_config.ini
elif [[ "${reranker_feats}" == "eolm" ]]; then
    echo "EditOps0= 0.2 0.2 0.2" >> ${TRAIN_DIR}/rerank_config.ini
    echo "LM0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
    echo "WordPenalty0= -1" >> ${TRAIN_DIR}/rerank_config.ini
elif [[ "${reranker_feats}" == "lm" ]]; then
    echo "LM0= 0.5" >> ${TRAIN_DIR}/rerank_config.ini
    echo "WordPenalty0= -1" >> ${TRAIN_DIR}/rerank_config.ini
else
    echo "Unknown re-ranker features string, got ${reranker_feats}"
    exit -4
fi

if [[ "${reranker_feats}" == "eo" ]]; then
    featstring="EditOps(name='EditOps0')"
elif [[ "${reranker_feats}" == "lm" ]]; then
    featstring="LM('LM0', '$lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
elif [[ "${reranker_feats}" == "eolm" ]]; then
    featstring="EditOps(name='EditOps0'), LM('LM0', '$lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
fi


train_reranker_starttime=$(date +%s)

if [[ -d "$BPE_MODEL_DIR" ]]; then
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/train.bpe.model < ${dev_data_dir}/dev.input.txt > ${output_dir}/dev.input.bpe.txt
    beam_search_input=${output_dir}/dev.input.bpe.txt
else
    beam_search_input=${dev_data_dir}/dev.input.txt
fi

CUDA_VISIBLE_DEVICES="${device}" python ${FAIRSEQPY}/interactive.py \
    --no-progress-bar --path ${models} \
    --beam ${beam} --nbest ${beam} \
    ${DATA_BIN_DIR} < ${beam_search_input} > ${output_dir}/beamsearch.output.nbest.txt

# reformating the nbest file
if [[ -d "$BPE_MODEL_DIR" ]]; then
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${output_dir}/beamsearch.output.nbest.txt --debpe > ${output_dir}/dev.output.tok.nbest.reformat.txt
else
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${output_dir}/beamsearch.output.nbest.txt > ${output_dir}/dev.output.tok.nbest.reformat.txt
fi

# augmenting the dev nbest
${NBEST_RERANKER}/augmenter.py -s ${dev_data_dir}/dev.input.txt -i ${output_dir}/dev.output.tok.nbest.reformat.txt -o ${output_dir}/dev.output.tok.nbest.reformat.augmented.txt -f "$featstring"

# training the nbest to obtain the weights
${NBEST_RERANKER}/train.py -i ${output_dir}/dev.output.tok.nbest.reformat.augmented.txt -r ${dev_data_dir}/dev.m2 -c ${TRAIN_DIR}/rerank_config.ini --threads ${threads} --tuning-metric m2 --predictable-seed -o ${TRAIN_DIR} --moses-dir ${moses_path} --no-add-weight

cp ${TRAIN_DIR}/weights.txt ${output_dir}/weights.${reranker_feats}.txt

train_reranker_endtime=$(date +%s)
cost=$((train_reranker_endtime - train_reranker_starttime))
echo "re-ranker training end. cost ${cost}s"

