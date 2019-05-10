#!/usr/bin/env bash

set -e
set -x

source ../paths.sh

multi_channel_translation_sh=${BASE_DIR}/training/multi_channel_translation.sh
merge_py=${BASE_DIR}/training/merge.py
NBEST_RERANKER=${SOFTWARE_DIR}/nbest-reranker

encoding='utf-8'


if [[ $# -ge 12 ]]; then
    input_char_file=$1
    output_dir=$2
    device=$3
    use_which_channels=$4
    channel_output_mode=$5
    force_redo_translation=$6
    char_model_path=$7
    CHAR_BIN_DATA_DIR=$8
    bpe_model_path=$9
    BPE_BIN_DATA_DIR=${10}
    BPE_MODEL_DIR=${11}
    reranker_feats=${12}  # nmt-score, lm-score, lm-score-normalized, eo-feats, lm-feats, lm-feats-normalized, eo+lm
    reranker_weights=${13}
    char_lm_url=${14}
    if [[ "${reranker_feats}" == "eo-feats" || "${reranker_feats}" == "lm-feats" || \
        ${reranker_feats} == "lm-feats-normalized" || "${reranker_feats}" == "eo+lm" ]]; then
        if [[ ! -f ${reranker_weights} ]]; then
            echo "weights file not found in ${reranker_weights}"
            exit -2
        fi
    fi
    if [[ "${reranker_feats}" == "lm-score" || "${reranker_feats}" == "lm-score-normalized" || \
        "${reranker_feats}" == "lm-feats" || ${reranker_feats} == "lm-feats-normalized" || \
        "${reranker_feats}" == "eo+lm" ]]; then
        if [[ ! -f ${char_lm_url} ]]; then
            echo "char-level language model not found in ${char_lm_url}"
            exit -3
        fi
    fi
else
    echo "Usage: `basename $0` <input file in char level> <output_dir> <GPU device id to use(e.g: 0)> <use which channels, e.g: '1 1 0 0'> <channel output mode, e.g: 1-best or n-best> <whether to force redo translation(e.g: false)> <path to char level model file/dir> <dir to bin data of char level model> <path to bpe level model file/dir> <dir to bin data of bpe level model> <dir to BPE model> <features, e.g: nmt-score, lm-score, lm-score-normalized, eo-feats, lm-feats, lm-feats-normalized, eo+lm> [optional args: <path-to-reranker-weights> <trained char-level language model's url>]"
    exit -1
fi


rerank_dir=${output_dir}/${reranker_feats}
mkdir -p ${rerank_dir}

if [[ "${reranker_feats}" == "nmt-score" ]]; then
    featstring=None
    reranker_weights=${rerank_dir}/weights.${reranker_feats}.txt
    echo 1 > ${reranker_weights}
elif [[ "${reranker_feats}" == "eo-feats" ]]; then
    featstring="EditOps(name='EditOps0')"
elif [[ "${reranker_feats}" == "lm-score" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=False)"
    reranker_weights=${rerank_dir}/weights.${reranker_feats}.txt
    echo 0 1 > ${reranker_weights}
elif [[ "${reranker_feats}" == "lm-score-normalized" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=True)"
    reranker_weights=${rerank_dir}/weights.${reranker_feats}.txt
    echo 0 1 > ${reranker_weights}
elif [[ "${reranker_feats}" == "lm-feats" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
elif [[ "${reranker_feats}" == "lm-feats-normalized" ]]; then
    featstring="LM('LM0', '$char_lm_url', normalize=True), WordPenalty(name='WordPenalty0')"
elif [[ "${reranker_feats}" == "eo+lm" ]]; then
    featstring="EditOps(name='EditOps0'), LM('LM0', '$char_lm_url', normalize=False), WordPenalty(name='WordPenalty0')"
else
    echo "unknown re-ranker features string, got ${reranker_feats}"
    exit -4
fi


${multi_channel_translation_sh} ${input_char_file} ${output_dir} ${device} \
    ${char_model_path} ${CHAR_BIN_DATA_DIR} \
    ${bpe_model_path} ${BPE_BIN_DATA_DIR} ${BPE_MODEL_DIR} \
    "${use_which_channels}" ${channel_output_mode} ${force_redo_translation}

M1_output=${output_dir}/M1.fairseq.output.${channel_output_mode}.txt
M2_output=${output_dir}/M2.fairseq.output.${channel_output_mode}.txt
M3_output=${output_dir}/M3.fairseq.output.${channel_output_mode}.txt
M4_output=${output_dir}/M4.fairseq.output.${channel_output_mode}.txt


echo "###################"
echo "re-ranker component"
echo "###################"
rerank_starttime=$(date +%s)

used_channels=(${use_which_channels})
use_M1=${used_channels[0]}
use_M2=${used_channels[1]}
if [[ ${use_M2} -ne 0 ]]; then
    use_M1=1
fi
use_M3=${used_channels[2]}
use_M4=${used_channels[3]}
if [[ ${use_M4} -ne 0 ]]; then
    use_M3=1
fi

if [[ ${use_M1} -ne 0 ]]; then
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${M1_output} -o ${M1_output}.reformat
    to_merge_files=${M1_output}.reformat
fi
if [[ ${use_M2} -ne 0 ]]; then
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${M2_output} --debpe --char-seg -o ${M2_output}.reformat
    to_merge_files="${to_merge_files} ${M2_output}.reformat"
fi
if [[ ${use_M3} -ne 0 ]]; then
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${M3_output} --debpe --char-seg -o ${M3_output}.reformat
    to_merge_files="${to_merge_files} ${M3_output}.reformat"
fi
if [[ ${use_M4} -ne 0 ]]; then
    ${SCRIPTS_DIR}/nbest_reformat.py -i ${M4_output} -o ${M4_output}.reformat
    to_merge_files="${to_merge_files} ${M4_output}.reformat"
fi

merged_file=${rerank_dir}/output.${channel_output_mode}.char.reformat.merged
python ${merge_py} --to-merge-files="${to_merge_files}" --trg-fname=${merged_file} --encoding=${encoding}

if [[ ${featstring} != None ]]; then
    ${NBEST_RERANKER}/augmenter.py -s ${input_char_file} -i ${merged_file} -o ${merged_file}.augmented -f "$featstring"
else
    cp ${merged_file} ${merged_file}.augmented
fi

${NBEST_RERANKER}/rerank.py -i ${merged_file}.augmented -w ${reranker_weights} -o ${rerank_dir} --clean-up
mv ${merged_file}.augmented.reranked.1best ${rerank_dir}/output.char.merged.reranked.txt


rerank_endtime=$(date +%s)
rerank_cost=$((rerank_endtime - rerank_starttime))
echo "re-rank end. cost ${rerank_cost}s"

