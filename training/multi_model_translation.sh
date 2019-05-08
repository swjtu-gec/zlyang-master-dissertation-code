#!/usr/bin/env bash

set -e
set -x

source ../paths.sh

segment_py=${BASE_DIR}/data/segment.py

encoding='utf-8'
beam=12


if [[ $# -eq 11 ]]; then
    input_file=$1
    output_dir=$2
    device=$3
    char_model_path=$4
    CHAR_BIN_DATA_DIR=$5
    bpe_model_path=$6
    BPE_BIN_DATA_DIR=$7
    BPE_MODEL_DIR=$8
    use_which_channels=$9
    channel_output_mode=${10}
    force_redo_translation=${11}
else
    echo "Usage: `basename $0` <input_file> <output_dir> <GPU device id to use(e.g: 0)> <path to char level model file/dir> <dir to bin data of char level model> <path to bpe level model file/dir> <dir to bin data of bpe level model> <dir to BPE model> <use which channels, e.g: '1 1 0 0'> <channel output mode, e.g: 1-best or n-best> <whether to force redo translation(e.g: false)>"
    exit -1
fi


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


if [[ ${channel_output_mode} == '1-best' ]]; then
    nbest=1
elif [[ ${channel_output_mode} == 'n-best' ]]; then
    nbest=${beam}
else
    echo "illegal channel output mode, got $channel_output_mode"
    exit -2
fi


if [[ -d "$char_model_path" ]]; then
    char_models=`ls ${char_model_path}/*pt | tr '\n' ' '`
    char_models=${char_models//pt /pt:}
    char_models=${char_models/%:/''}
    echo ${char_models}
elif [[ -f "$char_model_path" ]]; then
    char_models=${char_model_path}
elif [[ ! -e "$char_model_path" ]]; then
    echo "char level model(s) path not found in $char_model_path"
    exit -3
fi

if [[ -d "$bpe_model_path" ]]; then
    bpe_models=`ls ${bpe_model_path}/*pt | tr '\n' ' '`
    bpe_models=${bpe_models//pt /pt:}
    bpe_models=${bpe_models/%:/''}
    echo ${bpe_models}
elif [[ -f "$bpe_model_path" ]]; then
    bpe_models=${bpe_model_path}
elif [[ ! -e "$bpe_model_path" ]]; then
    echo "bpe level model(s) path not found in $bpe_model_path"
    exit -4
fi


##########################
# multi prediction channel
##########################
translation_starttime=$(date +%s)
mkdir -p ${output_dir}
sed 's/ //g' ${input_file} > ${output_dir}/input.sen.txt

# ============================
# M1 channel: char level model
# ============================
M1_output=${output_dir}/M1.fairseq.output.${channel_output_mode}.txt
if [[ ! -e ${M1_output} || ${force_redo_translation} == true ]]; then
    if [[ ${use_M1} -ne 0 ]]; then
        # prepare input for M1 channel
        M1_input=${output_dir}/M1.input.char.txt
        python ${segment_py} --raw-fname=${output_dir}/input.sen.txt --seg-fname=${M1_input} --char-level --encoding=${encoding}

        CUDA_VISIBLE_DEVICES="${device}" python ${FAIRSEQPY}/interactive.py \
            --no-progress-bar \
            --path ${char_models} \
            --beam ${beam} --nbest ${nbest} \
            --model-overrides "{'encoder_embed_path': None, 'decoder_embed_path': None}" \
            ${CHAR_BIN_DATA_DIR} < ${M1_input} > ${M1_output}
    fi
fi

# ==============================================
# M2 channel: char level model + bpe level model
# ==============================================
M2_output=${output_dir}/M2.fairseq.output.${channel_output_mode}.txt
if [[ ${use_M2} -ne 0 ]] && [[ ! -e ${M2_output} || ${force_redo_translation} == true ]]; then
    M2_input=${output_dir}/M2.input.bpe.txt
    # getting best hypotheses for M2 input
    cat ${M1_output} | grep "^H"  | \
        python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if(i%$nbest == 0) ]); print(x)" | \
        cut -f3 | sed '$d' | sed 's| ||g' > ${output_dir}/M2.input.sen.txt
    python ${segment_py} --raw-fname=${output_dir}/M2.input.sen.txt --seg-fname=${output_dir}/M2.input.jieba.txt --encoding=${encoding}
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/train.bpe.model < ${output_dir}/M2.input.jieba.txt > ${M2_input}

    CUDA_VISIBLE_DEVICES="${device}" python ${FAIRSEQPY}/interactive.py \
        --no-progress-bar \
        --path ${bpe_models} \
        --beam ${beam} --nbest ${nbest} \
        --model-overrides "{'encoder_embed_path': None, 'decoder_embed_path': None}" \
        ${BPE_BIN_DATA_DIR} < ${M2_input} > ${M2_output}
fi

# ===========================
# M3 channel: BPE level model
# ===========================
M3_output=${output_dir}/M3.fairseq.output.${channel_output_mode}.txt
if [[ ${use_M3} -ne 0 ]] && [[ ! -e ${M3_output} || ${force_redo_translation} == true ]]; then
    # prepare input for M3 channel
    M3_input=${output_dir}/M3.input.bpe.txt
    python ${segment_py} --raw-fname=${output_dir}/input.sen.txt --seg-fname=${output_dir}/M3.input.jieba.txt --encoding=${encoding}
    ${SCRIPTS_DIR}/apply_bpe.py -c ${BPE_MODEL_DIR}/train.bpe.model < ${output_dir}/M3.input.jieba.txt > ${M3_input}

    CUDA_VISIBLE_DEVICES="${device}" python ${FAIRSEQPY}/interactive.py \
        --no-progress-bar \
        --path ${bpe_models} \
        --beam ${beam} --nbest ${nbest} \
        --model-overrides "{'encoder_embed_path': None, 'decoder_embed_path': None}" \
        ${BPE_BIN_DATA_DIR} < ${M3_input} > ${M3_output}
fi

# ==============================================
# M4 channel: bpe level model + char level model
# ==============================================
M4_output=${output_dir}/M4.fairseq.output.${channel_output_mode}.txt
if [[ ${use_M4} -ne 0 ]] && [[ ! -e ${M4_output} || ${force_redo_translation} == true ]]; then
    # prepare input for M4 channel
    M4_input=${output_dir}/M4.input.char.txt
    cat ${M3_output} | grep "^H"  | \
        python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if(i%$nbest == 0) ]); print(x)" | \
        cut -f3 | sed 's|@@ ||g' | sed '$d' | sed 's| ||g' > ${output_dir}/M4.input.sen.txt
    python ${segment_py} --raw-fname=${output_dir}/M4.input.sen.txt --seg-fname=${M4_input} --char-level --encoding=${encoding}

    CUDA_VISIBLE_DEVICES="${device}" python ${FAIRSEQPY}/interactive.py \
        --no-progress-bar \
        --path ${bpe_models} \
        --beam ${beam} --nbest ${nbest} \
        --model-overrides "{'encoder_embed_path': None, 'decoder_embed_path': None}" \
        ${BPE_BIN_DATA_DIR} < ${M4_input} > ${M4_output}
fi

translation_endtime=$(date +%s)
translation_cost=$((translation_endtime - translation_starttime))
echo "multi model translation end. cost ${translation_cost}s"

