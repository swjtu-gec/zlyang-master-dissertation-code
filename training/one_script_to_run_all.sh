#!/usr/bin/env bash

## This script will do everything, including remove-same, training data processing, remove illegal sentences,
## random split, preprocess_yangzl.sh and full_process.sh

set -x
set -e

source ../paths.sh

preprocessing_py=${BASE_DIR}/data/preprocessing.py
nlpcc2018_gec_process_py=${BASE_DIR}/data/nlpcc2018_gec_process.py
segment_py=${BASE_DIR}/data/segment.py
random_split_py=${BASE_DIR}/data/random_split.py
preprocess_yangzl_sh=${BASE_DIR}/training/preprocess_yangzl.sh
convert_parallel_to_m2_sh=${BASE_DIR}/data/convert_parallel_to_m2.sh
full_process_sh=${BASE_DIR}/training/full_process.sh

nlpcc_raw_train_data=${BASE_DIR}/data/nlpcc-2018-traindata/data.train
nlpcc_train_data_remove_same=${BASE_DIR}/data/nlpcc-2018-traindata/data.train.removesame
encoding='utf-8'
nlpcc_betterseg_src=${BASE_DIR}/data/zh-official-nlpcc2018/lang8.src
nlpcc_betterseg_trg=${BASE_DIR}/data/zh-official-nlpcc2018/lang8.trg
hsk_data_dir=${BASE_DIR}/data/zh-hsk
blcu_data_dir=${BASE_DIR}/data/zh-blcu-nlpcc2018
fusion_data_dir=${BASE_DIR}/data/zh-fusion
fusion_src=${fusion_data_dir}/fusion.src
fusion_trg=${fusion_data_dir}/fusion.trg

dev_samples_num=5000
raw_test_input=${BASE_DIR}/data/test/nlpcc2018-test/source.txt
jieba_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/jieba.seg.txt
char_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/char.seg.txt
gold_edit=${BASE_DIR}/data/test/nlpcc2018-test/gold.01
m2scorer_url=${BASE_DIR}/eval/m2scorer/scripts/m2scorer.py
edit_creator_sh=${BASE_DIR}/software/m2scorer/scripts/edit_creator.py


if [[ $# -eq 17 ]]; then
    model_arch=$1  # lstm, fconv, transformer
    model_level=$2  # bpe, char, word
    which_pretrained_embed=$3  # None, word2vec, fasttext, wang2vec, bpemb
    fusion_mode=$4  # 1: nlpcc_betterseg; 2: nlpcc_betterseg+HSK; 3: nlpcc_betterseg+HSK+BLCU
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
    random_seed=${15}
    want_ensemble=${16}
    force_redo_remove_same_and_seg=${17}
else
    echo "Usage: `basename $0` <model arch, e.g: fconv, lstm, transformer> <model level, e.g: bpe, char, word> <use which pre-trained token embeddings, e.g: None, word2vec, fasttext, wang2vec, bpemb> <fusion mode: 1: nlpcc_betterseg; 2: nlpcc_betterseg+HSK; 3: nlpcc_betterseg+HSK+BLCU> <short, e.g: 1> <long, e.g: 100> <low, e.g: 0.0> <high, e.g: 9.0> <src_vocab_size> <trg_vocab_size> <GPU device id to use in training(e.g: 0)> <GPU device id used in test)> <max tokens> <max sentences> <random seed> <whether to use entire model dir to ensemble decoding(e.g: false)> <whether to force redo remove same and segmentation(e.g: false)>"
    exit -1
fi

if [[ "${model_level}" != 'bpe' && "${model_level}" != 'char' && "${model_level}" != 'word' ]]; then
    echo "illegal model level, got ${model_level}"
    exit -2
fi

if [[ ! -f ${nlpcc_train_data_remove_same} || "$force_redo_remove_same_and_seg" == true ]]; then
    python ${preprocessing_py} remove-same \
        --raw-fname=${nlpcc_raw_train_data} \
        --after-fname=${nlpcc_train_data_remove_same} \
        --encoding=${encoding}
fi

if [[ "$model_level" == 'char' ]]; then
    token_suffix='char'
    flag='--char-level'
else
    token_suffix='word'
    flag=''
fi

if [[ ! -f "$nlpcc_betterseg_src.$token_suffix" || ! -f "$nlpcc_betterseg_trg.$token_suffix" || "$force_redo_remove_same_and_seg" == true ]]; then
    if [[ "${model_level}" == 'char' ]]; then
        python ${segment_py} \
            --raw-fname=${raw_test_input} \
            --seg-fname=${char_seg_test_input} \
            ${flag} --encoding=${encoding}
    else
        python ${segment_py} \
            --raw-fname=${raw_test_input} \
            --seg-fname=${jieba_seg_test_input} \
            ${flag} --encoding=${encoding}
    fi
    python ${nlpcc2018_gec_process_py} \
        --nlpcc-traindata=${nlpcc_train_data_remove_same} \
        --train-all-src=${nlpcc_betterseg_src}.${token_suffix} \
        --train-all-trg=${nlpcc_betterseg_trg}.${token_suffix} \
        ${flag} --encoding=${encoding}
fi

if [[ ! -f "$hsk_data_dir/hsk.src.$token_suffix" || ! -f "$hsk_data_dir/hsk.trg.$token_suffix" || "$force_redo_remove_same_and_seg" == true ]]; then
    if [[ ${fusion_mode} == 2 || ${fusion_mode} == 3 ]]; then
        sed 's/ //g' ${hsk_data_dir}/hsk.src > ${hsk_data_dir}/hsk.src.remove.spac
        python ${segment_py} \
            --raw-fname=${hsk_data_dir}/hsk.src.remove.spac \
            --seg-fname=${hsk_data_dir}/hsk.src.${token_suffix} \
            ${flag} --encoding=${encoding}
        sed 's/ //g' ${hsk_data_dir}/hsk.trg > ${hsk_data_dir}/hsk.trg.remove.spac
        python ${segment_py} \
            --raw-fname=${hsk_data_dir}/hsk.trg.remove.spac \
            --seg-fname=${hsk_data_dir}/hsk.trg.${token_suffix} \
            ${flag} --encoding=${encoding}
    fi
fi

if [[ ! -f "$blcu_data_dir/lang8.src.$token_suffix" || ! -f "$blcu_data_dir/lang8.trg.$token_suffix" || "$force_redo_remove_same_and_seg" == true ]]; then
    if [[ ${fusion_mode} == 3 ]]; then
        sed 's/ //g' ${blcu_data_dir}/lang8.src > ${blcu_data_dir}/lang8.src.remove.spac
        python ${segment_py} \
            --raw-fname=${blcu_data_dir}/lang8.src.remove.spac \
            --seg-fname=${blcu_data_dir}/lang8.src.${token_suffix} \
            ${flag} --encoding=${encoding}
        sed 's/ //g' ${blcu_data_dir}/lang8.trg > ${blcu_data_dir}/lang8.trg.remove.spac
        python ${segment_py} \
            --raw-fname=${blcu_data_dir}/lang8.trg.remove.spac \
            --seg-fname=${blcu_data_dir}/lang8.trg.${token_suffix} \
            ${flag} --encoding=${encoding}
    fi
fi

if [[ ${fusion_mode} == 1 ]]; then
    fusion_contain="NLPCC_betterseg"
    less ${nlpcc_betterseg_src}.${token_suffix} > ${fusion_src}.${token_suffix}
    less ${nlpcc_betterseg_trg}.${token_suffix} > ${fusion_trg}.${token_suffix}
elif [[ ${fusion_mode} == 2 ]]; then
    fusion_contain="NLPCC_betterseg+HSK"
    cat ${nlpcc_betterseg_src}.${token_suffix} ${hsk_data_dir}/hsk.src.${token_suffix} > ${fusion_src}.${token_suffix}
    cat ${nlpcc_betterseg_trg}.${token_suffix} ${hsk_data_dir}/hsk.trg.${token_suffix} > ${fusion_trg}.${token_suffix}
elif [[ ${fusion_mode} == 3 ]]; then
    fusion_contain="NLPCC_betterseg+HSK+BLCU"
    cat ${nlpcc_betterseg_src}.${token_suffix} ${hsk_data_dir}/hsk.src.${token_suffix} ${blcu_data_dir}/lang8.src.${token_suffix} > ${fusion_src}.${token_suffix}
    cat ${nlpcc_betterseg_trg}.${token_suffix} ${hsk_data_dir}/hsk.trg.${token_suffix} ${blcu_data_dir}/lang8.trg.${token_suffix} > ${fusion_trg}.${token_suffix}
else
    echo "illegal fusion mode, got $fusion_mode"
    exit -3
fi

# must to do remove same operation here
python ${preprocessing_py} remove-same-src-trg \
    --src-fname=${fusion_src}.${token_suffix} \
    --trg-fname=${fusion_trg}.${token_suffix} \
    --encoding=${encoding}

# remove len ratio illegal sentence pairs
python ${preprocessing_py} remove-len-ratio \
    --src-fname=${fusion_src}.${token_suffix}.removesame \
    --trg-fname=${fusion_trg}.${token_suffix}.removesame \
    --low=${low} --high=${high} --encoding=${encoding}

# remove too long or too short sen pairs
python ${preprocessing_py} remove-long-short \
    --src-fname=${fusion_src}.${token_suffix}.removesame.remove_low${low}_high${high} \
    --trg-fname=${fusion_trg}.${token_suffix}.removesame.remove_low${low}_high${high} \
    --short=${short} --long=${long} --encoding=${encoding}

python ${random_split_py} \
    --all-src-fname=${fusion_src}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long} \
    --all-trg-fname=${fusion_trg}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long} \
    --output-dir=${fusion_data_dir} \
    --dev-samples-num=${dev_samples_num} --encoding=${encoding}


processed_dir=${BASE_DIR}/training/zh-${model_level}-processed-${fusion_contain}-remove-short${short}-long${long}-low${low}-high${high}
BPE_MODEL_DIR=${BASE_DIR}/training/models/zh_bpe_model_${fusion_contain}_remove_short${short}_long${long}_low${low}_high${high}
if [[ "${model_level}" == 'bpe' ]]; then
    ${preprocess_yangzl_sh} ${processed_dir} true false \
        ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.train.tok \
        ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.dev.tok \
        ${BPE_MODEL_DIR} 30000 ${src_vocab_size} ${trg_vocab_size}
else
    ${preprocess_yangzl_sh} ${processed_dir} false false \
        ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.train.tok \
        ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.dev.tok \
        None -1 ${src_vocab_size} ${trg_vocab_size}
fi

${convert_parallel_to_m2_sh} ${edit_creator_sh} ${processed_dir} \
    ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.dev.tok.src \
    ${fusion_src//.src/''}.${token_suffix}.removesame.remove_low${low}_high${high}.remove_short${short}_long${long}.dev.tok.trg

if [[ "${model_level}" == 'bpe' ]]; then
    if [[ ${which_pretrained_embed} == 'wang2vec' ]]; then
        EMBED_URL=${BASE_DIR}/data/embeddings/chinesegigawordv5.jian.jieba.seg.bpe.structed.skipngram.500d.txt
    elif [[ ${which_pretrained_embed} == 'word2vec' ]]; then
        EMBED_URL=${BASE_DIR}/data/embeddings/wiki.zh.jian.jieba.seg.bpe.word2vec.skipgram.500d.txt
    elif [[ ${which_pretrained_embed} == 'fasttext' ]]; then
        EMBED_URL=${BASE_DIR}/data/embeddings/wiki.zh.jian.jieba.seg.bpe.fasttext.500d.txt
    elif [[ ${which_pretrained_embed} == 'bpemb' ]]; then
        EMBED_URL=noEmb
    else
        EMBED_URL=noEmb
    fi
elif [[ ${model_level} == 'word' ]]; then
    if [[ ${which_pretrained_embed} == 'wang2vec' ]]; then
        EMBED_URL=${BASE_DIR}/data/embeddings/wiki.zh.jian.jieba.seg.word.structed.skipngram.500d.txt
    else
        EMBED_URL=noEmb
    fi
else
    if [[ ${which_pretrained_embed} == 'wang2vec' ]]; then
        EMBED_URL=${BASE_DIR}/data/embeddings/wiki.zh.jian.char.structed.skipngram.500d.txt
    else
        EMBED_URL=noEmb
    fi
fi

model_name=${processed_dir##*/}
model_name=${model_name//processed/$which_pretrained_embed}
model_name=${model_name//-/_}
model_name=${model_arch}_${model_name}
if [[ "${model_level}" == 'bpe' ]]; then
    ${full_process_sh} ${model_arch} ${processed_dir}/bin ${BPE_MODEL_DIR} \
        ${EMBED_URL} ${jieba_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPU_used_training} ${GPU_used_inference} ${model_name} \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
elif [[ "${model_level}" == 'word' ]]; then
    ${full_process_sh} ${model_arch} ${processed_dir}/bin None \
        ${EMBED_URL} ${jieba_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPU_used_training} ${GPU_used_inference} ${model_name} \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
else
    ${full_process_sh} ${model_arch} ${processed_dir}/bin None \
        ${EMBED_URL} ${char_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPU_used_training} ${GPU_used_inference} ${model_name} \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
fi

