#!/usr/bin/env bash

## This script will do everything, including remove-same, training data processing, remove illegal sentences,
## random split, preprocess_yangzl.sh and full_process.sh

set -x
set -e

source ../paths.sh

preprocessing_py=${BASE_DIR}/data/preprocessing.py
nlpcc2018_gec_process_py=${BASE_DIR}/data/nlpcc2018_gec_process.py
seg_test_input_py=${BASE_DIR}/data/seg_test_input.py
random_split_py=${BASE_DIR}/data/random_split.py
preprocess_yangzl_sh=${BASE_DIR}/training/preprocess_yangzl.sh
convert_parallel_to_m2_sh=${BASE_DIR}/data/convert_parallel_to_m2.sh
full_process_sh=${BASE_DIR}/training/full_process.sh

raw_train_data=${BASE_DIR}/data/nlpcc-2018-traindata/data.train
train_data_remove_same=${BASE_DIR}/data/nlpcc-2018-traindata/data.train.removesame
encoding='utf-8'
nlpcc_betterseg_src=${BASE_DIR}/data/zh-official-nlpcc2018/lang8.src
nlpcc_betterseg_trg=${BASE_DIR}/data/zh-official-nlpcc2018/lang8.trg
remove_lenratio_suffix='.remove.lenratio'
remove_longshort_suffix='.remove.longshort'
dev_samples_num=5000
raw_test_input=${BASE_DIR}/data/test/nlpcc2018-test/source.txt
jieba_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/jieba.seg.txt
char_seg_test_input=${BASE_DIR}/data/test/nlpcc2018-test/char.seg.txt
gold_edit=${BASE_DIR}/data/test/nlpcc2018-test/gold.01
m2scorer_url=${BASE_DIR}/eval/m2scorer/scripts/m2scorer.py
edit_creator_sh=${BASE_DIR}/software/m2scorer/scripts/edit_creator.py


if [[ $# -eq 14 ]]; then
    model_level=$1
    remove_low=$2
    remove_high=$3
    remove_short=$4
    remove_long=$5
    src_vocab_size=$6
    trg_vocab_size=$7
    GPUs_used_training=$8
    GPUs_used_inference=$9
    MAX_TOKENS=${10}
    MAX_SENS=${11}
    random_seed=${12}
    want_ensemble=${13}
    force_redo_remove_same=${14}
else
    echo "Usage: `basename $0` <model level, e.g: bpe, char, word> <low, e.g: 0.1> <high, e.g: 9> <short, e.g: 1> <long, e.g: 100> <src_vocab_size> <trg_vocab_size> <GPU device id to use in training(e.g: '0, 1, 2')> <GPU device id used in test)> <max tokens> <max sentences> <random seed> <whether to use entire model dir to ensemble decoding(e.g: true or false)> <whether to force redo remove same(e.g: false)>"
    exit -1
fi

if [[ "${model_level}" != 'bpe' && "${model_level}" != 'char' && "${model_level}" != 'word' ]]; then
    echo "illegal model level, got ${model_level}"
    exit -2
fi

if [[ "${model_level}" == 'bpe' ]]; then
    EMBED_URL=${BASE_DIR}/data/embeddings/chinesegigawordv5.jian.jieba.seg.bpe.structed.skipngram.500d.txt
else
    EMBED_URL=None_nothing_null
fi

if [[ ! -f ${train_data_remove_same} || "$force_redo_remove_same" == true ]]; then
    python ${preprocessing_py} remove-same \
        --raw-fname=${raw_train_data} \
        --after-fname=${train_data_remove_same} \
        --encoding=${encoding}
fi

if [[ "${model_level}" == 'char' ]]; then
    python ${nlpcc2018_gec_process_py} \
        --nlpcc-traindata=${train_data_remove_same} \
        --train-all-src=${nlpcc_betterseg_src} \
        --train-all-trg=${nlpcc_betterseg_trg} \
        --char-level --encoding=${encoding}
    python ${seg_test_input_py} \
        --raw-fname=${raw_test_input} \
        --seg-fname=${char_seg_test_input} \
        --char-level --encoding=${encoding}
else
    python ${nlpcc2018_gec_process_py} \
        --nlpcc-traindata=${train_data_remove_same} \
        --train-all-src=${nlpcc_betterseg_src} \
        --train-all-trg=${nlpcc_betterseg_trg} \
        --encoding=${encoding}
    python ${seg_test_input_py} \
        --raw-fname=${raw_test_input} \
        --seg-fname=${jieba_seg_test_input} \
        --encoding=${encoding}
fi

# remove len ratio illegal sentence pairs
python ${preprocessing_py} remove-len-ratio \
    --src-fname=${nlpcc_betterseg_src} \
    --trg-fname=${nlpcc_betterseg_trg} \
    --low=${remove_low} --high=${remove_high} --encoding=${encoding}

# remove too long or too short sen pairs
python ${preprocessing_py} remove-long-short \
    --src-fname=${nlpcc_betterseg_src}${remove_lenratio_suffix} \
    --trg-fname=${nlpcc_betterseg_trg}${remove_lenratio_suffix} \
    --short=${remove_short} --long=${remove_long} --encoding=${encoding}

python ${random_split_py} \
    --all-src-fname=${nlpcc_betterseg_src}${remove_lenratio_suffix}${remove_longshort_suffix} \
    --all-trg-fname=${nlpcc_betterseg_trg}${remove_lenratio_suffix}${remove_longshort_suffix} \
    --output-dir=`dirname ${nlpcc_betterseg_src}` \
    --dev-samples-num=${dev_samples_num} --encoding=${encoding}


processed_dir=${BASE_DIR}/training/zh-${model_level}-processed-nlpcc-betterseg-remove-short${remove_short}-long${remove_long}-low${remove_low}-high${remove_high}
BPE_MODEL_DIR=${BASE_DIR}/training/models/zh_bpe_model_nlpcc_betterseg_remove_short${remove_short}_long${remove_long}_low${remove_low}_high${remove_high}
if [[ "${model_level}" == 'bpe' ]]; then
    ${preprocess_yangzl_sh} ${processed_dir} true false `dirname ${nlpcc_betterseg_src}` \
        ${BPE_MODEL_DIR} 30000 ${src_vocab_size} ${trg_vocab_size}
else
    ${preprocess_yangzl_sh} ${processed_dir} false false `dirname ${nlpcc_betterseg_src}` \
        None -1 ${src_vocab_size} ${trg_vocab_size}
fi

${convert_parallel_to_m2_sh} ${edit_creator_sh} ${processed_dir} \
    `dirname ${nlpcc_betterseg_src}`/dev.tok.src \
    `dirname ${nlpcc_betterseg_src}`/dev.tok.trg

model_name=${processed_dir##*/}
model_name=${model_name//processed-/''}
model_name=${model_name//-/_}
model_name=fconv_${model_name}
if [[ "${model_level}" == 'bpe' ]]; then
    ${full_process_sh} ${processed_dir}/bin ${BPE_MODEL_DIR} \
        ${EMBED_URL} ${jieba_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPUs_used_training} ${GPUs_used_inference} ${model_name} \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
elif [[ "${model_level}" == 'word' ]]; then
    ${full_process_sh} ${processed_dir}/bin None \
        ${EMBED_URL} ${jieba_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPUs_used_training} ${GPUs_used_inference} ${model_name}  \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
else
    ${full_process_sh} ${processed_dir}/bin None \
        ${EMBED_URL} ${char_seg_test_input} ${gold_edit} ${m2scorer_url} \
        ${GPUs_used_training} ${GPUs_used_inference} ${model_name}  \
        ${MAX_TOKENS} ${MAX_SENS} ${random_seed} ${want_ensemble} ${processed_dir}
fi

