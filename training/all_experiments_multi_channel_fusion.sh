#!/usr/bin/env bash

set -e
set -x

source ../paths.sh

multi_channel_fusion_experiment_sh=${BASE_DIR}/training/multi_channel_fusion_experiment.sh

device=0
use_which_channels='1 1 1 0'
force_redo_translation=false
char_models_dir=${BASE_DIR}/training/models/4_ens_fconv_char_random_1_2_4_1000
bpe_models_dir=models/4_ens_fconv_bpe_word2vec_2_4_5_6/


echo "#####################"
echo "re-ranking comparison"
echo "#####################"

echo "#################################"
echo "M1+M2+M3(4 ens, n-best)+nmt-score"
echo "#################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} nmt-score

echo "################################"
echo "M1+M2+M3(4 ens, n-best)+eo-feats"
echo "################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} eo-feats

echo "################################"
echo "M1+M2+M3(4 ens, n-best)+lm-score"
echo "################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score

echo "###########################################"
echo "M1+M2+M3(4 ens, n-best)+lm-score-normalized"
echo "###########################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score-normalized

echo "################################"
echo "M1+M2+M3(4 ens, n-best)+lm-feats"
echo "################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-feats

echo "###########################################"
echo "M1+M2+M3(4 ens, n-best)+lm-feats-normalized"
echo "###########################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-feats-normalized

echo "#############################"
echo "M1+M2+M3(4 ens, n-best)+eo+lm"
echo "#############################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} eo+lm


#################
# model ablations
#################
echo "##############################"
echo "M1+M2+M3(single, n-best)+eo+lm"
echo "##############################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" n-best ${force_redo_translation} \
    ${char_models_dir}/checkpoint_seed1000_best.pt \
    ${bpe_models_dir}/checkpoint_seed2_best.pt \
    eo+lm

echo "#############################"
echo "M1+M2+M3(4 ens, 1-best)+eo+lm"
echo "#############################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" 1-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} eo+lm


#####################
# other people's work
#####################
echo "################################"
echo "M1+M2+M3(4 ens, 1-best)+lm-score"
echo "################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" 1-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score

echo "###########################################"
echo "M1+M2+M3(4 ens, 1-best)+lm-score-normalized"
echo "###########################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "$use_which_channels" 1-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score-normalized

echo "#################################"
echo "M1+M2+M3+M4(4 ens, 1-best)+lm-score"
echo "#################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "1 1 1 1" 1-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score

echo "#################################"
echo "M1+M2+M3+M4(4 ens, 1-best)+lm-score-normalized"
echo "#################################"
${multi_channel_fusion_experiment_sh} \
    ${device} "1 1 1 1" 1-best ${force_redo_translation} \
    ${char_models_dir} ${bpe_models_dir} lm-score-normalized

