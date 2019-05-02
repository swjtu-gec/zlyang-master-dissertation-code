#!/usr/bin/env bash

## This script used to copy best checkpoint model to specified directory according to seeds.

set -e
set -x

if [[ $# == 3 ]]; then
    checkpoint_dir=$1
    seeds="$2"
    target_dir=$3
else
    echo "Usage: `basename $0` <model checkpoint directory> <a set of seeds, e.g: '1 2 3 4'> <target dir>"
fi

for seed in ${seeds}
do
    best_checkpoint_model=${checkpoint_dir}/model${seed}/checkpoint_best.pt
    if [[ -f ${best_checkpoint_model} ]]; then
        test ! -e ${target_dir} && mkdir -p ${target_dir}
        cp ${best_checkpoint_model} ${target_dir}/checkpoint_seed${seed}_best.pt
    else
        echo "**********************************************"
        echo "${best_checkpoint_model} not found, skip it..."
        echo "**********************************************"
    fi
done
