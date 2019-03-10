#!/usr/bin/env bash

set -x
set -e

if [[ $# != 4 ]]; then
    echo "Usage: `basename $0` <edit creator script's url> <dir to m2 format file> <the source input> <the target side of a parallel corpus or a system output>"
    exit -1
fi

edit_creator=${1}
output_dir=$2
source=$3
target=${4}

${edit_creator} --output ${output_dir}/dev.m2 ${source} ${target}
cp ${source} ${output_dir}/dev.input.txt

