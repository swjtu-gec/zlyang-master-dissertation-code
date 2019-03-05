#!/usr/bin/env bash

set -x
set -e

if [[ $# != 4 ]]; then
    echo "Usage: `basename $0` <dir to concat data(e.g: data/to-concat-data/)> <src extend name(e.g: src)> <target ext name(e.g: trg)> <concat result file prefix(e.g: data/zh-fusion/lang8)>"
    exit -1
fi

cur_path=`pwd`
TO_CONCAT_DIR=$1
src_ext=$2
trg_ext=$3
fusion_prefix=$4

cd ${TO_CONCAT_DIR} && src_fnames=`ls | grep ${src_ext}`
for src_fname in ${src_fnames}
do
    src_fname=${TO_CONCAT_DIR}/${src_fname}
    trg_fname=${src_fname/%$src_ext/$trg_ext}
    if [[ -f "${trg_fname}" ]]; then
        to_fusion_src="${to_fusion_src} ${src_fname}"
        to_fusion_trg="${to_fusion_trg} ${trg_fname}"
    fi
done

cd ${cur_path}
cat ${to_fusion_src} > ${fusion_prefix}.${src_ext}
cat ${to_fusion_trg} > ${fusion_prefix}.${trg_ext}

echo "lines count: `wc -l ${fusion_prefix}.${src_ext}`"
echo "lines count: `wc -l ${fusion_prefix}.${trg_ext}`"
