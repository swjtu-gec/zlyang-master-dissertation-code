#!/usr/bin/env bash

set -x
set -e

if [[ $# != 4 ]]; then
    echo "Usage: `basename $0` <url to m2scorer script> <gec system output> <url to gold edit> <whether to use re-ranked system output to evaluate model performance(e.g: true or false)>"
    exit -1
fi

m2scorer=$1
system_out=$2
gold_edit=$3
use_reranked_eval=$4

echo '==== use system output to evaluate model performance ===='
eval_starttime=$(date +%s)

# 去除空格（分词信息）
sed 's/ //g' ${system_out} > ${system_out}.remove.spac
# 使用pkunlp进行分词，得到文件xxx.remove.spac.seg
python pkunlp_segment.py --corpus ${system_out}.remove.spac --segsuffix seg
# 使用m2score计算得分
${m2scorer} ${system_out}.remove.spac.seg ${gold_edit}

eval_endtime=$(date +%s)
cost=$((eval_endtime - eval_starttime))
echo "evaluate end. cost ${cost}s"


if [[ "${use_reranked_eval}" == 'true' ]]; then
    SYSTEM_OUT_RESCORED=`dirname ${system_out}`/output.reranked.tok.txt
    if [[ -f "${SYSTEM_OUT_RESCORED}" ]]; then
        echo '==== use re-ranked system output to evaluate model performance ===='
        eval_starttime=$(date +%s)

        sed 's/ //g' ${SYSTEM_OUT_RESCORED} > ${SYSTEM_OUT_RESCORED}.remove.spac
        python pkunlp_segment.py --corpus ${SYSTEM_OUT_RESCORED}.remove.spac --segsuffix seg
        ${m2scorer} ${SYSTEM_OUT_RESCORED}.remove.spac.seg ${gold_edit}

        eval_endtime=$(date +%s)
        cost=$((eval_endtime - eval_starttime))
        echo "evaluate end. cost ${cost}s"
    fi
fi

