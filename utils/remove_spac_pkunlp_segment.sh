#!/usr/bin/env bash

set -x
set -e

if [[ $# != 3 ]]; then
    echo "Usage: `basename $0` <gec system output> <url to m2scorer script> <url to gold edit>"
    exit -1
fi

system_out=$1
m2scorer=$2
gold_edit=$3

# 去除空格（分词信息）
sed 's/ //g' ${system_out} > ${system_out}.remove.spac
# 使用pkunlp进行分词，得到文件xxx.remove.spac.seg
python pkunlp_segment.py --corpus ${system_out}.remove.spac --segsuffix seg
# 使用m2score计算得分
${m2scorer} ${system_out}.remove.spac.seg ${gold_edit}
