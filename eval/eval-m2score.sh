#!/bin/bash

set -x
set -e

source ../paths.sh

if [[ $# != 1 ]]; then
    echo 'Usage: system_name'
    exit -1
fi

SYSTEM_NAME=$1

SYSTEM_OUT=./${SYSTEM_NAME}/output.tok.txt
SYSTEM_OUT_RESCORED=./${SYSTEM_NAME}/output.reranked.tok.txt
GOLD_EDIT=${DATA_DIR}/test/conll14st-test/conll14st-test.m2

echo '==== use system output to evaluate model performance ===='
eval_starttime=$(date +%s)
./m2scorer/scripts/m2scorer.py ${SYSTEM_OUT} ${GOLD_EDIT}
eval_endtime=$(date +%s)
cost=$((eval_endtime - eval_starttime))
echo "evaluate end. cost ${cost}s"

if [[ -f ${SYSTEM_OUT_RESCORED} ]]; then
    echo '==== use re-ranked system output to evaluate model performance ===='
    eval_starttime=$(date +%s)
    ./m2scorer/scripts/m2scorer.py ${SYSTEM_OUT_RESCORED} ${GOLD_EDIT}
    eval_endtime=$(date +%s)
    cost=$((eval_endtime - eval_starttime))
    echo "evaluate end. cost ${cost}s"
fi