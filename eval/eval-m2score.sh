#!/bin/bash

set -x
set -e

source ../paths.sh

if [[ $# != 2 ]]; then
    echo "Usage: `basename $0` <system_name> <gold standard>"
    exit -1
fi

SYSTEM_NAME=$1
GOLD_EDIT=$2

SYSTEM_OUT=./${SYSTEM_NAME}/output.tok.txt
SYSTEM_OUT_RESCORED=./${SYSTEM_NAME}/output.reranked.tok.txt

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
