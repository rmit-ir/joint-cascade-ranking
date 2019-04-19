#!/usr/bin/env bash

set -e

SPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE="${SPATH}/../.."
BIN="${BASE}/bin"
EVAL="${BASE}/eval"

LOGP="${SPATH}/result/log"
MODELP="${SPATH}/result/model"
SCOREP="${SPATH}/result/score"
RUNP="${SPATH}/result/run"
EVALP="${SPATH}/result/eval"
CONFIGP="${SPATH}/config"
COSTF=${SPATH}/feature_cost.txt
QRELSF=${SPATH}/set1.test.qrels

export PYTHONPATH="${BASE}:$PYTHONPATH"
if [ -f ${BASE}/local.sh ]; then
    . ${BASE}/local.sh
fi

mkdir -p $LOGP $MODELP $SCOREP $RUNP $EVALP

for cfg in $CONFIGP/[fiw]*.yml; do
name=$(basename ${cfg%.yml})
$BIN/train.py \
    --log_dir $LOGP \
    --model_dir $MODELP \
    --name $name \
    --approx_grads \
    $cfg \
    $COSTF \
    $SPATH/set1.{train,valid}.txt

# sklearn.joblib.cache dumps to stdout (grep)
$BIN/evaluate.py \
    --log_dir $LOGP \
    --score_dir $SCOREP \
    $COSTF \
    $SPATH/set1.test.txt \
    $MODELP/${name}.pkl | grep '^(n_features|cost)' > $EVALP/${name}.cost

$EVAL/score2run.sh $QRELSF $SCOREP/${name}.txt > $RUNP/${name}.run
$EVAL/eval.sh $QRELSF $RUNP/${name}.run > $EVALP/${name}.txt
done
