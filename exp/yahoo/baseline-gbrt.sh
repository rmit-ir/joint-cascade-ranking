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

trees=2000
leaves=31
estop=120
lr=0.05
feature_fraction=0.5

name="baseline.gbrt"
$BIN/baseline_train.py \
    --log_dir $LOGP \
    --model_dir $MODELP \
    --name $name \
    --n_estimators $trees \
    --num_leaves $leaves \
    --learning_rate $lr \
    --early_stopping_rounds $estop \
    --colsample_bytree $feature_fraction \
    $COSTF \
    $SPATH/set1.{train,valid}.txt

$BIN/baseline_evaluate.py \
    --log_dir $LOGP \
    --score_dir $SCOREP \
    $SPATH/set1.test.txt \
    $MODELP/${name}.pkl
$EVAL/score2run.sh $QRELSF $SCOREP/${name}.txt > $RUNP/${name}.run
$EVAL/eval.sh $QRELSF $RUNP/${name}.run > $EVALP/${name}.txt
