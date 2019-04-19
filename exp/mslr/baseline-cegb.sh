#!/bin/bash

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
QRELSF=${SPATH}/all.test.qrels

export PYTHONPATH="${BASE}:$PYTHONPATH"
if [ -f ${BASE}/local.sh ]; then
    . ${BASE}/local.sh
fi

mkdir -p $LOGP $MODELP $SCOREP $RUNP $EVALP

trees=667
estop=667
lr=0.05
feature_fraction=0.5
cegb_tradeoff=0.00001

for leaves in 15 31; do
name="baseline.cegb.${leaves}"
for i in {1..5}; do
suffix="${name}.fold${i}"
foldqrels=${SPATH}/fold${i}.test.qrels

$BIN/baseline_train.py \
    --log_dir $LOGP \
    --model_dir $MODELP \
    --name $suffix \
    --boosting_type 'cegb' \
    --n_estimators $trees \
    --num_leaves $leaves \
    --learning_rate $lr \
    --early_stopping_rounds $estop \
    --colsample_bytree $feature_fraction \
    --cegb_tradeoff=$cegb_tradeoff \
    --cegb_predict_lazy \
    $COSTF \
    $SPATH/MSLR-WEB10K/Fold${i}/{train,vali}.txt

$BIN/baseline_evaluate.py \
    --log_dir $LOGP \
    --score_dir $SCOREP \
    $SPATH/MSLR-WEB10K/Fold${i}/test.txt \
    $MODELP/${suffix}.pkl | grep -E '^(n_features|cost)' > $EVALP/${suffix}.cost

$EVAL/score2run.sh $foldqrels $SCOREP/${suffix}.txt > $RUNP/${suffix}.run
done
cat $RUNP/${name}.fold?.run > $RUNP/all.${name}.run
$EVAL/eval.sh $QRELSF $RUNP/all.${name}.run > $EVALP/${name}.txt
grep ^n_features $EVALP/${name}.fold?.cost | awk '{
    sum += $2
}
END {
    printf "avg_n_features: %d\n", sum / NR
}' >> $EVALP/${name}.txt
grep ^cost $EVALP/${name}.fold?.cost | awk '{
    sum += $2
}
END {
    printf "avg_cost: %d\n", sum / NR
}' >> $EVALP/${name}.txt
done
