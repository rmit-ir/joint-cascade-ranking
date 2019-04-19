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
QRELSF=${SPATH}/all.test.qrels

export PYTHONPATH="${BASE}:$PYTHONPATH"
if [ -f ${BASE}/local.sh ]; then
    . ${BASE}/local.sh
fi

mkdir -p $LOGP $MODELP $SCOREP $RUNP $EVALP

for cfg in $CONFIGP/[fiw]*.yml; do
name=$(basename ${cfg%.yml})
for i in {1..5}; do
suffix="${name}.fold${i}"
foldqrels=${SPATH}/fold${i}.test.qrels

$BIN/train.py \
    --log_dir $LOGP \
    --model_dir $MODELP \
    --name $suffix \
    --approx_grads \
    $cfg \
    $COSTF \
    $SPATH/MSLR-WEB10K/Fold${i}/{train,vali}.txt

# sklearn.joblib.Memory dumps to stdout (grep)
$BIN/evaluate.py \
    --log_dir $LOGP \
    --score_dir $SCOREP \
    $COSTF \
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
