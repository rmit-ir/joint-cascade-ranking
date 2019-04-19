#!/bin/bash

SPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

test ! -f "$SPATH/rbp_eval" \
    && test ! -f "$SPATH/gdeval.pl" \
    && echo "run \`cd $SPATH && make\` first" 1>&2 \
    && exit 1

QRELS=$1
TMP1=$(mktemp -p .)
TMP3=$(mktemp -p .)
TMP5=$(mktemp -p .)
TMP10=$(mktemp -p .)
TMP20=$(mktemp -p .)
MAXJ=$(awk '{print $4}' $QRELS | sort -nu | tail -1)
($SPATH/gdeval.pl -k 1 -j $MAXJ $QRELS $2 | tail -1 > $TMP1)&
($SPATH/gdeval.pl -k 3 -j $MAXJ $QRELS $2 | tail -1 > $TMP3)&
($SPATH/gdeval.pl -k 5 -j $MAXJ $QRELS $2 | tail -1 > $TMP5)&
($SPATH/gdeval.pl -k 10 -j $MAXJ $QRELS $2 | tail -1 > $TMP10)&
($SPATH/gdeval.pl -k 20 -j $MAXJ $QRELS $2 | tail -1 > $TMP20)&
wait
echo -n "ERR_1 "
awk -F, '{printf "%.4f\n", $4}' $TMP1
echo -n "ERR_3 "
awk -F, '{printf "%.4f\n", $4}' $TMP3
echo -n "ERR_5 "
awk -F, '{printf "%.4f\n", $4}' $TMP5
echo -n "ERR_10 "
awk -F, '{printf "%.4f\n", $4}' $TMP10
echo -n "ERR_20 "
awk -F, '{printf "%.4f\n", $4}' $TMP20
echo -n "NDCG_1 "
awk -F, '{printf "%.4f\n", $3}' $TMP1
echo -n "NDCG_3 "
awk -F, '{printf "%.4f\n", $3}' $TMP3
echo -n "NDCG_5 "
awk -F, '{printf "%.4f\n", $3}' $TMP5
echo -n "NDCG_10 "
awk -F, '{printf "%.4f\n", $3}' $TMP10
echo -n "NDCG_20 "
awk -F, '{printf "%.4f\n", $3}' $TMP20
rm $TMP1 $TMP3 $TMP5 $TMP10 $TMP20
$SPATH/rbp_eval -HW -f $(echo "scale=2; 1 / $MAXJ" | bc) -p 0.5 $QRELS $2 | awk '{print "RBP_050", $8 $9}'
$SPATH/rbp_eval -HW -f $(echo "scale=2; 1 / $MAXJ" | bc) -p 0.8 $QRELS $2 | awk '{print "RBP_080", $8 $9}'
$SPATH/rbp_eval -HW -f $(echo "scale=2; 1 / $MAXJ" | bc) -p 0.9 $QRELS $2 | awk '{print "RBP_090", $8 $9}'
$SPATH/rbp_eval -HW -f $(echo "scale=2; 1 / $MAXJ" | bc) -p 0.95 $QRELS $2 | awk '{print "RBP_095", $8 $9}'
