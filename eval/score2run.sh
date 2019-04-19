#!/usr/bin/env bash

set -e

qrels=$1
scores=$2

if [ $# -ne 2 ]; then
    echo "usage: score2run.sh <qrels> <scorefile> <rundir>" 1>&2
    exit 1
fi

paste -d' ' $qrels $scores | awk '{print $1, "Q0", $3, 0, $5, "cascade"}'
