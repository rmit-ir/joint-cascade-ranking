#!/bin/bash

# Pass any no. of svm files as args

awk '{
    curr = $2
    if (last != curr) {
        count = 1
        last = curr
    }
    print $2, "0", $2 "." count++, $1
}' "$@" | sed 's/qid://g'
