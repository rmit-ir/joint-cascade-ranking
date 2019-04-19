import pytest

import numpy as np

import fixtures

from cascade import Cascade
from cascade import Ranker


def dummy_cascade():
    dummy_config = fixtures.cascade_config()
    dummy_cost = fixtures.feature_costs()
    return Cascade(dummy_config, dummy_cost)


def cascade(num_stages, cutoffs, score_type='independent'):
    X, y, qid = fixtures.train_data()
    X_val, y_val, qid_val = fixtures.valid_data()
    config = fixtures.cascade_config(num_stages=num_stages, cutoffs=cutoffs)
    config['score_type'] = score_type
    cost = fixtures.feature_costs()
    cascade = Cascade(config, cost)
    cascade.create_boosters(X, y, qid, X_val, y_val, qid_val)
    return cascade


def ranker():
    return Ranker({})
