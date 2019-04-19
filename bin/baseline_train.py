#!/usr/bin/env python

import time
import argparse
import logging
import pickle

from pathlib import Path

import numpy as np
import lightgbm as lgb

from cascade import cache
from cascade import group_counts

from sklearn.externals import joblib

from cli import AttrDict


class CliConfig(AttrDict):
    GBRT = 'gbrt'
    CEGB = 'cegb'

    def __init__(self):
        self.stamp = time.strftime("%Y%m%d.%H%M%S")
        self.log_fmt = '%(asctime)s: %(levelname)s: %(message)s'
        self.log_dir = 'log'
        self.model_dir = 'model'
        self.name = 'default-name'
        self.n_jobs = -1
        self.boosting_type = 'gbdt'
        self.n_estimators = 100
        self.num_leaves = 31
        self.learning_rate = 0.05
        self.colsample_bytree = 1.0
        self.early_stopping_rounds = None
        self.max_position = 20
        self.subsample_for_bin = 200000
        self.min_child_samples = 20
        self.min_child_weight = 1e-3
        self.sigmoid = 1.0
        self.silent = False
        self.cegb_tradeoff = 0.0
        self.cegb_penalty_split = 0.0
        self.cegb_independent_branches = False
        self.cegb_predict_lazy = False
        # set by cost flie, not cli
        self.cegb_penalty_feature_lazy = "0:0"


def prelude(config):
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format=config.log_fmt, level=logging.INFO)
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path /
                             "{}-{}.log".format(config.name, config.stamp))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(config.log_fmt))
    logging.getLogger().addHandler(fh)


def main(args):
    config = CliConfig.from_parseargs(args)
    prelude(config)
    logging.info("Start...")
    logging.info(config)

    logging.info("Loading data...")
    X, y, qid = cache.load_svmlight_file(args.train, query_id=True)
    X_val, y_val, qid_val = cache.load_svmlight_file(args.valid, query_id=True)

    # fix feature cost shape for Yahoo! data. There are actually 699 features
    # in the data files, but the cost file has 700.
    feature_cost = np.loadtxt(args.feature_cost)
    feature_cost = feature_cost[:X.shape[1]]

    if CliConfig.CEGB == config.boosting_type:
        config.cegb_penalty_feature_lazy = ','.join(
            ["{}:{}".format(i, x) for i, x in enumerate(feature_cost)])

    model = lgb.LGBMRanker(
        objective='lambdarank',
        boosting_type=config.boosting_type,
        n_estimators=config.n_estimators,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        colsample_bytree=config.colsample_bytree,
        max_position=config.max_position,
        subsample_for_bin=config.subsample_for_bin,
        min_child_samples=config.min_child_samples,
        min_child_weight=config.min_child_weight,
        sigmoid=config.sigmoid,
        subsample=1.0,
        subsample_freq=0,
        max_depth=-1,
        cegb_tradeoff=config.cegb_tradeoff,
        cegb_penalty_split=config.cegb_penalty_split,
        cegb_independent_branches=config.cegb_independent_branches,
        cegb_predict_lazy=config.cegb_predict_lazy,
        cegb_penalty_feature_lazy=config.cegb_penalty_feature_lazy,
        nthread=config.n_jobs,
        silent=config.silent)
    logging.info(model)
    model.fit(X,
              y,
              group=group_counts(qid),
              eval_names=['train', 'valid'],
              eval_set=[(X, y), (X_val, y_val)],
              eval_group=[group_counts(qid),
                          group_counts(qid_val)],
              eval_metric='ndcg',
              eval_at=[139],
              early_stopping_rounds=config.early_stopping_rounds)

    used_features = model.feature_importances_
    model.n_features = len(np.flatnonzero(used_features))
    model.cost = np.sum(feature_cost[np.flatnonzero(used_features)])

    logging.info("Best iteration {}...".format(model.best_iteration))
    logging.info("Best score {}...".format(model.best_score))
    modelpath = Path(config.model_dir) / "{}.pkl".format(config.name)
    logging.info("Save model to {}...".format(modelpath))
    joblib.dump(model, modelpath)


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("feature_cost", default=None)
    parser.add_argument("train", default=None)
    parser.add_argument("valid", default=None)
    parser.add_argument("--log_dir", default='log')
    parser.add_argument("--model_dir", default='model')
    parser.add_argument("--name", default='default')
    parser.add_argument("--n_jobs", default=-1, type=int)
    parser.add_argument("--boosting_type", default='gbdt')
    parser.add_argument("--n_estimators", default=100, type=int)
    parser.add_argument("--num_leaves", default=31, type=int)
    parser.add_argument("--learning_rate", default=0.05, type=float)
    parser.add_argument("--colsample_bytree", default=1.0, type=float)
    parser.add_argument("--early_stopping_rounds", default=None, type=int)
    parser.add_argument("--max_position", default=20, type=int)
    parser.add_argument("--subsample_for_bin", default=200000, type=int)
    parser.add_argument("--min_child_samples", default=20, type=int)
    parser.add_argument("--min_child_weight", default=1e-3, type=float)
    parser.add_argument("--sigmoid", default=1.0, type=float)
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--cegb_tradeoff", default=0.0, type=float)
    parser.add_argument("--cegb_penalty_split", default=0.0, type=float)
    parser.add_argument("--cegb_independent_branches", action='store_true')
    parser.add_argument("--cegb_predict_lazy", action='store_true')

    main(parser.parse_args())
