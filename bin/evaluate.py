#!/usr/bin/env python

import time
import math
import argparse
import logging
import pickle
import yaml

from pathlib import Path

import numpy as np
import lightgbm as lgb

from cascade import cache
from cascade import Cascade

from sklearn.externals import joblib

from cli import AttrDict


class CliConfig(AttrDict):
    def __init__(self):
        self.stamp = time.strftime("%Y%m%d.%H%M%S")
        self.log_fmt = '%(asctime)s: %(levelname)s: %(message)s'
        self.log_dir = 'log'
        self.score_dir = 'score'
        self.name = 'default'


def prelude(config):
    Path(config.score_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format=config.log_fmt, level=logging.INFO)
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path / "score-{}.log".format(config.name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(config.log_fmt))
    logging.getLogger().addHandler(fh)


def main(args):
    config = CliConfig.from_parseargs(args)
    config.name = str(Path(args.model).stem).replace('.pkl', '')
    prelude(config)
    logging.info("Start...")
    logging.info(config)

    logging.info("Loading data...")
    X, _, qid = cache.load_svmlight_file(args.test, query_id=True)

    # fix feature cost shape for Yahoo! data. There are actually 699 features
    # in the data files, but the cost file has 700.
    feature_cost = np.loadtxt(args.feature_cost)
    feature_cost = feature_cost[:X.shape[1]]

    logging.info("Predict...")
    model = joblib.load(args.model)
    scores = model.predict(X, qid)
    cost = model.cost(X, qid, feature_cost)
    print("n_features: {}".format(cost['n_features']))
    print("cost: {}".format(cost['cost']))

    scorepath = Path(config.score_dir) / "{}.txt".format(config.name)
    logging.info("Save scores to {}...".format(scorepath))
    np.savetxt(scorepath, scores, fmt="%.9f")
    logging.info(cost)


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default='log')
    parser.add_argument("--score_dir", default='score')
    parser.add_argument("feature_cost", default=None)
    parser.add_argument("test", default=None)
    parser.add_argument("model", default=None)

    main(parser.parse_args())
