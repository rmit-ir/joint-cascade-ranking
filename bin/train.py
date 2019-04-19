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
        self.model_dir = 'model'
        self.name = 'default-name'
        self.num_threads = -1


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

    with open(args.config) as f:
        cascade_config = yaml.safe_load(f)
    feature_cost = np.loadtxt(args.feature_cost)

    logging.info("Loading data...")
    X, y, qid = cache.load_svmlight_file(args.train, query_id=True)
    X_val, y_val, qid_val = cache.load_svmlight_file(args.valid, query_id=True)

    # fix feature cost shape for Yahoo! data. There are actually 699 features
    # in the data files, but the cost file has 700.
    feature_cost = feature_cost[:X.shape[1]]

    logging.info("Fit...")
    model = Cascade(cascade_config, feature_cost)
    model.fit(X, y, qid, X_val, y_val, qid_val, approx_grads=args.approx_grads)

    modelpath = Path(config.model_dir) / "{}.pkl".format(config.name)
    logging.info("Save model to {}...".format(modelpath))
    joblib.dump(model, modelpath)


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default='log')
    parser.add_argument("--model_dir", default='model')
    parser.add_argument("--name", default='default')
    parser.add_argument("--num_threads", default=-1, type=int)
    parser.add_argument("--approx_grads", action='store_true')
    parser.add_argument("config", default=None)
    parser.add_argument("feature_cost", default=None)
    parser.add_argument("train", default=None)
    parser.add_argument("valid", default=None)

    main(parser.parse_args())
