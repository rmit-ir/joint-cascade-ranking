import math
import enum
import logging
import collections

from operator import attrgetter

import numpy as np
import lightgbm as lgb

import scipy
import sklearn.utils

from .util import group_offsets
from .util import group_counts


def _get_defaults():
    return {
        'boosting_type': "gbdt",
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 10,
        'max_bin': 255,
        'subsample_for_bin': 50000,
        'objective': "lambdarank",
        'min_split_gain': 0,
        'min_child_weight': 5,
        'min_child_samples': 10,
        'subsample': 1,
        'subsample_freq': 0,
        'colsample_bytree': 1,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'scale_pos_weight': 1,
        'is_unbalance': False,
        'seed': 0,
        'nthread': -1,
        'silent': True,
        'sigmoid': 1.0,
        'max_position': 20,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'uniform_drop': False,
        'xgboost_dart_mode': False,
        'use_missing': True,
        'cegb_tradeoff': 0.001,
        'cegb_penalty_split': 0.0,
        'cegb_independent_branches': False,
        'cegb_predict_lazy': True,
        'cegb_penalty_feature_lazy': '',
    }


def kappa(stage_score, cutoff_point, qid):
    res = []
    for a, b in group_offsets(qid):
        pivot = min(b - a, cutoff_point)
        values = np.full(
            b - a, -np.partition(-stage_score[a:b], pivot - 1)[pivot - 1])
        res.append(values)
    if len(res) > 0:
        res = np.concatenate(res)
    return res


class ScoreType(enum.Enum):
    Independent = 0x0
    Full = 0x1
    Weak = 0x2

    # alias for lowercase key lookup
    independent = 0x0
    full = 0x1
    weak = 0x2


class IndicatorType(enum.Enum):
    Relu = 0x0
    Logistic = 0x1
    Hard = 0x2
    Flat = 0x3

    # alias for lowercase key lookup
    relu = 0x0
    logistic = 0x1
    hard = 0x2
    flat = 0x3


class Ranker:
    def __init__(self, params):
        self.params = params
        self.booster = None
        self.train = None
        self.val = None
        self.best_iteration = 0
        self.best_score = 0.
        self.callbacks = set()
        self.stopped = False
        self.indicator_score = np.zeros(0)
        self.indicator_derivative = np.zeros(0)
        self.predict = None
        self.kappa = None
        self.mask = None
        self.estimate = None
        self.sigma = 0.1
        self.delta = 0.1
        self.weights = np.zeros(0)
        self.set_indicator_type('logistic')
        for k, v in params.items():
            setter = "set_{}".format(k)
            if hasattr(self, setter) and callable(getattr(self, setter)):
                getattr(self, setter)(v)
            else:
                setattr(self, k, v)

    def set_booster(self, booster):
        self.booster = booster

    def set_val(self, val):
        self.val = val

    def set_train(self, train):
        self.train = train

    def set_weights(self, weights):
        self.weights = weights

    def set_indicator_type(self, indicator_type: str):
        self.indicator_type = IndicatorType[indicator_type]

    def indicator_func(self):
        """ ReLU is normalized between [0-1]."""
        if self.indicator_type == IndicatorType.Logistic:
            self.indicator_score = scipy.special.expit(
                (self.predict - self.kappa) / self.sigma)
            self.indicator_derivative = self.indicator_score * (
                1 - self.indicator_score) / self.sigma
        elif self.indicator_type == IndicatorType.Relu:
            self.indicator_score = 0.5 * (1 + np.minimum(
                1, np.maximum(-1, (self.predict - self.kappa) / self.delta)))
            self.indicator_derivative = (
                (self.kappa - self.delta < self.predict) &
                (self.predict < self.kappa + self.delta)) / (2 * self.delta)
        elif self.indicator_type == IndicatorType.Flat:
            raise NotImplementedError("Flat indicator not implemented")
        elif self.indicator_type == IndicatorType.Hard:
            raise NotImplementedError("Hard indicator not implemented")
        else:
            raise RuntimeError("Unkown inidicator function")


class Cascade:
    SCORE_MASK = -999.
    EPSILON = 1e-4

    def __init__(self, config, cost):
        self.rankers = []
        self.set_score_type('independent')
        self.set_config(config)
        self.set_cost(cost)
        for s in self.stages:
            self.rankers.append(Ranker({**_get_defaults(), **self.lgbm, **s}))

    def __iter__(self):
        yield from self.rankers

    def __len__(self):
        return len(self.rankers)

    def set_score_type(self, score_type: str):
        self.score_type = ScoreType[score_type]

    def set_cost(self, cost_arr):
        # features are zero-indexed
        self.cost_arr = cost_arr
        cost_str = ','.join([
            "{}:{:d}".format(x, math.trunc(y))
            for x, y in enumerate(cost_arr.tolist())
        ])
        for s in self.stages:
            s['cegb_penalty_feature_lazy'] = cost_str

    def set_config(self, config):
        for k, v in config.items():
            setter = "set_{}".format(k)
            if hasattr(self, setter) and callable(getattr(self, setter)):
                getattr(self, setter)(v)
            else:
                setattr(self, k, v)

    def create_boosters(self, X, y, qid, X_val, y_val, qid_val):
        previous = None
        for i, r in enumerate(self):
            if i > 0:
                previous = self.rankers[i - 1].booster
            train = lgb.Dataset(X,
                                y,
                                group=group_counts(qid),
                                free_raw_data=False,
                                silent=True)
            train._update_params(r.params)
            train.set_weight(np.ones(X.shape[0]))
            self.rankers[i].set_weights(np.ones(X.shape[0]))
            val = lgb.Dataset(X_val,
                              y_val,
                              group=group_counts(qid_val),
                              free_raw_data=False,
                              silent=True)
            val.set_reference(train)
            val._update_params(r.params)
            # previous booster is linked to current stage via the argument
            # `predecessor`
            booster = lgb.Booster(r.params,
                                  train_set=train,
                                  predecessor=previous,
                                  silent=True)
            # assumes both train and valid data are passed to `cascade.fit`, no
            # checks are performed in `cascade.fit`
            booster.set_train_data_name("train_ranker_{}".format(i + 1))
            booster.add_valid(val, "valid_ranker_{}".format(i + 1))
            self.rankers[i].set_train(train)
            self.rankers[i].set_val(val)
            self.rankers[i].set_booster(booster)

    def delete_data(self):
        # drop LightGBM pointers, etc so model can be pickled
        for r in self:
            r.set_train(None)
            r.set_val(None)
            r.callbacks = set()
            r.callbacks_before_iter = []
            r.callbacks_after_iter = []
            r.weights = []
            r.predict = []
            r.kappa = []

    def max_trees(self):
        max = 0
        for r in self:
            if r.num_trees > max:
                max = r.num_trees
        return max

    def prepare_callbacks(self):
        for i, r in enumerate(self):
            callbacks = set()
            callbacks.add(lgb.callback.print_evaluation())
            if r.early_stopping_rounds is not None:
                verbose_eval = True
                callbacks.add(
                    lgb.callback.early_stopping(r.early_stopping_rounds,
                                                verbose=bool(verbose_eval)))
            callbacks_before_iter = {
                cb
                for cb in callbacks if getattr(cb, 'before_iteration', False)
            }
            callbacks_after_iter = callbacks - callbacks_before_iter
            callbacks_before_iter = sorted(callbacks_before_iter,
                                           key=attrgetter('order'))
            callbacks_after_iter = sorted(callbacks_after_iter,
                                          key=attrgetter('order'))
            r.callbacks = callbacks
            r.callbacks_before_iter = callbacks_before_iter
            r.callbacks_after_iter = callbacks_after_iter

    def fit(self,
            X,
            y,
            qid,
            X_val=None,
            y_val=None,
            qid_val=None,
            approx_grads=False):

        X, y = sklearn.utils.check_X_y(X, y, 'csr')
        if X_val is None:
            X_val, y_val, qid_val = X, y, qid
        else:
            X_val, y_val = sklearn.utils.check_X_y(X_val, y_val, 'csr')

        self.create_boosters(X, y, qid, X_val, y_val, qid_val)
        self.prepare_callbacks()
        results = collections.defaultdict(list)

        init_iteration = 0
        num_boost_round = self.max_trees()
        for epoch in range(init_iteration, init_iteration + num_boost_round):
            """ 1. Check ranker early stopped.
                2. Run before callbacks.
                3. Update booster.
            """
            did_update = False
            for ranker in self:
                if epoch == ranker.num_trees:
                    ranker.stopped = True
                if ranker.stopped:
                    continue
                # TODO lgr: callback to mark booster stopped when num_trees
                # reached for that ranker (may be different for each booster)
                for cb in ranker.callbacks_before_iter:
                    cb(
                        lgb.callback.CallbackEnv(
                            model=ranker.booster,
                            params=ranker.params,
                            iteration=epoch,
                            begin_iteration=init_iteration,
                            end_iteration=init_iteration + num_boost_round,
                            evaluation_result_list=None))
                ranker.booster.update()
                did_update = True
            if not did_update:
                logging.info("all rankers early stopped")
                break
            """ 1. Run `predcit`.
                2. Calculate indicator function.
                3. Calculate document weights.
                4. Update training data with new weights according to `ScoreType`.
            """
            self.predict(X, qid, is_train=True)

            # compute I[x \in X_j] for each stage j.
            self.ranker_indicator()

            # compute and store I[x \in X_j] in `ranker.weights`.
            self.ranker_weights()

            # update training data with the newly computed document weights
            # according to the scoring function G_j(x) in Table 1.
            for i, ranker in enumerate(self):
                G_j = self.collect_weights(i, ranker)
                if not approx_grads:
                    G_j = self.compute_grads(i, ranker, G_j)
                ranker.train.set_weight(G_j)
            """ 1. Run after callbacks.
                2. Handle early stopping.
            """
            for i, ranker in enumerate(self.rankers):
                if ranker.stopped:
                    continue
                evaluation_result_list = []
                # TODO lgr: assumes that we are always evaluating train and valid sets
                # evaluation_result_list.extend(ranker.booster.eval_train(feval=None))
                evaluation_result_list.extend(
                    ranker.booster.eval_valid(feval=None))
                for name, metric, value, _ in evaluation_result_list:
                    logging.info("{} {} {} {}".format(epoch + 1, name, metric,
                                                      value))
                    key = "{}_{}".format(name, metric)
                    results[key].append(value)
                try:
                    for cb in ranker.callbacks_after_iter:
                        cb(
                            lgb.callback.CallbackEnv(
                                model=ranker.booster,
                                params=ranker.params,
                                iteration=epoch,
                                begin_iteration=init_iteration,
                                end_iteration=init_iteration + num_boost_round,
                                evaluation_result_list=evaluation_result_list))
                except lgb.callback.EarlyStopException as earlyStopException:
                    evaluation_result_list = earlyStopException.best_score
                    ranker.stopped = True
                    ranker.booster.best_iteration = earlyStopException.best_iteration + 1
                    ranker.booster.best_score = collections.defaultdict(dict)
                    logging.info("Early stopping:")
                    logging.info("  best iteration: {}".format(
                        ranker.booster.best_iteration))
                    logging.info("  best score(s):")
                    for name, metric, score, _ in evaluation_result_list:
                        ranker.booster.best_score[name][metric] = score
                        logging.info("    {} {} {}".format(
                            name, metric, score))

                    # TODO lgr: Save best iteration and best score, for chained_stopping?
                    # also early-stop all previous rankers
                    if self.chained_stopping:
                        # FIXME: need to get the best iteration and best score
                        # for the previous rankers, the current ranker's best
                        # iteration may not be the best for previous rankers.
                        for prev_ranker in self.rankers[:i]:
                            if not prev_ranker.stopped:
                                prev_ranker.stopped = True

        # count used features
        mask = self.first().booster.feature_importance()
        for i, ranker in enumerate(self.rankers[1:], start=1):
            mask += ranker.booster.feature_importance()
        self.num_features = np.count_nonzero(mask)

        self.delete_data()

        return results

    def predict_reset(self):
        for ranker in self:
            ranker.predict = None
            ranker.kappa = None
            ranker.mask = None
            ranker.estimate = None

    def predict(self, X, qid, is_train=False, num_iteration=-1):
        """ The `Cascade.SCORE_MASK` prevents a later stage from giving higher
            scores to documents that are below that later stage's cutoff. This
            prevents documents from leaking across cutoff boundaries.
            This issue is related to the fact that each stage has it's own copy
            of the dataset and therefore makes predictions on all documents
            (one of the goals was to use all data during training).

            The following example starts with Stage 1 scores sorted left to
            right for convenience, note documents remain in the same list
            position between stages:
                
                * `SCORE_MASK` is -99 in these examples.
                * estimate is the final output of a stage's ranker.
                * previous mask details are omitted.

                Stage 1 cutoff:   3
                Stage 1 scores:   1.2, 1.0, 0.5, 0.25, 0.1, -0.03
                Stage 1 mask:     1, 1, 1, 0, 0, 0
                Stage 1 kappa:    0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                Stage 1 estimate: 1.2, 1.0, 0.5, 0.25, 0.1, -0.03
                --
                Stage 2 cutoff:   2
                Stage 2 scores:   1.0, 1.1, 0.5, -99, -99, -99
                Stage 2 mask:     1, 1, 0, 0, 0, 0
                Stage 2 kappa:    1.0, 1.0, 1.0, 1.0, 1.0
                Stage 2 estimate: 1.0, 1.1, 0.5, 0.25, 0.1, -0.03
        """
        prev_mask = 1
        prev_ranker = None

        self.predict_reset()

        for ranker in self:
            if not is_train:
                logging.info("best it {}, best score {}".format(
                    ranker.booster.best_iteration, ranker.booster.best_score))
            full_scores = ranker.booster.predict(X,
                                                 num_iteration=num_iteration)
            ranker.predict = np.where(prev_mask, full_scores,
                                      Cascade.SCORE_MASK)

            self.ranker_apply_cutoff(ranker,
                                     full_scores,
                                     prev_mask,
                                     qid,
                                     is_train=is_train)

            self.ranker_score(ranker, prev_ranker)
            prev_mask = ranker.mask
            prev_ranker = ranker
        return np.copy(self.last().estimate)

    def ranker_apply_cutoff(self, ranker, scores, mask, qid, is_train):
        # During inference documents below the cutoff must have a score of
        # `Cascade.SCORE_MASK` because they can't be used.
        if not is_train:
            scores = ranker.predict
        ranker.kappa = kappa(scores, ranker.cutoff, qid)
        ranker.mask = np.where(ranker.predict >= ranker.kappa, mask, 0)
        return scores

    def ranker_score(self, ranker, prev_ranker=None):
        if ranker == self.first():
            ranker.estimate = ranker.predict
        else:
            if self.score_type is ScoreType.Independent:
                ranker.estimate = np.where(prev_ranker.mask, ranker.predict,
                                           prev_ranker.estimate)
            elif self.score_type is ScoreType.Full:
                ranker.estimate = prev_ranker.estimate + np.where(
                    prev_ranker.mask, ranker.predict, 0)
            elif self.score_type is ScoreType.Weak:
                ranker.estimate = np.where(
                    prev_ranker.mask,
                    np.maximum(ranker.predict, prev_ranker.estimate),
                    prev_ranker.estimate)
            else:
                raise RuntimeError("Unkown `ScoreType`")

    def ranker_indicator(self):
        for ranker in self:
            if self.last() == ranker:
                # I_K(x) is always 0.
                ranker.indicator_score = np.zeros(ranker.weights.shape[0])
                ranker.indicator_derivative = np.zeros(ranker.weights.shape[0])
                break
            # compute and store I_j(x) in `ranker.indicator_score`.
            ranker.indicator_func()

    def ranker_weights(self):
        for i, ranker in enumerate(self):
            ranker.weights = 1 - ranker.indicator_score
            for j in range(i):
                ranker.weights *= self.rankers[j].indicator_score
            ranker.weights = self.clip(ranker.weights)

    def collect_weights(self, index, ranker):
        """ Does not handle ties in the predictions. Assumption is to take the
            weight from the last stage where a tie occurs.
        """
        G_j = 0.
        if self.score_type == ScoreType.Independent:
            G_j = self.rankers[index].weights
        elif self.score_type == ScoreType.Full:
            G_j = sum(r.weights for r in self.rankers[index:])
        elif self.score_type == ScoreType.Weak:
            for r in self.rankers[index:]:
                G_j += np.where(self.rankers[index].predict == r.estimate,
                                r.weights, 0)
        else:
            raise RuntimeError("Unkown `ScoreType`")
        return G_j

    def compute_grads(self, index, ranker, weights):
        S = 0
        if ranker is not self.last():
            for j in range(index, len(self)):
                # clip `denom` to prevent divsion by 0
                denom = (ranker.indicator_score - int(j == index))
                # denom = self.clip(denom)
                denom = np.clip(denom, a_min=1e-10, a_max=None)
                part = (self.rankers[j].weights /
                        denom *
                        self.rankers[j].estimate) # yapf: disable
                S += part
            weights += ranker.indicator_derivative * S
        weights = self.clip(weights)
        return weights

    def cost(self, X, qid, cost):
        ret = {}
        self.cost_arr = cost
        used_features = [x.booster.feature_importance() for x in self]
        # zero out unused features just to be sure (e.g. Y!S1 has 519 features)
        not_used = np.where(X.getnnz(0) == 0)[0]
        if not_used.size > 0:
            self.cost_arr[not_used] = 0
            for f in used_features:
                f[not_used] = 0
        masks = [np.flatnonzero(x) for x in used_features]
        l = list(zip(used_features, masks))
        features_a = l[0][0]
        mask_a = l[0][1]
        stage_cost = np.sum(self.cost_arr[mask_a]) * X.shape[0]
        total_cost = stage_cost.copy()
        total_features = np.abs(features_a)
        ret['stage1_features'] = len(mask_a)
        ret['stage1_cost'] = int(stage_cost / X.shape[0])
        for i, tup in enumerate(l[1:], 1):
            features, mask = tup
            doc_count = np.sum(
                np.minimum(self.rankers[i - 1].cutoff,
                           np.array(group_counts(qid))))
            stage_cost = np.sum(self.cost_arr[np.setdiff1d(
                mask, mask_a)]) * doc_count
            total_cost += stage_cost
            total_features += np.abs(features)
            key = "stage{}_features".format(i + 1)
            ret[key] = len(mask)
            key = "stage{}_cost".format(i + 1)
            ret[key] = int(stage_cost / X.shape[0])
            mask_a = mask
        ret['n_features'] = len(np.flatnonzero(total_features))

        # the overall cost is normalized by the total number of instances
        cascade_cost = total_cost / X.shape[0]
        ret['cost'] = int(cascade_cost)
        return ret

    def first(self):
        return self.rankers[0]

    def last(self):
        return self.rankers[-1]

    def update(self):
        for ranker in self:
            ranker.booster.update()

    def clip(self, arr):
        return np.clip(arr, Cascade.EPSILON, 1.0)
