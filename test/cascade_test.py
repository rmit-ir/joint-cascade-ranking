import pytest

import numpy as np

from cascade import kappa
from cascade import group_offsets
from cascade import Cascade
from cascade import ScoreType
from cascade import IndicatorType

import factories
import fixtures

np.random.seed(42)


def test_kappa_with_empy_qid():
    qid = np.array([])
    cutoff = 5

    assert [] == kappa([], cutoff, qid)


def _make_query_data(num_queries=1, depth=5, random_depth=False):
    queries = np.random.choice(list(range(num_queries)),
                               num_queries,
                               replace=False)
    scores = np.random.uniform(low=0., high=10., size=num_queries * depth)

    qid = []
    for x in queries:
        qid += [x] * depth
    return scores, np.array(qid)


def test_kappa_when_query_equals_cutoff():
    cutoff = 5
    query_depth = 5
    scores, qid = _make_query_data(depth=query_depth)
    topk = sorted(scores, reverse=True)

    res = kappa(scores, cutoff, qid)

    np.testing.assert_almost_equal(res, np.full_like(res, topk[cutoff - 1]))


def test_kappa_score_when_query_shorter_than_cutoff():
    cutoff = 10
    query_depth = 5
    scores, qid = _make_query_data(depth=query_depth)
    topk = sorted(scores, reverse=True)

    res = kappa(scores, cutoff, qid)

    np.testing.assert_almost_equal(res, np.full_like(res,
                                                     topk[query_depth - 1]))


def test_kappa_score_when_query_longer_than_cutoff():
    cutoff = 5
    query_depth = 10
    scores, qid = _make_query_data(depth=query_depth)
    topk = sorted(scores, reverse=True)

    res = kappa(scores, cutoff, qid)

    np.testing.assert_almost_equal(res, np.full_like(res, topk[cutoff - 1]))


def test_single_stage_cascade_resets_predict_attributes():
    cascade = factories.dummy_cascade()

    cascade.predict_reset()

    for ranker in cascade:
        assert None == ranker.predict
        assert None == ranker.kappa
        assert None == ranker.mask
        assert None == ranker.estimate


def test_cascade_first_stage_has_no_score_mask():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=1, cutoffs=[5])
    ranker = cascade.rankers[0]

    cascade.predict(X, qid)

    assert Cascade.SCORE_MASK not in ranker.predict


def test_cascade_first_stage_applies_cutoff():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=1, cutoffs=[2])
    ranker = cascade.rankers[0]
    ranker.booster.update()
    offsets = group_offsets(qid)
    a, b = next(offsets)

    cascade.predict(X, qid)

    expected = (b - a) * [0.01948363]
    np.testing.assert_almost_equal(ranker.kappa[a:b], expected)


def test_cascade_first_stage_applies_mask():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=1, cutoffs=[2])
    ranker = cascade.rankers[0]
    ranker.booster.update()
    offsets = group_offsets(qid)
    a, b = next(offsets)

    cascade.predict(X, qid)

    expected = [0, 0, 1, 1, 0]
    np.testing.assert_almost_equal(ranker.mask[a:b], expected)


def test_cascade_second_stage_applies_cutoff():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.update()
    ranker_two = cascade.rankers[1]
    offsets = group_offsets(qid)
    a, b = next(offsets)

    cascade.predict(X, qid)

    topk = sorted(ranker_two.predict[a:b], reverse=True)
    expected = (b - a) * [topk[1]]
    np.testing.assert_almost_equal(ranker_two.kappa[a:b], expected)


def test_cascade_second_stage_applies_mask():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.update()
    ranker_one = cascade.rankers[0]
    ranker_two = cascade.rankers[1]
    offsets = group_offsets(qid)
    a, b = next(offsets)

    cascade.predict(X, qid)

    expected = [1, 0, 1, 1, 1]
    np.testing.assert_almost_equal(ranker_one.mask[a:b], expected)
    expected = [0, 0, 1, 1, 0]
    np.testing.assert_almost_equal(ranker_two.mask[a:b], expected)


def test_cascade_score_mask_does_not_appear_in_first_stage():
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.update()
    ranker_one = cascade.rankers[0]
    ranker_two = cascade.rankers[1]
    offsets = group_offsets(qid)
    a, b = next(offsets)

    cascade.predict(X, qid, is_train=True)

    assert Cascade.SCORE_MASK not in ranker_one.predict


def test_cascade_uses_score_mask():
    """As per previous implementation, always use the SCORE_MASK during predict
    regardless of whether we are doing training or inference.
    """
    X, _, qid = fixtures.train_data()
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.update()
    ranker_one = cascade.rankers[0]
    ranker_two = cascade.rankers[1]
    offsets = group_offsets(qid)
    a, b = next(offsets)

    for is_train in [True, False]:
        cascade.predict(X, qid, is_train=is_train)

        assert Cascade.SCORE_MASK in ranker_two.predict


def test_cascade_computed_kappa_when_training():
    qid = np.array([1, 1, 1, 1, 1])
    offsets = group_offsets(qid)
    a, b = next(offsets)
    cascade = factories.dummy_cascade()
    ranker = factories.ranker()
    ranker.cutoff = 2
    prev_mask = [1, 1, 0, 1, 1]
    scores = np.array([0.1, 1.0, -0.03, 0.5, 0.25])
    ranker.predict = np.copy(scores)
    # according to previous mask
    ranker.predict[2] = Cascade.SCORE_MASK

    scores = cascade.ranker_apply_cutoff(ranker,
                                         scores,
                                         prev_mask,
                                         qid,
                                         is_train=True)

    expected = [0.5] * 5
    np.testing.assert_almost_equal(ranker.kappa[a:b], expected)
    assert scores is not ranker.predict


def test_cascade_computed_kappa_when_inference():
    qid = np.array([1, 1, 1, 1, 1])
    offsets = group_offsets(qid)
    a, b = next(offsets)
    cascade = factories.dummy_cascade()
    ranker = factories.ranker()
    ranker.cutoff = 2
    prev_mask = [1, 1, 0, 1, 1]
    # put 10. to test if SCORE_MASK is used in `ranker_apply_cutoff`
    scores = np.array([0.1, 1.0, 10., 0.5, 0.25])
    ranker.predict = np.copy(scores)
    # according to previous mask
    ranker.predict[2] = Cascade.SCORE_MASK

    scores = cascade.ranker_apply_cutoff(ranker,
                                         scores,
                                         prev_mask,
                                         qid,
                                         is_train=False)

    expected = [0.5] * 5
    np.testing.assert_almost_equal(ranker.kappa[a:b], expected)
    assert scores is ranker.predict


def test_cascade_first_stage_score_any_type():
    cascade = factories.cascade(num_stages=1, cutoffs=[4])

    for name, member in ScoreType.__members__.items():
        if member.name != name:  # skip alias names
            continue
        cascade.set_score_type(name)
        ranker_one = cascade.rankers[0]
        cascade.ranker_score(ranker_one)

        assert ranker_one.predict is ranker_one.estimate


def test_cascade_second_stage_score_independent_type():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('independent')
    ranker_one = cascade.rankers[0]
    ranker_one.mask = np.array([1, 1, 1, 1, 0])
    ranker_one.estimate = np.array([4., 3., 2., 1., 0.])
    ranker_two = cascade.rankers[1]
    ranker_two.predict = np.array([5., 5., 5., 5., 5.])

    prev_ranker = ranker_one
    cascade.ranker_score(ranker_two, prev_ranker)

    assert ranker_two.predict is not ranker_two.estimate
    np.testing.assert_almost_equal(ranker_two.estimate,
                                   np.array([5., 5., 5., 5., 0.]))


def test_cascade_second_stage_score_full_type():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('full')
    ranker_one = cascade.rankers[0]
    ranker_one.mask = np.array([1, 1, 1, 1, 0])
    ranker_one.estimate = np.array([4., 3., 2., 1., 0.])
    ranker_two = cascade.rankers[1]
    ranker_two.predict = np.array([5., 5., 5., 5., 5.])

    prev_ranker = ranker_one
    cascade.ranker_score(ranker_two, prev_ranker)

    assert ranker_two.predict is not ranker_two.estimate
    np.testing.assert_almost_equal(ranker_two.estimate,
                                   np.array([9., 8., 7., 6., 0.]))


def test_cascade_second_stage_score_weak_type():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('weak')
    ranker_one = cascade.rankers[0]
    ranker_one.mask = np.array([1, 1, 1, 1, 0])
    ranker_one.estimate = np.array([4., 3., 2., 1., 0.])
    ranker_two = cascade.rankers[1]
    ranker_two.predict = np.array([3., 5., 1., 5., 5.])

    prev_ranker = ranker_one
    cascade.ranker_score(ranker_two, prev_ranker)

    assert ranker_two.predict is not ranker_two.estimate
    np.testing.assert_almost_equal(ranker_two.estimate,
                                   np.array([4., 5., 2., 5., 0.]))


def test_cascade_second_stage_score_unkown_type():
    cascade = factories.cascade(num_stages=2, cutoffs=[2, 4])
    ranker_one = cascade.rankers[0]
    ranker_two = cascade.rankers[1]

    with pytest.raises(RuntimeError):
        cascade.score_type = 'unkown-foo-bar-baz'
        prev_ranker = ranker_one
        cascade.ranker_score(ranker_two, prev_ranker)


def test_cascade_set_unkown_score_type():
    cascade = factories.cascade(num_stages=1, cutoffs=[4])

    with pytest.raises(KeyError):
        cascade.set_score_type('unkown-foo-bar-baz')


def test_ranker_set_unkown_indicator_type():
    ranker = factories.ranker()

    with pytest.raises(KeyError):
        ranker.set_indicator_type('unkown-foo')


def test_ranker_indicator_function_logistic():
    ranker = factories.ranker()
    ranker.set_indicator_type('logistic')
    ranker.sigma = 0.1
    ranker.cutoff = 5
    ranker.predict = np.array(
        [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
    ranker.kappa = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

    ranker.indicator_func()

    # indicator score
    expected = [
        4.5397869e-05, 5.5277864e-04, 6.6928509e-03, 7.5858180e-02,
        5.0000000e-01, 9.2414182e-01, 9.9330715e-01, 9.9944722e-01,
        9.9995460e-01
    ]
    np.testing.assert_almost_equal(expected, ranker.indicator_score)

    # indicator derivative
    expected = [
        4.5395808e-04, 5.5247307e-03, 6.6480567e-02, 7.0103717e-01,
        2.5000000e+00, 7.0103717e-01, 6.6480567e-02, 5.5247307e-03,
        4.5395808e-04
    ]
    np.testing.assert_almost_equal(expected, ranker.indicator_derivative)


def test_ranker_indicator_function_relu():
    ranker = factories.ranker()
    ranker.set_indicator_type('relu')
    ranker.delta = 0.1
    ranker.cutoff = 3
    ranker.predict = np.array(
        [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
    ranker.kappa = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

    ranker.indicator_func()

    # indicator score
    expected = [0., 0., 0., 0., 0.5, 1., 1., 1., 1.]
    np.testing.assert_almost_equal(expected, ranker.indicator_score)

    # indicator derivative
    expected = [0., 0., 0., 0., 5., 0., 0., 0., 0.]
    np.testing.assert_almost_equal(expected, ranker.indicator_derivative)


def test_cascade_last_ranker_indicator_is_zero():
    cascade = factories.cascade(num_stages=1, cutoffs=[1])
    ranker = cascade.last()
    ranker.predict = np.array([1., 2., 3.])
    ranker.kappa = np.array([1., 1., 1.])

    cascade.ranker_indicator()

    assert 0. == ranker.indicator_score.all()
    assert 0. == ranker.indicator_derivative.all()


def test_cascade_collect_weights_independent_scoring():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('independent')
    ranker_one = cascade.rankers[0]
    ranker_one.weights = np.array([1., 2., 3.])
    ranker_two = cascade.rankers[1]
    ranker_two.weights = np.array([3., 2., 1.])

    result = cascade.collect_weights(0, ranker_one)
    expected = [1., 2., 3.]
    np.testing.assert_almost_equal(expected, result)

    result = cascade.collect_weights(1, ranker_two)
    expected = [3., 2., 1.]
    np.testing.assert_almost_equal(expected, result)


def test_cascade_collect_weights_full_scoring():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('full')

    ranker_one = cascade.rankers[0]
    ranker_one.weights = np.array([0.75, 0.5, 0.2])
    ranker_two = cascade.rankers[1]
    ranker_two.weights = np.array([0.75, 0.3, 0.1])

    result = cascade.collect_weights(0, ranker_one)
    expected = [1.5, 0.8, 0.3]
    np.testing.assert_almost_equal(expected, result)

    result = cascade.collect_weights(1, ranker_two)
    expected = [0.75, 0.3, 0.1]
    np.testing.assert_almost_equal(expected, result)


def test_cascade_collect_weights_weak_scoring():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    cascade.set_score_type('weak')

    ranker_one = cascade.rankers[0]
    ranker_one.weights = np.array([0.7, 0.5, 0.2])
    ranker_one.predict = np.array([0.1, 0.2, 0.5])
    ranker_one.estimate = np.array([0.1, 0.2, 0.5])
    ranker_two = cascade.rankers[1]
    ranker_two.weights = np.array([0.3, 0.3, 0.1])
    ranker_two.predict = np.array([-0.1, 0.3, 0.7])
    ranker_two.estimate = np.array([0.1, 0.3, 0.7])

    result = cascade.collect_weights(0, ranker_one)
    expected = [1.0, 0.5, 0.2]
    np.testing.assert_almost_equal(expected, result)

    result = cascade.collect_weights(1, ranker_two)
    expected = [0.0, 0.3, 0.1]
    np.testing.assert_almost_equal(expected, result)


def test_cascade_compute_grads_last_stage():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])
    weights = np.array([1., 0.5, 0.25])

    grads = cascade.compute_grads(1, cascade.last(), weights)
    expected = [1., 0.5, 0.25]
    np.testing.assert_almost_equal(expected, grads)


def test_cascade_compute_grads():
    cascade = factories.cascade(num_stages=2, cutoffs=[4, 2])

    ranker_one = cascade.rankers[0]
    ranker_one.weights = np.array([0.5, 0.5, 0.5])
    ranker_one.estimate = np.array([-0.5, 0.2, 1.])
    ranker_one.indicator_score = np.array([0.5, 0.5, 0.5])
    ranker_one.indicator_derivative = np.array([0.25, 0., 0.5])

    ranker_two = cascade.rankers[1]
    ranker_two.weights = np.array([0.7, 0.7, 0.7])
    ranker_two.estimate = np.array([0.25, 0.6, 0.3])
    ranker_two.indicator_score = np.array([0.5, 0.5, 0.5])
    ranker_two.indicator_derivative = np.array([0., 0., 0.])

    weights = cascade.compute_grads(0, ranker_one, ranker_one.weights)
    expected = [Cascade.EPSILON, 0.5, 1.0]
    np.testing.assert_almost_equal(expected, weights, decimal=4)
