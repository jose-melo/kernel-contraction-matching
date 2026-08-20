import numpy as np
import pytest

from karkcm.data import MAX_REF, subsample

from .conftest import (
    PAPER_DATASETS,
    SEEDS,
    anchor,
    estimator,
    requires_datasets,
    split,
)

CASES = [(d, s) for d in PAPER_DATASETS for s in SEEDS]
CASES += [("24_mnist", 0), ("30_satellite", 0)]
IDS = [f"{d}-seed{s}" for d, s in CASES]
ALL_DATASETS = tuple(dict.fromkeys(d for d, _ in CASES))

parity_case = pytest.mark.parametrize("dataset,seed", CASES, ids=IDS)


def _score_at(m, X, chunk_size):
    original = m.chunk_size
    m.set_params(chunk_size=chunk_size)
    try:
        return m.anomaly_score(X)
    finally:
        m.set_params(chunk_size=original)


@requires_datasets(*ALL_DATASETS)
@parity_case
def test_bandwidth_bitwise(dataset, seed):
    assert estimator(dataset, seed).bandwidth_ == anchor(dataset, seed).h


@requires_datasets(*ALL_DATASETS)
@parity_case
def test_reference_set_identical(dataset, seed):
    X_tr = split(dataset, seed)[0]
    assert np.array_equal(
        estimator(dataset, seed).reference_, subsample(X_tr, seed, MAX_REF)
    )


@requires_datasets(*ALL_DATASETS)
@parity_case
def test_scores_bitwise_unchunked(dataset, seed):
    """chunk_size is read at score time only, so setting it on a fitted model is exact."""
    X_te = split(dataset, seed)[1]
    got = _score_at(estimator(dataset, seed), X_te, None)
    assert np.array_equal(got, anchor(dataset, seed).score_np(X_te))


@requires_datasets(*ALL_DATASETS)
@parity_case
def test_scores_close_default_chunk(dataset, seed):
    X_te = split(dataset, seed)[1]
    got = _score_at(estimator(dataset, seed), X_te, 4096)
    assert np.max(np.abs(got - anchor(dataset, seed).score_np(X_te))) <= 1e-12


@requires_datasets("24_mnist")
def test_chunk_invariance():
    X_te = split("24_mnist", 0)[1]
    m = estimator("24_mnist", 0)
    ref = _score_at(m, X_te, None)
    for chunk_size in (64, 333, 1024, 4096):
        assert np.max(np.abs(_score_at(m, X_te, chunk_size) - ref)) <= 1e-12
