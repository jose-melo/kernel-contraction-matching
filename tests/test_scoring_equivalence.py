import numpy as np
import pytest

from karkcm import KCM
from karkcm.kcm import (
    KCMAnchor,
    nw_reconstruction,
    nw_residual_score,
    rbf_gram,
)

CHUNK_SIZES = (1, 7, 64, 333, 1024, 4096, 100000, None)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    Q = rng.normal(size=(900, 6)) * 1.5
    m = KCM(assume_scaled=True, random_state=0, max_reference=None, chunk_size=None)
    return X, Q, m.fit(X), KCMAnchor.fit(X, n_grid=21)


def test_bandwidth_selection_is_the_frozen_one(case):
    _, _, m, anchor = case
    assert m.bandwidth_ == anchor.h
    assert np.array_equal(m.bandwidth_grid_, anchor.h_grid)
    assert np.array_equal(m.loo_curve_, anchor.loo_curve)


def test_anomaly_score_is_bitwise_the_frozen_score(case):
    X, Q, m, anchor = case
    assert np.array_equal(m.anomaly_score(Q), anchor.score_np(Q))
    assert np.array_equal(m.anomaly_score(Q), nw_residual_score(X, Q, anchor.h))
    assert np.array_equal(m.anomaly_score(X), anchor.score_np(X))


def test_reconstruct_is_bitwise_the_frozen_reconstruction(case):
    X, Q, m, anchor = case
    assert np.array_equal(m.reconstruct(Q), anchor.reconstruct_np(Q))
    assert np.array_equal(m.reconstruct(Q), nw_reconstruction(X, Q, anchor.h))


def test_kernel_mass_is_the_gram_row_sum(case):
    X, Q, m, anchor = case
    assert np.array_equal(m.kernel_mass(Q), rbf_gram(Q, X, anchor.h).sum(axis=1))


def test_anomaly_score_is_the_residual_of_reconstruct(case):
    _, Q, m, _ = case
    residual = np.linalg.norm(Q - m.reconstruct(Q), axis=1)
    assert np.array_equal(m.anomaly_score(Q), residual)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_invariance(case, chunk_size):
    _, Q, m, anchor = case
    expected_mass = rbf_gram(Q, m.reference_, m.bandwidth_).sum(axis=1)
    m.set_params(chunk_size=chunk_size)
    try:
        assert np.max(np.abs(m.anomaly_score(Q) - anchor.score_np(Q))) <= 1e-12
        assert np.max(np.abs(m.reconstruct(Q) - anchor.reconstruct_np(Q))) <= 1e-12
        assert np.max(np.abs(m.kernel_mass(Q) - expected_mass)) <= 1e-12
    finally:
        m.set_params(chunk_size=None)


def test_chunking_does_not_change_labels(case):
    _, Q, m, _ = case
    reference = m.predict(Q)
    for chunk_size in CHUNK_SIZES:
        m.set_params(chunk_size=chunk_size)
        try:
            assert np.array_equal(m.predict(Q), reference)
        finally:
            m.set_params(chunk_size=None)


def test_chunk_boundaries_cover_every_row(case):
    """A block loop that drops or repeats a tail row would still pass a norm-only check."""
    _, Q, m, _ = case
    m.set_params(chunk_size=7)
    try:
        got = m.anomaly_score(Q)
    finally:
        m.set_params(chunk_size=None)
    assert got.shape == (len(Q),)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - m.anomaly_score(Q))) <= 1e-12
