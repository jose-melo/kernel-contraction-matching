import warnings

import numpy as np
import pytest

from karkcm import KCM
from karkcm.data import subsample


def _blob(n=250, d=5, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_one_sample_is_refused():
    with pytest.raises(ValueError, match="1 sample"):
        KCM(assume_scaled=True).fit(_blob()[:1])


def test_two_samples_is_the_minimum_that_fits():
    X = _blob()[:2]
    m = KCM(assume_scaled=True).fit(X)
    assert m.n_reference_ == 2
    assert m.bandwidth_ > 0.0
    assert np.all(np.isfinite(m.anomaly_score(X)))
    assert m.loo_score_samples_.shape == (2,)


def test_single_feature():
    X = _blob(n=200, d=1)
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    assert m.n_features_in_ == 1
    assert m.reference_.shape == (200, 1)
    assert m.reconstruct(X).shape == (200, 1)
    assert np.all(np.isfinite(m.anomaly_score(X)))
    far = np.array([[500.0]])
    assert m.predict(far)[0] == -1
    assert m.predict(np.array([[float(np.median(X))]]))[0] == 1


def test_constant_feature_column():
    X = _blob()
    X[:, 2] = 7.0
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    assert np.all(np.isfinite(m.anomaly_score(X)))
    assert np.all(np.isfinite(m.reconstruct(X)))
    assert np.allclose(m.reconstruct(X)[:, 2], 7.0, rtol=0.0, atol=1e-9)


def test_constant_column_does_not_divide_by_zero_in_the_scaling_warning():
    """The warning predicate divides by the per-feature sd, which is 0 in column 2."""
    X = _blob()
    X[:, 2] = 7.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KCM(random_state=0).fit(X)


def test_all_features_constant():
    X = np.full((40, 3), 2.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = KCM(random_state=0).fit(X)
    assert np.all(np.isfinite(m.anomaly_score(X)))
    assert np.max(m.anomaly_score(X)) < 1e-9
    assert set(np.unique(m.predict(X))) == {1}


def test_duplicate_rows():
    X = np.repeat(_blob(n=150, d=3), 2, axis=0)
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    scores = m.anomaly_score(X)
    assert np.all(np.isfinite(scores))
    assert np.array_equal(scores[0::2], scores[1::2])
    assert m.bandwidth_at_grid_edge_ is True


def test_all_rows_identical():
    X = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
    m = KCM(assume_scaled=True).fit(X)
    scores = m.anomaly_score(X)
    assert np.all(np.isfinite(scores))
    assert np.max(scores) < 1e-9
    assert m.offset_ <= 0.0


def test_float32_matches_float64():
    X32 = _blob().astype(np.float32)
    X64 = X32.astype(np.float64)
    a = KCM(assume_scaled=True, random_state=0).fit(X32)
    b = KCM(assume_scaled=True, random_state=0).fit(X64)
    assert a.reference_.dtype == np.float64
    assert np.array_equal(a.reference_, b.reference_)
    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.anomaly_score(X32), b.anomaly_score(X64))
    assert np.array_equal(a.anomaly_score(X32), a.anomaly_score(X64))
    assert a.anomaly_score(X32).dtype == np.float64


def test_non_contiguous_input():
    X = _blob()
    view = X[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    a = KCM(assume_scaled=True, random_state=0).fit(view)
    b = KCM(assume_scaled=True, random_state=0).fit(np.ascontiguousarray(view))
    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.anomaly_score(view), b.anomaly_score(view))
    assert a.reference_.flags["C_CONTIGUOUS"]
    assert a.reference_.flags["OWNDATA"]


def test_fortran_ordered_input():
    X = _blob()
    a = KCM(assume_scaled=True, random_state=0).fit(np.asfortranarray(X))
    b = KCM(assume_scaled=True, random_state=0).fit(X)
    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.anomaly_score(np.asfortranarray(X)), b.anomaly_score(X))


def test_reference_does_not_alias_the_training_array():
    X = _blob()
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    before = m.reference_.copy()
    X[:] = 0.0
    assert np.array_equal(m.reference_, before)


def test_larger_than_reference_cap():
    X = _blob(n=3000, d=4)
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    assert m.n_reference_ == 2000
    assert m.reference_indices_.shape == (2000,)
    assert len(set(m.reference_indices_.tolist())) == 2000
    assert m.reference_indices_.min() >= 0
    assert m.reference_indices_.max() < len(X)
    assert np.array_equal(m.reference_, X[m.reference_indices_])
    assert np.array_equal(m.reference_, subsample(X, 0, 2000))
    assert m.loo_score_samples_.shape == (2000,)
    assert m.anomaly_score(X).shape == (3000,)


def test_reference_indices_are_not_sorted():
    """build_h_grid subsamples by position, so restoring input order would move h."""
    X = _blob(n=3000, d=4)
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    assert not np.all(np.diff(m.reference_indices_) > 0)


def test_custom_reference_cap():
    X = _blob(n=3000, d=4)
    m = KCM(assume_scaled=True, random_state=0, max_reference=500).fit(X)
    assert m.n_reference_ == 500
    assert np.array_equal(m.reference_, subsample(X, 0, 500))


def test_max_reference_none_uses_every_row():
    X = _blob(n=3000, d=4)
    m = KCM(assume_scaled=True, random_state=0, max_reference=None).fit(X)
    assert m.n_reference_ == 3000
    assert np.array_equal(m.reference_indices_, np.arange(3000))
    assert np.array_equal(m.reference_, X)


def test_single_query_row():
    X = _blob()
    m = KCM(assume_scaled=True, random_state=0).fit(X)
    assert m.anomaly_score(X[:1]).shape == (1,)
    assert m.predict(X[:1]).shape == (1,)
    assert m.reconstruct(X[:1]).shape == (1, X.shape[1])
    assert m.kernel_mass(X[:1]).shape == (1,)


def test_scoring_is_row_order_invariant():
    X = _blob()
    m = KCM(assume_scaled=True, random_state=0, chunk_size=7).fit(X)
    order = np.random.default_rng(1).permutation(len(X))
    deviation = np.max(np.abs(m.anomaly_score(X)[order] - m.anomaly_score(X[order])))
    assert deviation <= 1e-12
