import numpy as np
import pytest

from karkcm import KCM
from karkcm.data import subsample

CAP = 2000
N_ABOVE_CAP = 3000
N_BELOW_CAP = 300


def _blob(n, d=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


@pytest.fixture(scope="module")
def big():
    return _blob(N_ABOVE_CAP)


@pytest.fixture(scope="module")
def small():
    return _blob(N_BELOW_CAP)


def test_cap_binds_on_the_fixture(big, small):
    """Guards every test below: if the cap stopped binding they would pass vacuously."""
    assert len(big) > CAP
    assert len(small) <= CAP
    assert KCM(assume_scaled=True, random_state=0).fit(big).n_reference_ == CAP
    assert KCM(assume_scaled=True, random_state=0).fit(small).n_reference_ == len(small)


def test_same_random_state_is_bitwise_identical(big):
    a = KCM(assume_scaled=True, random_state=0).fit(big)
    b = KCM(assume_scaled=True, random_state=0).fit(big)
    assert np.array_equal(a.reference_indices_, b.reference_indices_)
    assert np.array_equal(a.reference_, b.reference_)
    assert a.bandwidth_ == b.bandwidth_
    assert np.array_equal(a.bandwidth_grid_, b.bandwidth_grid_)
    assert np.array_equal(a.loo_curve_, b.loo_curve_)
    assert np.array_equal(a.loo_score_samples_, b.loo_score_samples_)
    assert a.offset_ == b.offset_
    assert np.array_equal(a.anomaly_score(big), b.anomaly_score(big))
    assert np.array_equal(a.predict(big), b.predict(big))


def test_different_random_state_changes_the_reference_subset(big):
    a = KCM(assume_scaled=True, random_state=0).fit(big)
    b = KCM(assume_scaled=True, random_state=1).fit(big)
    assert not np.array_equal(a.reference_indices_, b.reference_indices_)
    assert set(a.reference_indices_) != set(b.reference_indices_)
    assert not np.array_equal(a.reference_, b.reference_)


def test_random_state_is_inert_when_the_cap_does_not_bind(small):
    a = KCM(assume_scaled=True, random_state=0).fit(small)
    b = KCM(assume_scaled=True, random_state=999).fit(small)
    c = KCM(assume_scaled=True, random_state=None).fit(small)
    assert np.array_equal(a.reference_indices_, np.arange(len(small)))
    assert np.array_equal(a.reference_, b.reference_)
    assert np.array_equal(a.reference_, c.reference_)
    assert np.array_equal(a.anomaly_score(small), c.anomaly_score(small))


def test_random_state_none_draws_fresh_entropy(big):
    a = KCM(assume_scaled=True, random_state=None).fit(big)
    b = KCM(assume_scaled=True, random_state=None).fit(big)
    assert not np.array_equal(a.reference_indices_, b.reference_indices_)


def test_default_fit_above_the_cap_is_not_reproducible(big):
    """The one default that separates a user's number from a published one.

    check_estimator cannot see this: check_fit_idempotent builds 100 rows and
    check_outliers_train 300, both far below max_reference=2000, so the whole
    subsampling path is unexercised by any of the generated checks, at every
    parameter setting and in either novelty mode. The check total is not pinned here
    on purpose, because it moves with the scikit-learn release and with which half of
    the API is exposed. The documented remedy is an integer random_state, pinned by
    the second half here.
    """
    a = KCM(assume_scaled=True).fit(big)
    b = KCM(assume_scaled=True).fit(big)
    assert a.n_reference_ == CAP < len(big)
    assert not np.array_equal(a.reference_, b.reference_)
    assert a.bandwidth_ != b.bandwidth_
    assert not np.array_equal(a.predict(big), b.predict(big))

    c = KCM(assume_scaled=True, random_state=0).fit(big)
    d = KCM(assume_scaled=True, random_state=0).fit(big)
    assert c.bandwidth_ == d.bandwidth_
    assert np.array_equal(c.anomaly_score(big), d.anomaly_score(big))
    assert np.array_equal(c.predict(big), d.predict(big))


def test_rows_beyond_the_cap_are_discarded_not_summarised(big):
    """Replacing every discarded row with 1e6 changes nothing about the fitted model."""
    m = KCM(assume_scaled=True, random_state=0).fit(big)
    dropped = np.setdiff1d(np.arange(len(big)), m.reference_indices_)
    assert len(dropped) == len(big) - CAP

    mutated = big.copy()
    mutated[dropped] = 1e6
    other = KCM(assume_scaled=True, random_state=0).fit(mutated)
    assert np.array_equal(other.reference_, m.reference_)
    assert other.bandwidth_ == m.bandwidth_
    assert np.array_equal(other.loo_score_samples_, m.loo_score_samples_)
    assert other.offset_ == m.offset_


def test_random_state_reproduces_data_subsample(big):
    for seed in (0, 1, 7):
        m = KCM(assume_scaled=True, random_state=seed).fit(big)
        assert np.array_equal(m.reference_, subsample(big, seed, CAP))


def test_randomstate_instance_differs_from_the_equal_int(big):
    """A documented departure: the published subsample uses default_rng, not RandomState."""
    a = KCM(assume_scaled=True, random_state=0).fit(big)
    b = KCM(assume_scaled=True, random_state=np.random.RandomState(0)).fit(big)
    assert not np.array_equal(a.reference_indices_, b.reference_indices_)


def test_scoring_is_stateless(big, small):
    m = KCM(assume_scaled=True, random_state=0).fit(big)
    first = m.anomaly_score(small)
    m.anomaly_score(big)
    assert np.array_equal(m.anomaly_score(small), first)
    assert np.array_equal(m.anomaly_score(small), first)


def test_refit_overwrites_rather_than_accumulates(big, small):
    m = KCM(assume_scaled=True, random_state=0)
    m.fit(big)
    m.fit(small)
    fresh = KCM(assume_scaled=True, random_state=0).fit(small)
    assert m.n_reference_ == fresh.n_reference_
    assert m.bandwidth_ == fresh.bandwidth_
    assert np.array_equal(m.anomaly_score(small), fresh.anomaly_score(small))
