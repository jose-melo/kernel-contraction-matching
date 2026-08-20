import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from karkcm import KCM, anomaly_auprc, anomaly_auroc

from .conftest import PAPER_DATASETS, estimator, requires_datasets, split

RAY = (0.0, 1.0, 3.0, 10.0, 100.0, 1.0e4)


def _hull(n=300, d=4, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return X, KCM(assume_scaled=True, random_state=0).fit(X)


def _ray(X):
    centre = X.mean(axis=0, keepdims=True)
    return np.vstack([centre + t for t in RAY])


def test_far_point_outside_the_hull_is_an_outlier():
    """Synthetic and two-sided: a constant or sign-flipped predictor fails one half."""
    X, m = _hull()
    centre = X.mean(axis=0, keepdims=True)
    far = centre + RAY[-1]

    assert m.anomaly_score(far)[0] > 100.0 * m.anomaly_score(centre)[0]
    assert m.score_samples(far)[0] < m.score_samples(centre)[0]
    assert m.decision_function(far)[0] < m.decision_function(centre)[0]
    assert m.decision_function(far)[0] < 0.0 <= m.decision_function(centre)[0]
    assert m.predict(far)[0] == -1
    assert m.predict(centre)[0] == 1


def test_scores_are_monotone_along_a_ray_leaving_the_hull():
    X, m = _hull()
    ray = _ray(X)
    assert np.all(np.diff(m.anomaly_score(ray)) > 0)
    assert np.all(np.diff(m.score_samples(ray)) < 0)
    assert np.all(np.diff(m.decision_function(ray)) < 0)
    labels = m.predict(ray)
    assert labels[0] == 1
    assert labels[-1] == -1
    assert np.all(np.diff(labels) <= 0)


def test_score_samples_is_the_negation_not_the_score():
    """The sign trap: score_samples is <= 0 while anomaly_score is >= 0 on the same rows."""
    X, m = _hull()
    ray = _ray(X)
    assert np.array_equal(m.score_samples(ray), -m.anomaly_score(ray))
    assert not np.array_equal(m.score_samples(ray), m.anomaly_score(ray))
    assert np.all(m.score_samples(ray) <= 0.0)
    assert np.all(m.anomaly_score(ray) >= 0.0)
    assert np.max(m.anomaly_score(ray)) > 0.0


def test_far_point_underflows_to_the_distance_from_the_origin():
    X, m = _hull()
    far = X.mean(axis=0, keepdims=True) + RAY[-1]
    assert m.kernel_mass(far)[0] == 0.0
    assert np.allclose(
        m.anomaly_score(far), np.linalg.norm(far, axis=1), rtol=0.0, atol=1e-9
    )


def test_labelled_synthetic_auroc_is_not_inverted():
    X, m = _hull()
    rng = np.random.default_rng(1)
    normals = rng.normal(size=(200, X.shape[1]))
    outliers = rng.normal(size=(50, X.shape[1])) + 8.0
    Q = np.vstack([normals, outliers])
    y = np.r_[np.zeros(len(normals)), np.ones(len(outliers))]
    assert roc_auc_score(y, m.anomaly_score(Q)) == 1.0
    assert roc_auc_score(y, m.decision_function(Q)) == 0.0
    assert anomaly_auroc(m, Q, y) == 1.0
    assert anomaly_auprc(m, Q, y) == 1.0


@requires_datasets("29_Pima")
def test_identities():
    X_te = split("29_Pima", 0)[1]
    m = estimator("29_Pima", 0)
    assert np.array_equal(m.score_samples(X_te), -m.anomaly_score(X_te))
    assert np.array_equal(
        m.decision_function(X_te), m.score_samples(X_te) - m.offset_
    )
    assert np.all(m.score_samples(X_te) <= 0)
    assert m.offset_ <= 0
    assert np.array_equal(
        m.predict(X_te), np.where(m.decision_function(X_te) < 0, -1, 1)
    )
    assert m.predict(X_te).dtype.kind == "i"


@requires_datasets("29_Pima")
def test_mirror_image_regression():
    """0.2957 is what a user typing scoring="roc_auc" silently gets instead of 0.7043."""
    X_te, y_te = split("29_Pima", 0)[1:]
    m = estimator("29_Pima", 0)
    assert roc_auc_score(y_te, m.anomaly_score(X_te)) == pytest.approx(
        0.7043, abs=1e-4
    )
    assert roc_auc_score(y_te, m.decision_function(X_te)) == pytest.approx(
        0.2957, abs=1e-4
    )


@requires_datasets("29_Pima")
def test_scorers():
    X_te, y_te = split("29_Pima", 0)[1:]
    m = estimator("29_Pima", 0)
    assert anomaly_auroc(m, X_te, y_te) == pytest.approx(0.7043, abs=1e-4)
    ap = anomaly_auprc(m, X_te, y_te)
    assert np.isfinite(ap)
    assert 0.0 < ap < 1.0


@requires_datasets(*PAPER_DATASETS)
@pytest.mark.parametrize("dataset", PAPER_DATASETS)
def test_contamination_calibration(dataset):
    X_te, y_te = split(dataset, 0)[1:]
    normals = X_te[y_te == 0]
    flagged = float(np.mean(estimator(dataset, 0).predict(normals) == -1))
    assert 0.04 <= flagged <= 0.16


@requires_datasets("24_mnist")
def test_kernel_mass_degeneracy():
    X_te = split("24_mnist", 0)[1]
    m = estimator("24_mnist", 0)
    dead = m.kernel_mass(X_te) == 0.0
    assert dead.sum() >= 1
    assert np.allclose(
        m.anomaly_score(X_te)[dead],
        np.linalg.norm(X_te[dead], axis=1),
        atol=1e-9,
    )
