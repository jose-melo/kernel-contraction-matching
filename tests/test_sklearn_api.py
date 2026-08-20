import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from karkcm import (
    KCM,
    KCMPipeline,
    NotStandardizedWarning,
    anomaly_auprc,
    anomaly_auroc,
    make_kcm_pipeline,
)

NON_DEFAULT_PARAMS = {
    "bandwidth": 0.75,
    "n_grid": 9,
    "trim_frac": 0.2,
    "max_reference": 64,
    "contamination": 0.25,
    "chunk_size": 17,
    "assume_scaled": True,
    "novelty": False,
    "random_state": 7,
}


def _normals(n=400, d=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def _labelled(n_normal=400, n_anomaly=40, d=4, spread=1.8, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n_normal, d))
    anomaly = rng.normal(size=(n_anomaly, d)) * spread
    X = np.vstack([normal, anomaly])
    y = np.r_[np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_clone_round_trips_every_parameter():
    assert set(KCM().get_params()) == set(NON_DEFAULT_PARAMS)
    original = KCM(**NON_DEFAULT_PARAMS)
    copied = clone(original)
    assert copied is not original
    assert copied.get_params() == NON_DEFAULT_PARAMS
    assert original.get_params() == NON_DEFAULT_PARAMS
    for name, value in NON_DEFAULT_PARAMS.items():
        assert getattr(copied, name) == value


def test_clone_of_a_fitted_estimator_is_unfitted():
    X = _normals()
    fitted = KCM(assume_scaled=True).fit(X)
    fresh = clone(fitted)
    assert not hasattr(fresh, "bandwidth_")
    assert fresh.get_params() == fitted.get_params()


def test_get_params_set_params_round_trip():
    m = KCM()
    defaults = m.get_params()
    assert m.set_params(**NON_DEFAULT_PARAMS) is m
    assert m.get_params() == NON_DEFAULT_PARAMS
    m.set_params(**defaults)
    assert m.get_params() == defaults
    assert m.get_params(deep=True) == m.get_params(deep=False)


def test_pickle_round_trip_scores_identically():
    X = _normals()
    query = _normals(n=150, seed=1)
    m = KCM(assume_scaled=True, chunk_size=13, random_state=0).fit(X)
    restored = pickle.loads(pickle.dumps(m))

    assert restored.get_params() == m.get_params()
    assert restored.bandwidth_ == m.bandwidth_
    assert restored.offset_ == m.offset_
    assert restored.n_reference_ == m.n_reference_
    assert np.array_equal(restored.reference_, m.reference_)
    assert np.array_equal(restored.loo_score_samples_, m.loo_score_samples_)
    assert np.array_equal(restored.bandwidth_grid_, m.bandwidth_grid_)
    assert np.array_equal(restored.loo_curve_, m.loo_curve_)

    assert np.array_equal(restored.anomaly_score(query), m.anomaly_score(query))
    assert np.array_equal(restored.score_samples(query), m.score_samples(query))
    assert np.array_equal(restored.decision_function(query), m.decision_function(query))
    assert np.array_equal(restored.predict(query), m.predict(query))
    assert np.array_equal(restored.reconstruct(query), m.reconstruct(query))
    assert np.array_equal(restored.kernel_mass(query), m.kernel_mass(query))


def test_pipeline_fits_and_predicts():
    X = _normals() * np.array([1.0, 1000.0, 1.0, 0.001, 1.0, 1.0])
    query = _normals(n=150, seed=1) * np.array([1.0, 1000.0, 1.0, 0.001, 1.0, 1.0])
    pipe = Pipeline([("scale", StandardScaler()), ("kcm", KCM())])
    assert pipe.fit(X) is pipe
    labels = pipe.predict(query)
    assert labels.shape == (len(query),)
    assert set(np.unique(labels)) <= {-1, 1}
    assert np.array_equal(labels, np.where(pipe.decision_function(query) < 0, -1, 1))
    assert np.all(pipe.score_samples(query) <= 0)
    assert pipe.named_steps["kcm"].n_features_in_ == X.shape[1]


def test_pipeline_scaling_silences_the_warning():
    X = _normals() * np.array([1.0, 1000.0, 1.0, 0.001, 1.0, 1.0])
    pipe = Pipeline([("scale", StandardScaler()), ("kcm", KCM())])
    with pytest.warns(NotStandardizedWarning):
        KCM().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NotStandardizedWarning)
        pipe.fit(X)


def test_grid_search_cv_over_two_parameters():
    """The working GridSearchCV pattern for an outlier detector in scikit-learn 1.8.

    Four things have to be right at once, and three of them fail SILENTLY:

    1. `scoring` must be a callable of (estimator, X, y). The string "roc_auc"
       scores `decision_function` in the normal-is-high direction, so with
       y == 1 meaning anomaly it returns exactly 1 - AUROC and raises nothing.
       See test_roc_auc_string_scorer_returns_the_complement.
    2. `y` must be passed to `fit`. Without it the scorer is called with no
       labels, TypeErrors, and scikit-learn turns that into a nan fold score plus
       a warning. See test_cross_val_score_without_y_is_all_nan. KCM.fit ignores
       y, so the labels are used only by the scorer; each training fold is
       contaminated with that fold's anomalies, which is what unsupervised model
       selection over labelled data means here.
    3. `cv` must be a StratifiedKFold passed explicitly. is_classifier(KCM()) is
       False, so sklearn's default check_cv hands back a plain KFold, and in
       scikit-learn 1.8 a single-class test fold makes roc_auc_score return nan
       with a warning rather than raise. See
       test_default_kfold_silently_produces_nan_fold_scores.
    4. Score the refit best_estimator_ only on data it was not fitted on. A row
       inside the reference set reconstructs itself, so in-sample AUROC is
       inverted, not merely optimistic. See
       test_scoring_the_refit_estimator_in_sample_is_inverted.
    5. Do not grid-search chunk_size. It is in get_params() and it cannot change
       the answer; see test_chunk_size_is_searchable_but_inert.
    """
    X, y = _labelled()
    X_fit, X_held, y_fit, y_held = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    grid = {"bandwidth": ["loo", 0.5, 1.0], "trim_frac": [0.0, 0.1]}
    search = GridSearchCV(
        KCM(assume_scaled=True),
        grid,
        scoring=anomaly_auroc,
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        refit=True,
        error_score="raise",
    )
    search.fit(X_fit, y_fit)

    assert len(search.cv_results_["params"]) == 6
    scores = search.cv_results_["mean_test_score"]
    assert np.all(np.isfinite(scores))
    assert np.all((scores > 0.5) & (scores <= 1.0))
    assert scores.max() - scores.min() > 0.01
    assert set(search.best_params_) == {"bandwidth", "trim_frac"}
    assert search.best_score_ == pytest.approx(scores.max())
    assert search.best_estimator_.bandwidth_ > 0.0
    assert search.best_estimator_.n_features_in_ == X.shape[1]

    assert anomaly_auroc(search.best_estimator_, X_held, y_held) > 0.6
    assert 0.0 < anomaly_auprc(search.best_estimator_, X_held, y_held) < 1.0
    assert search.score(X_held, y_held) == pytest.approx(
        anomaly_auroc(search.best_estimator_, X_held, y_held)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scoring_the_refit_estimator_in_sample_is_inverted(seed):
    """Self-reconstruction is not optimism, it is a sign flip.

    Every training row is in the reference set, so K(z, z) = 1 dominates its own
    Nadaraya-Watson reconstruction. The more isolated the row, the more
    completely it reconstructs itself, so the anomalies get the SMALLEST
    residuals and in-sample AUROC lands below 0.5 rather than above it. Measured
    here at 0.012, 0.001 and 0.102 for seeds 0, 1 and 2, against 1.000 for the
    same model fitted on the normals alone.
    """
    X, y = _labelled(d=6, spread=3.0, seed=seed)
    contaminated = KCM(assume_scaled=True).fit(X)
    clean = KCM(assume_scaled=True).fit(X[y == 0])
    assert anomaly_auroc(contaminated, X, y) < 0.25
    assert anomaly_auroc(clean, X, y) > 0.95


@pytest.mark.filterwarnings("ignore:One or more of the test scores are non-finite")
def test_default_kfold_silently_produces_nan_fold_scores():
    """Why cv must be StratifiedKFold and not left to the default.

    On label-sorted data a plain KFold hands roc_auc_score single-class folds.
    scikit-learn 1.8 answers those with nan and a warning instead of an
    exception, so GridSearchCV completes, every mean_test_score is nan, and
    best_params_ is whatever np.nanargmax falls back to.
    """
    X, y = _labelled()
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    grid = {"bandwidth": ["loo", 0.5]}
    kwargs = dict(scoring=anomaly_auroc, error_score="raise")

    unstratified = GridSearchCV(KCM(assume_scaled=True), grid, cv=KFold(3), **kwargs)
    with pytest.warns(UserWarning, match="Only one class is present"):
        unstratified.fit(X, y)
    assert np.all(np.isnan(unstratified.cv_results_["mean_test_score"]))

    stratified = GridSearchCV(
        KCM(assume_scaled=True), grid, cv=StratifiedKFold(3), **kwargs
    )
    stratified.fit(X, y)
    assert np.all(np.isfinite(stratified.cv_results_["mean_test_score"]))


def test_roc_auc_string_scorer_returns_the_complement():
    """scoring="roc_auc" is the mirror image, with no error raised."""
    X, y = _labelled()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    grid = {"bandwidth": ["loo"]}
    mirrored = GridSearchCV(
        KCM(assume_scaled=True), grid, scoring="roc_auc", cv=cv, error_score="raise"
    ).fit(X, y)
    correct = GridSearchCV(
        KCM(assume_scaled=True), grid, scoring=anomaly_auroc, cv=cv, error_score="raise"
    ).fit(X, y)
    assert mirrored.best_score_ == pytest.approx(1.0 - correct.best_score_)
    assert correct.best_score_ > 0.5 > mirrored.best_score_


def test_cross_val_score_without_y_is_all_nan():
    """Dropping y does not raise: the scorer TypeErrors and sklearn writes nan.

    ``cross_val_score(KCM(), X, scoring=anomaly_auroc)`` looks like the natural
    unsupervised call. It calls the scorer as ``scorer(estimator, X_test)`` with
    no labels, the TypeError is caught by ``_score``, and the fold is recorded as
    nan behind a UserWarning. Pass y even though ``KCM.fit`` ignores it.
    """
    X, y = _labelled()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    supervised = cross_val_score(
        KCM(assume_scaled=True), X, y, scoring=anomaly_auroc, cv=cv
    )
    assert supervised.shape == (3,)
    assert np.all(np.isfinite(supervised))
    assert np.all(supervised > 0.5)

    with pytest.warns(UserWarning, match="Scoring failed"):
        unsupervised = cross_val_score(
            KCM(assume_scaled=True), X, scoring=anomaly_auroc, cv=KFold(3)
        )
    assert np.all(np.isnan(unsupervised))


def test_chunk_size_is_searchable_but_inert():
    """chunk_size is a memory knob GridSearchCV will happily and pointlessly search."""
    X, y = _labelled()
    search = GridSearchCV(
        KCM(assume_scaled=True),
        {"chunk_size": [7, 64, 4096]},
        scoring=anomaly_auroc,
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        error_score="raise",
    ).fit(X, y)
    scores = search.cv_results_["mean_test_score"]
    assert np.allclose(scores, scores[0], atol=1e-12)


def test_fit_predict_is_transductive_and_differs_from_fit_then_predict():
    """fit_predict labels anchors by leave-one-out score, so it is not fit().predict().

    This is the LocalOutlierFactor design, and the novelty parameter is what keeps
    the two apart: no single instance exposes both. fit(X).predict(X) flags nothing
    at all, because every training row reconstructs itself; fit_predict flags
    contamination * n.
    """
    X = _normals()
    for params in ({}, {"contamination": 0.25}, {"chunk_size": 11}, {"bandwidth": 0.5}):
        template = KCM(assume_scaled=True, random_state=0, **params)
        labels = clone(template).set_params(novelty=False).fit_predict(X)
        in_sample = clone(template).fit(X).predict(X)
        assert labels.dtype.kind == "i"
        assert set(np.unique(labels)) == {-1, 1}
        assert np.array_equal(np.unique(in_sample), np.array([1]))
        assert not np.array_equal(labels, in_sample)
        expected = params.get("contamination", 0.1) * len(X)
        assert abs(int((labels == -1).sum()) - expected) <= 1.0


def test_fit_predict_on_a_capped_fit_labels_every_row():
    """Anchors get their leave-one-out label; rows the cap left out are out of sample."""
    X = _normals(n=900, seed=3)
    template = KCM(assume_scaled=True, max_reference=300, random_state=0)
    labels = clone(template).set_params(novelty=False).fit_predict(X)
    fitted = clone(template).fit(X)
    assert fitted.n_reference_ == 300
    assert labels.shape == (900,)
    anchors = np.sort(fitted.reference_indices_)
    others = np.setdiff1d(np.arange(900), anchors)
    assert np.array_equal(labels[others], fitted.predict(X[others]))
    assert not np.array_equal(labels[anchors], fitted.predict(X[anchors]))


@pytest.mark.parametrize("seed,floor", [(11, 0.80), (3, 0.80), (7, 0.95)])
def test_fit_predict_recovers_the_ranking_that_predict_inverts(seed, floor):
    """The blocker this method exists for: in sample the raw score is anti-correlated.

    400 standard normals plus 8 points at radius 20. anomaly_score ranks those 8
    among the most NORMAL rows in the set, because each one is alone inside its own
    reference set and reconstructs itself exactly, so in-sample AUROC lands at 0.06
    to 0.13. The leave-one-out scores fit_predict uses are not self-reconstructed and
    put it back above 0.84.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, 8)
    X = np.vstack([rng.normal(size=(400, 2)), 20.0 * np.c_[np.cos(angle), np.sin(angle)]])
    y = np.r_[np.zeros(400, dtype=int), np.ones(8, dtype=int)]

    m = KCM(assume_scaled=True, contamination=0.02).fit(X)
    assert anomaly_auroc(m, X, y) < 0.25
    assert int((m.predict(X)[400:] == -1).sum()) == 0

    labels = KCM(assume_scaled=True, contamination=0.02, novelty=False).fit_predict(X)
    assert int((labels[400:] == -1).sum()) >= 5
    loo = np.empty(len(X))
    loo[:] = m.score_samples(X)
    loo[m.reference_indices_] = m.loo_score_samples_
    assert roc_auc_score(y, -loo) > floor


def test_make_kcm_pipeline_forwards_the_three_extra_methods():
    """A plain Pipeline forwards none of them, which pushes callers back to -score_samples."""
    X = _normals() * np.array([1.0, 1000.0, 1.0, 0.001, 1.0, 1.0])
    query = _normals(n=40, seed=1) * np.array([1.0, 1000.0, 1.0, 0.001, 1.0, 1.0])
    pipe = make_kcm_pipeline(random_state=0).fit(X)
    assert isinstance(pipe, KCMPipeline)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["standardscaler", "kcm"]

    kcm = pipe.named_steps["kcm"]
    scaled = pipe.named_steps["standardscaler"].transform(query)
    assert np.array_equal(pipe.anomaly_score(query), kcm.anomaly_score(scaled))
    assert np.array_equal(pipe.reconstruct(query), kcm.reconstruct(scaled))
    assert np.array_equal(pipe.kernel_mass(query), kcm.kernel_mass(scaled))
    assert np.array_equal(pipe.anomaly_score(query), -pipe.score_samples(query))

    plain = Pipeline([("standardscaler", StandardScaler()), ("kcm", KCM())]).fit(X)
    for name in ("anomaly_score", "reconstruct", "kernel_mass"):
        assert not hasattr(plain, name)


def test_kcm_pipeline_clones_pickles_and_grid_searches():
    X, y = _labelled()
    pipe = make_kcm_pipeline(random_state=0)
    assert type(clone(pipe)) is KCMPipeline
    fitted = pipe.fit(X)
    restored = pickle.loads(pickle.dumps(fitted))
    assert np.array_equal(restored.anomaly_score(X), fitted.anomaly_score(X))
    search = GridSearchCV(
        make_kcm_pipeline(),
        {"kcm__bandwidth": ["loo", 0.5]},
        scoring=anomaly_auroc,
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        error_score="raise",
    ).fit(X, y)
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert hasattr(search.best_estimator_, "anomaly_score")


def test_uncentered_input_with_assume_scaled_inverts_the_detector():
    """Why assume_scaled=True is a silencer, not a remedy, and the pipeline is.

    Fitted on a blob at (50, 50), eight queries at the coordinate origin are 50 sigma
    out. The Gaussian underflows, NW(z) collapses to the origin, and the score becomes
    ||z|| = 0 exactly, so they are the most NORMAL rows the model has ever seen.
    Centering is what makes that fallback conservative.
    """
    rng = np.random.default_rng(0)
    centre = np.array([50.0, 50.0])
    train = rng.normal(size=(300, 2)) + centre
    query = np.vstack([rng.normal(size=(200, 2)) + centre, np.zeros((8, 2))])
    y = np.r_[np.zeros(200, dtype=int), np.ones(8, dtype=int)]

    silenced = KCM(assume_scaled=True, random_state=0).fit(train)
    assert np.all(silenced.anomaly_score(query)[200:] == 0.0)
    assert np.all(silenced.kernel_mass(query)[200:] == 0.0)
    assert np.all(silenced.predict(query)[200:] == 1)
    assert roc_auc_score(y, silenced.anomaly_score(query)) == 0.0

    scaled = make_kcm_pipeline(random_state=0).fit(train)
    assert roc_auc_score(y, scaled.anomaly_score(query)) == 1.0
    assert np.all(scaled.predict(query)[200:] == -1)


@pytest.mark.parametrize("n", [37, 50, 101, 200, 373, 500])
@pytest.mark.parametrize("contamination", [0.01, 0.05, 0.1, 0.2, 0.33, 0.5])
def test_contamination_flags_that_fraction_of_the_calibration_set(n, contamination):
    """offset_ flags contamination * n of the set it is calibrated on, to one sample.

    That set is loo_score_samples_, the leave-one-out residuals of the anchors,
    which is the KCM analogue of LocalOutlierFactor.negative_outlier_factor_. It
    is NOT the in-sample score of the training rows; see
    test_contamination_is_calibrated_on_leave_one_out_not_in_sample.
    """
    X = _normals(n=n, d=4, seed=n)
    m = KCM(assume_scaled=True, contamination=contamination).fit(X)
    flagged = int((m.loo_score_samples_ < m.offset_).sum())
    assert m.n_reference_ == n
    assert abs(flagged - contamination * n) <= 1.0


def test_contamination_is_calibrated_on_leave_one_out_not_in_sample():
    """The documented departure from IsolationForest, and the reason it is right.

    A training row is inside its own reference set, so K(z, z) = 1 dominates its
    own reconstruction and its in-sample residual is near zero. predict on the
    TRAINING set therefore flags nothing, which is what makes check_outliers_train
    an expected failure on the bandwidths where it bites. The
    threshold is nonetheless correct, because it is calibrated out of sample: a
    fresh draw from the training distribution is flagged at the requested rate,
    and reverting to an in-sample percentile would flag essentially all of it.
    """
    X = _normals(n=400, seed=11)
    held = _normals(n=2000, seed=12)
    m = KCM(assume_scaled=True, contamination=0.1).fit(X)

    assert int((m.predict(X) == -1).sum()) == 0
    assert np.array_equal(np.unique(m.predict(X)), np.array([1]))

    in_sample_offset = float(np.percentile(m.score_samples(X), 10.0))
    in_sample_rate = float((m.score_samples(held) < in_sample_offset).mean())
    loo_rate = float((m.predict(held) == -1).mean())
    assert in_sample_rate > 0.9
    assert abs(loo_rate - 0.1) < 0.05
    assert loo_rate < in_sample_rate


@pytest.mark.parametrize("contamination", [0.05, 0.1, 0.2, 0.3])
def test_contamination_matches_the_flag_rate_on_held_out_draws(contamination):
    rates = []
    for seed in range(100, 105):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(400, 6))
        held = rng.normal(size=(2000, 6))
        m = KCM(assume_scaled=True, contamination=contamination).fit(X)
        rates.append(float((m.predict(held) == -1).mean()))
    assert max(abs(r - contamination) for r in rates) < 0.05
    assert abs(float(np.mean(rates)) - contamination) < 0.02
