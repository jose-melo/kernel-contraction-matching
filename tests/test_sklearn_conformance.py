import collections
import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from karkcm import KCM, make_kcm_pipeline

OUTLIERS_TRAIN_REASON = (
    "KCM calibrates offset_ on leave-one-out anchor residuals (loo_score_samples_), "
    "not on in-sample scores. A training row sits inside its own reference set, so "
    "K(z, z) = 1 dominates its own Nadaraya-Watson reconstruction and its in-sample "
    "residual is near zero; an in-sample percentile is therefore an absurdly tight "
    "threshold and flags 0.108 to 1.000 of clean held-out rows at contamination=0.1 "
    "over all 47 ADBench datasets at seed 0, median 0.822 (measured 0.873 on "
    "6_cardio, 1.000 on 24_mnist, against 0.093 and 0.090 for the leave-one-out "
    "threshold, whose corpus range is 0.062 to 0.225). This check's unguarded "
    "assertion is np.unique(predict(X_train)) == [-1, 1]. On its own 300-row blob "
    "the in-sample scores span [-0.223, -0.0002] and the leave-one-out scores span "
    "[-1.115, -0.0048], so offset_ at the check's contamination=0.1 is -0.2649, "
    "below the whole in-sample range, and every row comes back an inlier; the "
    "crossing is between contamination=0.14 and 0.15. LocalOutlierFactor escapes "
    "the same assertion only because its contamination='auto' offset is the "
    "scale-free constant -1.5 rather than a percentile. KCM(novelty=False) passes "
    "check_estimator with nothing excused. See "
    "test_contamination_is_calibrated_on_leave_one_out_not_in_sample."
)

EXPECTED_FAILED_CHECKS = {"check_outliers_train": OUTLIERS_TRAIN_REASON}


CONFIGURATIONS = [
    (KCM(), True),
    (KCM(assume_scaled=True), True),
    (KCM(assume_scaled=True, max_reference=None), True),
    (KCM(assume_scaled=True, bandwidth=0.5), False),
    (KCM(assume_scaled=True, trim_frac=0.1, n_grid=5), False),
    (KCM(assume_scaled=True, chunk_size=7, contamination=0.25), False),
    (KCM(assume_scaled=True, novelty=False), False),
    (KCM(assume_scaled=True, novelty=False, contamination=0.25), False),
]

ESTIMATORS = [est for est, _ in CONFIGURATIONS]

_EXCUSED = {repr(est): excused for est, excused in CONFIGURATIONS}


def expected_failures(estimator):
    return EXPECTED_FAILED_CHECKS if _EXCUSED[repr(estimator)] else {}


ENVIRONMENT_SKIPS = {"check_array_api_input"}


def _tally(results):
    out = {}
    for entry in results:
        out[entry["status"]] = out.get(entry["status"], 0) + 1
    return out


def _failed(results):
    """Return the checks that genuinely failed, keeping environment skips separate.

    A skip is not a pass, so it must not be swallowed, but it is also not a defect.
    scikit-learn 1.9 added ``check_array_api_input``, which reports ``skipped``
    unless ``SCIPY_ARRAY_API`` is set in the environment. Counting every non-pass as
    a failure turned the whole conformance suite red on 1.9 while 1.8 stayed green.
    Any skip outside `ENVIRONMENT_SKIPS` is still surfaced by
    :func:`test_skips_are_only_the_known_environment_ones`.
    """
    return {e["check_name"] for e in results if e["status"] == "failed"}


def test_check_estimator():
    """Every check either passes or is the one documented expected failure.

    The counts are deliberately derived rather than pinned. scikit-learn adds and
    removes estimator checks between minor releases and this artifact is attached to
    a DOI, so a hard-coded total turns the suite red for no defect.

    Measured on scikit-learn 1.8.0: 46 checks, 44 passed, 2 xfail, 0 failed, 0
    skipped, and check_outliers_train contributes BOTH xfails, once memmap-backed.
    The total also moves with novelty, not only with the scikit-learn version:
    KCM(novelty=False) generates 44, because check_outliers_train (twice) and
    check_classifier_data_not_an_array are generated only for an estimator exposing
    predict, while check_outliers_fit_predict is generated only for one exposing
    fit_predict. See test_novelty_changes_which_checks_are_generated.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(
            KCM(assume_scaled=True), expected_failed_checks=EXPECTED_FAILED_CHECKS
        )
    tally = _tally(results)
    assert tally.get("failed", 0) == 0
    skipped = {e["check_name"] for e in results if e["status"] == "skipped"}
    assert skipped <= ENVIRONMENT_SKIPS, skipped
    assert tally["passed"] + tally["xfail"] + len(skipped) == len(results)
    assert {e["check_name"] for e in results if e["status"] == "xfail"} == set(
        EXPECTED_FAILED_CHECKS
    )


def test_expected_failures_are_exactly_the_one_documented_check():
    """The excuse list must be neither stale nor wider than the real failures.

    Without expected_failed_checks nothing is excused, so this run reports the
    checks that genuinely fail. Asserting the set is exactly the one documented
    name keeps a future regression in any OTHER check from being swallowed, and
    keeps an entry that has stopped failing from lingering as dead cover.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(KCM(), on_fail=None)
    failed = [e for e in results if e["status"] == "failed"]
    assert {e["check_name"] for e in failed} == set(EXPECTED_FAILED_CHECKS)
    for name in EXPECTED_FAILED_CHECKS:
        assert any(e["check_name"] == name for e in failed)
    for entry in failed:
        assert "Arrays are not equal" in str(entry["exception"])


@pytest.mark.filterwarnings("ignore::karkcm.NotStandardizedWarning")
@parametrize_with_checks(ESTIMATORS, expected_failed_checks=expected_failures)
def test_sklearn_check(estimator, check):
    check(estimator)


def test_no_transform_method():
    """A method named transform makes check_estimator abort before any check runs."""
    assert not hasattr(KCM, "transform")


def test_novelty_false_passes_check_estimator_with_nothing_excused():
    """The excuse buys nothing at novelty=False, where predict is not exposed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(KCM(novelty=False), on_fail=None)
    assert _failed(results) == set()


def test_novelty_hides_exactly_one_half_of_the_api():
    """The two halves are disjoint and neither mode can reach the inverted path."""
    scoring = ("score_samples", "decision_function", "predict")
    assert all(hasattr(KCM(novelty=True), name) for name in scoring)
    assert not hasattr(KCM(novelty=True), "fit_predict")
    assert not any(hasattr(KCM(novelty=False), name) for name in scoring)
    assert hasattr(KCM(novelty=False), "fit_predict")


@pytest.mark.parametrize("novelty", [True, False])
def test_hidden_half_raises_attribute_error_after_fit(novelty):
    """Hiding survives fitting: available_if is consulted on every access."""
    X = np.random.default_rng(0).normal(size=(60, 4))
    hidden = "fit_predict" if novelty else "predict"
    fitted = KCM(assume_scaled=True, novelty=novelty).fit(X)
    with pytest.raises(AttributeError, match=hidden):
        getattr(fitted, hidden)


@pytest.mark.parametrize("estimator,excused", CONFIGURATIONS, ids=repr)
def test_the_excuse_is_pinned_per_configuration(estimator, excused):
    """No configuration carries an excuse it does not need, or lacks one it does.

    check_outliers_train fails on the bandwidth the leave-one-out grid picks for its
    300-row blob and passes at bandwidth=0.5, so the deviation is a property of the
    fitted bandwidth and not of KCM as such. A blanket excuse would turn three of
    these eight configurations into stale cover.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(estimator, on_fail=None)
    assert _failed(results) == ({"check_outliers_train"} if excused else set())


def test_novelty_changes_which_checks_are_generated():
    """The check set is a function of the exposed API, not only of the sklearn version.

    This is the pin the stale "47 checks, 3 xfail" prose lacked. scikit-learn
    generates checks from the methods an estimator exposes, so hiding half the
    labelling API removes three checks and adds one. Asserting the SET difference
    rather than the totals keeps this green when a release adds an unrelated check,
    while still failing if the novelty split stops changing what is generated.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scoring = collections.Counter(
            e["check_name"] for e in check_estimator(KCM(assume_scaled=True), on_fail=None)
        )
        transductive = collections.Counter(
            e["check_name"]
            for e in check_estimator(KCM(assume_scaled=True, novelty=False), on_fail=None)
        )
    assert dict(scoring - transductive) == {
        "check_outliers_train": 2,
        "check_classifier_data_not_an_array": 1,
    }
    assert dict(transductive - scoring) == {"check_outliers_fit_predict": 1}
    assert sum(scoring.values()) - 3 + 1 == sum(transductive.values())


@pytest.mark.parametrize("novelty", [True, False])
def test_pipeline_forwards_the_novelty_gate(novelty):
    """make_kcm_pipeline exposes the same half as its final step, through every round trip.

    Nothing else pins this. KCMPipeline gates on _final_step_has, and sklearn's own
    Pipeline gates predict/score_samples/decision_function on the final step too, so a
    regression in either would silently give a pipeline both halves or neither.
    """
    scoring = ("score_samples", "decision_function", "predict")
    X = np.vstack([
        np.random.default_rng(0).normal(size=(200, 3)),
        np.random.default_rng(1).normal(size=(20, 3)) + 8.0,
    ])
    built = make_kcm_pipeline(novelty=novelty)
    for pipe in (built, clone(built), built.fit(X), pickle.loads(pickle.dumps(built.fit(X)))):
        assert all(hasattr(pipe, name) is novelty for name in scoring)
        assert hasattr(pipe, "fit_predict") is not novelty
        assert all(
            hasattr(pipe, name) for name in ("anomaly_score", "reconstruct", "kernel_mass")
        )
    if not novelty:
        labels = make_kcm_pipeline(novelty=False).fit_predict(X)
        assert set(np.unique(labels)) == {-1, 1}
        assert int((labels == -1).sum()) == 22 == int(0.1 * len(X))


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=repr)
def test_skips_are_only_the_known_environment_ones(estimator):
    """A check that silently stops running is as invisible as one that is excused."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(estimator, on_fail=None)
    skipped = {e["check_name"] for e in results if e["status"] == "skipped"}
    assert skipped <= ENVIRONMENT_SKIPS, skipped
