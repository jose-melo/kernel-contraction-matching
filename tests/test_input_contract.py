import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from karkcm import KCM, KCMAnchor, NotStandardizedWarning, make_kcm_pipeline

BAD_PARAMETERS = [
    {"contamination": 0.7},
    {"contamination": 0.0},
    {"n_grid": 1},
    {"trim_frac": 1.0},
    {"max_reference": 0},
    {"chunk_size": 0},
    {"bandwidth": "silverman"},
    {"bandwidth": -1.0},
]


def _blob(n=200, d=5, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_dataframe_input():
    X = _blob()
    names = [f"f{i}" for i in range(X.shape[1])]
    m = KCM(assume_scaled=True).fit(pd.DataFrame(X, columns=names))
    assert list(m.feature_names_in_) == names
    assert m.anomaly_score(pd.DataFrame(X, columns=names)).shape == (len(X),)
    with pytest.raises(ValueError):
        m.anomaly_score(X[:, :2])


def test_frozen_path_dataframe_limitation():
    """The frozen path indexes by position, so KCM is the DataFrame-safe entry point."""
    X = _blob(n=600)
    with pytest.raises(KeyError):
        KCMAnchor.fit(pd.DataFrame(X))


def test_not_fitted():
    X = _blob()
    for name in (
        "anomaly_score",
        "predict",
        "score_samples",
        "decision_function",
        "reconstruct",
        "kernel_mass",
    ):
        with pytest.raises(NotFittedError):
            getattr(KCM(), name)(X)


@pytest.mark.parametrize("kwargs", BAD_PARAMETERS, ids=[str(k) for k in BAD_PARAMETERS])
def test_bad_parameters(kwargs):
    with pytest.raises(InvalidParameterError):
        KCM(**kwargs).fit(_blob())


def test_bad_parameter_message():
    with pytest.raises(InvalidParameterError) as excinfo:
        KCM(contamination=0.7).fit(_blob())
    assert str(excinfo.value) == (
        "The 'contamination' parameter of KCM must be a float in the range "
        "(0.0, 0.5]. Got 0.7 instead."
    )


def test_one_sample():
    with pytest.raises(ValueError, match="1 sample"):
        KCM().fit(_blob()[:1])


def test_nan_inf():
    X = _blob()
    X[3, 1] = np.nan
    with pytest.raises(ValueError):
        KCM().fit(X)
    X[3, 1] = np.inf
    with pytest.raises(ValueError):
        KCM().fit(X)


def test_standardization_warning():
    X = _blob() * np.array([1.0, 1000.0, 1.0, 1.0, 1.0])
    with pytest.warns(NotStandardizedWarning):
        KCM().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KCM(assume_scaled=True).fit(X)
        make_kcm_pipeline().fit(X)


def test_standardization_warning_points_at_the_caller():
    """stacklevel must clear both fit and the private helper, or the advice lands in the library.

    The whole content of the message is what the CALLER should do differently, so a
    warning attributed to karkcm/estimator.py cannot be traced to the fit that tripped
    it and a filterwarnings rule scoped to the caller's module will not match it.
    """
    X = _blob() * np.array([1.0, 1000.0, 1.0, 1.0, 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KCM().fit(X)
    assert len(caught) == 1
    assert Path(caught[0].filename).name == Path(__file__).name


def test_standardization_warning_names_the_pipeline_as_the_remedy():
    X = _blob() * np.array([1.0, 1000.0, 1.0, 1.0, 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KCM().fit(X)
    message = str(caught[0].message)
    assert "make_kcm_pipeline()" in message
    assert "centered as well as scaled" in message


def test_fixed_bandwidth():
    m = KCM(bandwidth=0.5, assume_scaled=True).fit(_blob())
    assert m.bandwidth_ == 0.5
    assert m.bandwidth_grid_.shape == (0,)
    assert m.loo_curve_.shape == (0,)
    assert m.bandwidth_at_grid_edge_ is False
