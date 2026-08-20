import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from karkcm import KCM
from karkcm.data import ADBENCH_47, find_dataset
from karkcm.metrics import auroc

from .conftest import have_dataset

MAX_TRAIN = 20000

PUBLISHED_GAINS = {
    "2_annthyroid": 0.211,
    "35_SpamBase": 0.181,
    "15_Hepatitis": 0.177,
    "27_PageBlocks": 0.139,
    "5_campaign": 0.088,
    "29_Pima": 0.026,
}

PUBLISHED_LOSSES = {
    "44_Wilt": -0.196,
    "37_Stamps": -0.052,
    "17_InternetAds": -0.043,
    "26_optdigits": -0.031,
    "41_Waveform": -0.025,
    "7_Cardiotocography": -0.020,
}


def _raw_split(name, seed, test_frac=0.5):
    blob = np.load(find_dataset(name), allow_pickle=True)
    X = blob["X"].astype(np.float64)
    y = blob["y"].astype(int)
    X_norm, X_anom = X[y == 0], X[y == 1]
    y_norm = np.zeros(len(X_norm), dtype=int)
    X_tr, X_held, _, y_held = train_test_split(
        X_norm, y_norm, test_size=test_frac, random_state=seed, stratify=y_norm
    )
    X_test = np.vstack([X_held, X_anom])
    y_test = np.concatenate([y_held, np.ones(len(X_anom), dtype=int)])
    return X_tr, X_test, y_test


def _auroc(X_tr, X_te, y_te, seed):
    m = KCM(assume_scaled=True, random_state=seed).fit(X_tr)
    return auroc(y_te, m.anomaly_score(X_te))


def _deltas(seed=0):
    out = {}
    for name in ADBENCH_47:
        if not have_dataset(name):
            continue
        X_tr, X_te, y_te = _raw_split(name, seed)
        if len(X_tr) > MAX_TRAIN:
            continue
        scaler = StandardScaler().fit(X_tr)
        scaled = _auroc(scaler.transform(X_tr), scaler.transform(X_te), y_te, seed)
        raw = _auroc(X_tr, X_te, y_te, seed)
        out[name] = scaled - raw
    return out


@pytest.mark.slow
def test_readme_scaling_table_is_current():
    """Pins the Scaling section's table, the one number that had drifted from the corpus.

    The README used to claim the median was 0.000, the sign split 17 to 17 and the worst
    loss 0.038, when the real worst loss is 44_Wilt at -0.196 and raw features win on 20
    of the 36 datasets. Every figure printed in that section is regenerated here.
    """
    deltas = _deltas(seed=0)
    if len(deltas) < len(PUBLISHED_GAINS) + len(PUBLISHED_LOSSES):
        pytest.skip("run: python scripts/download_datasets.py")

    for name, published in {**PUBLISHED_GAINS, **PUBLISHED_LOSSES}.items():
        assert name in deltas, name
        assert deltas[name] == pytest.approx(published, abs=0.002), name

    if len(deltas) < 36:
        return
    values = np.array(list(deltas.values()))
    assert len(deltas) == 36
    assert float(np.median(values)) == pytest.approx(-0.0011, abs=0.0005)
    assert int((values > 1e-9).sum()) == 14
    assert int((values < -1e-9).sum()) == 20
    assert min(deltas, key=deltas.get) == "44_Wilt"
    assert max(deltas, key=deltas.get) == "2_annthyroid"


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 2])
def test_wilt_loses_under_standardization_at_more_than_one_seed(seed):
    """The dataset the README's old figure omitted, and it is not a seed artefact."""
    if not have_dataset("44_Wilt"):
        pytest.skip("run: python scripts/download_datasets.py --datasets 44_Wilt")
    X_tr, X_te, y_te = _raw_split("44_Wilt", seed)
    scaler = StandardScaler().fit(X_tr)
    scaled = _auroc(scaler.transform(X_tr), scaler.transform(X_te), y_te, seed)
    raw = _auroc(X_tr, X_te, y_te, seed)
    assert scaled - raw < -0.10
