"""ADBench loading and the semi-supervised split used by every experiment."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .paths import DATA_ROOT

ADBENCH_47 = [
    "1_ALOI", "2_annthyroid", "3_backdoor", "4_breastw", "5_campaign",
    "6_cardio", "7_Cardiotocography", "8_celeba", "9_census", "10_cover",
    "11_donors", "12_fault", "13_fraud", "14_glass", "15_Hepatitis",
    "16_http", "17_InternetAds", "18_Ionosphere", "19_landsat", "20_letter",
    "21_Lymphography", "22_magic.gamma", "23_mammography", "24_mnist",
    "25_musk", "26_optdigits", "27_PageBlocks", "28_pendigits", "29_Pima",
    "30_satellite", "31_satimage-2", "32_shuttle", "33_skin", "34_smtp",
    "35_SpamBase", "36_speech", "37_Stamps", "38_thyroid", "39_vertebral",
    "40_vowels", "41_Waveform", "42_WBC", "43_WDBC", "44_Wilt",
    "45_wine", "46_WPBC", "47_yeast",
]

KAR_DATASETS = [
    "39_vertebral", "2_annthyroid", "29_Pima",
    "18_Ionosphere", "43_WDBC", "6_cardio",
]

TIERS = ("small", "medium", "high_dim", "large")

MAX_REF = 2000


def find_dataset(name: str) -> Path:
    """Locate an ADBench ``.npz`` under the dataset root, flat or tiered.

    Parameters
    ----------
    name : str
        Dataset name, for instance ``"6_cardio"``.

    Returns
    -------
    pathlib.Path
        Path to the archive.

    Raises
    ------
    FileNotFoundError
        When the file is absent; run ``scripts/download_datasets.py``.
    """
    flat = DATA_ROOT / f"{name}.npz"
    if flat.exists():
        return flat
    for tier in TIERS:
        path = DATA_ROOT / tier / f"{name}.npz"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{name}.npz not found under {DATA_ROOT}. Run scripts/download_datasets.py."
    )


def load_split(name: str, seed: int, test_frac: float = 0.5):
    """Semi-supervised split: train on normals only, test on the rest plus anomalies.

    The normal rows are split by `seed`; the held-out normals are stacked with every
    anomaly to form the test set. A ``StandardScaler`` is fitted on the training rows
    and applied to both, so the arrays returned here are already standardized.

    Parameters
    ----------
    name : str
        Dataset name.
    seed : int
        Split seed, also the seed :func:`subsample` expects.
    test_frac : float, default=0.5
        Fraction of the normal rows held out.

    Returns
    -------
    X_train : ndarray of shape (n_train, n_features)
        Scaled normal rows.
    X_test : ndarray of shape (n_test, n_features)
        Scaled held-out normals stacked above the anomalies.
    y_test : ndarray of shape (n_test,)
        ``1`` for anomaly, ``0`` for normal.
    """
    blob = np.load(find_dataset(name), allow_pickle=True)
    X = blob["X"].astype(np.float64)
    y = blob["y"].astype(int)
    X_norm = X[y == 0]
    X_anom = X[y == 1]
    y_norm = np.zeros(len(X_norm), dtype=int)
    X_tr, X_held, _, y_held = train_test_split(
        X_norm, y_norm, test_size=test_frac, random_state=seed, stratify=y_norm
    )
    X_test = np.vstack([X_held, X_anom])
    y_test = np.concatenate([y_held, np.ones(len(X_anom), dtype=int)])
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_test), y_test


def subsample(X: np.ndarray, seed: int, max_ref: int = MAX_REF) -> np.ndarray:
    """Cap the kernel anchor set at `max_ref` rows, in permuted order.

    The generator is seeded by the SPLIT seed rather than a constant, so the anchor
    set differs across seeds by design, and the returned rows are NOT sorted back into
    input order: :func:`karkcm.kcm.build_h_grid` subsamples by position, so reordering
    would change the bandwidth grid. ``KCM(random_state=seed).fit(X)`` reproduces this
    selection exactly for the same integer `seed`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training rows.
    seed : int
        The split seed.
    max_ref : int, default=2000
        Cap. Values <= 0 mean no cap.

    Returns
    -------
    ndarray of shape (min(n_samples, max_ref), n_features)
        `X` itself when no cap applies, otherwise a permuted subset.
    """
    if max_ref <= 0 or len(X) <= max_ref:
        return X
    idx = np.random.default_rng(seed).permutation(len(X))[:max_ref]
    return X[idx]
