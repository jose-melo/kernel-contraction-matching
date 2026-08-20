import csv
from functools import lru_cache
from pathlib import Path

import pytest

from karkcm import KCM
from karkcm.data import MAX_REF, find_dataset, load_split, subsample
from karkcm.kcm import KCMAnchor

PAPER_DATASETS = (
    "39_vertebral",
    "29_Pima",
    "18_Ionosphere",
    "43_WDBC",
    "6_cardio",
    "2_annthyroid",
)

SEEDS = (0, 1, 2)

PAPER_TABLE4 = {
    "2_annthyroid": {"kcm": 0.929, "rho": 0.5, "kar": 0.948},
    "29_Pima": {"kcm": 0.701, "rho": 0.5, "kar": 0.730},
    "18_Ionosphere": {"kcm": 0.974, "rho": 1.0, "kar": 0.976},
    "43_WDBC": {"kcm": 0.993, "rho": 1.0, "kar": 0.993},
    "6_cardio": {"kcm": 0.963, "rho": 0.5, "kar": 0.960},
    "39_vertebral": {"kcm": 0.550, "rho": 0.0, "kar": 0.550},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_RUN_CSV = REPO_ROOT / "results" / "section4_kcm" / "kcm_per_run_47.csv"


def requires_datasets(*names):
    missing = []
    for name in names:
        try:
            find_dataset(name)
        except FileNotFoundError:
            missing.append(name)
    return pytest.mark.skipif(
        bool(missing),
        reason="run: python scripts/download_datasets.py --datasets "
        + " ".join(missing or names),
    )


def have_dataset(name):
    try:
        find_dataset(name)
    except FileNotFoundError:
        return False
    return True


def read_per_run(datasets=None):
    if not PER_RUN_CSV.exists():
        return []
    with open(PER_RUN_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if datasets is None:
        return rows
    keep = set(datasets)
    return [r for r in rows if r["dataset"] in keep]


requires_results = pytest.mark.skipif(
    not PER_RUN_CSV.exists(),
    reason=f"{PER_RUN_CSV.name} is absent; the results tree ships with the git clone",
)


@lru_cache(maxsize=None)
def split(dataset, seed):
    return load_split(dataset, seed=seed)


@lru_cache(maxsize=None)
def anchor(dataset, seed):
    X_tr = split(dataset, seed)[0]
    return KCMAnchor.fit(subsample(X_tr, seed, MAX_REF), n_grid=21)


@lru_cache(maxsize=None)
def estimator(dataset, seed):
    X_tr = split(dataset, seed)[0]
    return KCM(random_state=seed, assume_scaled=True).fit(X_tr)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the full 141-row corpus regression",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: full-corpus regression, needs --runslow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
