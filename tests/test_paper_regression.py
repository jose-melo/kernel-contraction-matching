import numpy as np
import pytest

from karkcm.data import MAX_REF, subsample
from karkcm.metrics import auroc

from .conftest import (
    PAPER_DATASETS,
    PAPER_TABLE4,
    SEEDS,
    anchor,
    estimator,
    have_dataset,
    read_per_run,
    requires_datasets,
    requires_results,
    split,
)

ROWS = read_per_run(PAPER_DATASETS)
IDS = [f"{r['dataset']}-seed{r['seed']}" for r in ROWS]


@requires_datasets(*PAPER_DATASETS)
@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_frozen_path_reproduces_committed_runs(row):
    dataset, seed = row["dataset"], int(row["seed"])
    X_tr, X_te, y_te = split(dataset, seed)
    assert len(subsample(X_tr, seed, MAX_REF)) == int(row["n_train_ref"])
    measured = auroc(y_te, anchor(dataset, seed).score_np(X_te))
    assert abs(measured - float(row["auroc_iso_loo"])) <= 1e-12


@requires_datasets(*PAPER_DATASETS)
@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_estimator_reproduces_committed_runs(row):
    dataset, seed = row["dataset"], int(row["seed"])
    X_te, y_te = split(dataset, seed)[1:]
    m = estimator(dataset, seed)
    assert m.n_reference_ == int(row["n_train_ref"])
    measured = auroc(y_te, m.anomaly_score(X_te))
    deviation = abs(measured - float(row["auroc_iso_loo"]))
    assert deviation <= 1e-12, f"{dataset} seed {seed}: {deviation:.3e}"


@requires_results
def test_table4_means():
    """Pins the committed per-run CSV to the published Table 4."""
    for dataset, ref in PAPER_TABLE4.items():
        rows = read_per_run([dataset])
        assert sorted(int(r["seed"]) for r in rows) == [0, 1, 2]
        vals = [float(r["auroc_iso_loo"]) for r in rows]
        assert abs(float(np.mean(vals)) - ref["kcm"]) <= 0.001


@requires_datasets(*PAPER_DATASETS)
def test_estimator_reproduces_published_table4():
    """0.001 is the tightest honest bound: the printed table is rounded to 3 decimals."""
    assert set(PAPER_TABLE4) == set(PAPER_DATASETS)
    for dataset, ref in PAPER_TABLE4.items():
        vals = []
        for seed in SEEDS:
            X_te, y_te = split(dataset, seed)[1:]
            vals.append(auroc(y_te, estimator(dataset, seed).anomaly_score(X_te)))
        deviation = abs(float(np.mean(vals)) - ref["kcm"])
        assert deviation <= 0.001, f"{dataset}: {deviation:.5f}"


TIE_SENSITIVE = {("17_InternetAds", 2)}


@pytest.mark.slow
def test_full_corpus_regression():
    """140 of the 141 committed runs reproduce to 1.1e-16; one is tie-limited.

    17_InternetAds seed 2 has 13 test rows in 6 groups of scores equal to 1e-12, and
    a sub-1e-12 difference in the reference BLAS reorders one (anomaly, normal) pair.
    The measured deviation is 3.401e-06, which is exactly 1/(n_pos * n_neg), the
    smallest non-zero step AUROC can take. The tolerance below is that step, not a
    round number, so any real arithmetic drift still fails the assertion.
    """
    checked = 0
    for row in read_per_run():
        dataset, seed = row["dataset"], int(row["seed"])
        if not have_dataset(dataset):
            continue
        X_te, y_te = split(dataset, seed)[1:]
        measured = auroc(y_te, anchor(dataset, seed).score_np(X_te))
        deviation = abs(measured - float(row["auroc_iso_loo"]))
        if (dataset, seed) in TIE_SENSITIVE:
            n_pos = int((y_te == 1).sum())
            tol = 1.0 / (n_pos * (len(y_te) - n_pos))
        else:
            tol = 1e-12
        assert deviation <= tol, f"{dataset} seed {seed}: {deviation:.3e} > {tol:.3e}"
        checked += 1
    if checked == 0:
        pytest.skip("run: python scripts/download_datasets.py")
