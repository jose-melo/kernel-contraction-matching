"""KCM against IsolationForest, LocalOutlierFactor and OneClassSVM on ADBench.

Every detector sees exactly the same arrays: karkcm.data.load_split fits a
StandardScaler on the normal training rows and applies it to both halves, so this is a
semi-supervised protocol, fitted on normals only and scored on held-out normals plus
every anomaly. Scores are taken as -score_samples(X) for all four, which is the
scikit-learn convention KCM.anomaly_score exists to make explicit.

Wall clock is fit plus scoring on one core, and it is a property of this machine and
this BLAS, not of the methods. What it is good for is the shape: KCM has no epoch axis
and no optimizer, and its cost is quadratic in the anchor set, which max_reference caps
at 2000.

The default twelve datasets and three seeds finish in about a minute. This is not the
paper's leaderboard: that is 47 datasets against 46 detectors, computed once and
committed to results/section4_kcm/leaderboard.csv, where KCM ranks 1 at mean AUROC
0.8710.

  python scripts/download_datasets.py --datasets 6_cardio 29_Pima
  python benchmarks/compare_sklearn.py --datasets 6_cardio 29_Pima --seeds 0
  python benchmarks/compare_sklearn.py --out runs.csv
"""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

if importlib.util.find_spec("karkcm") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from karkcm import KCM
from karkcm.data import load_split
from karkcm.metrics import auprc, auroc

DEFAULT_DATASETS = [
    "39_vertebral",
    "29_Pima",
    "18_Ionosphere",
    "43_WDBC",
    "47_yeast",
    "20_letter",
    "6_cardio",
    "38_thyroid",
    "44_Wilt",
    "2_annthyroid",
    "30_satellite",
    "24_mnist",
]

METHODS = ("KCM", "IForest", "LOF", "OCSVM")


def _build(method, seed):
    if method == "KCM":
        return KCM(random_state=seed, assume_scaled=True)
    if method == "IForest":
        return IsolationForest(random_state=seed)
    if method == "LOF":
        return LocalOutlierFactor(novelty=True)
    return OneClassSVM(gamma="scale")


def _run(method, seed, X_tr, X_te, y_te):
    t0 = time.perf_counter()
    est = _build(method, seed).fit(X_tr)
    scores = -est.score_samples(X_te)
    seconds = time.perf_counter() - t0
    return {
        "auroc": auroc(y_te, scores),
        "auprc": auprc(y_te, scores),
        "seconds": seconds,
    }


def _table(title, rows, datasets, methods, shape, fmt):
    print(f"[{title}]")
    head = f"{'dataset':<16}{'n_train':>8}{'n_test':>8}{'D':>5}"
    for m in methods:
        head += f"{m:>10}"
    print(head)
    for ds in datasets:
        cells = [f"{ds:<16}{shape[ds][0]:>8}{shape[ds][1]:>8}{shape[ds][2]:>5}"]
        best = max(
            np.mean([r[title] for r in rows if r["dataset"] == ds and r["method"] == m])
            for m in methods
        )
        for m in methods:
            v = float(
                np.mean(
                    [r[title] for r in rows if r["dataset"] == ds and r["method"] == m]
                )
            )
            mark = "*" if title != "seconds" and v == best else " "
            cells.append(f"{format(v, fmt) + mark:>10}")
        print("".join(cells))
    print(f"{'mean':<16}{'':>8}{'':>8}{'':>5}", end="")
    for m in methods:
        v = float(np.mean([r[title] for r in rows if r["method"] == m]))
        print(f"{format(v, fmt) + ' ':>10}", end="")
    print("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    shape = {}
    datasets = []
    t0 = time.perf_counter()
    for ds in args.datasets:
        try:
            probe = load_split(ds, seed=args.seeds[0])
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        shape[ds] = (len(probe[0]), len(probe[1]), probe[0].shape[1])
        datasets.append(ds)
        for seed in args.seeds:
            X_tr, X_te, y_te = probe if seed == args.seeds[0] else load_split(ds, seed)
            for method in args.methods:
                row = _run(method, seed, X_tr, X_te, y_te)
                row.update(dataset=ds, seed=seed, method=method)
                rows.append(row)
        done = [r for r in rows if r["dataset"] == ds]
        print(f"[{ds:<16}] "
              + "  ".join(
                  f"{m} {np.mean([r['auroc'] for r in done if r['method'] == m]):.3f}"
                  for m in args.methods
              ), flush=True)

    if not rows:
        raise SystemExit("no datasets found: run scripts/download_datasets.py")

    print(f"\n{len(datasets)} datasets, {len(args.seeds)} seeds, "
          f"{len(rows)} runs, {time.perf_counter() - t0:.1f}s total\n")

    _table("auroc", rows, datasets, args.methods, shape, ".4f")
    _table("auprc", rows, datasets, args.methods, shape, ".4f")
    _table("seconds", rows, datasets, args.methods, shape, ".3f")

    print("[mean rank by AUROC, over every dataset and seed, 1 is best]")
    ranks = {m: [] for m in args.methods}
    wins = {m: 0 for m in args.methods}
    for ds in datasets:
        for seed in args.seeds:
            cell = {
                r["method"]: r["auroc"]
                for r in rows
                if r["dataset"] == ds and r["seed"] == seed
            }
            vals = [cell[m] for m in args.methods]
            for m, rk in zip(args.methods, rankdata([-v for v in vals])):
                ranks[m].append(rk)
            top = [m for m, v in zip(args.methods, vals) if v == max(vals)]
            if len(top) == 1:
                wins[top[0]] += 1
    n_runs = len(datasets) * len(args.seeds)
    for m in args.methods:
        print(f"  {m:<10}{np.mean(ranks[m]):>8.2f}   outright wins "
              f"{wins[m]:>3} of {n_runs}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["dataset", "seed", "method", "auroc", "auprc", "seconds"]
            )
            w.writeheader()
            w.writerows(
                {k: r[k] for k in w.fieldnames} for r in rows
            )
        print(f"\n[ok] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
