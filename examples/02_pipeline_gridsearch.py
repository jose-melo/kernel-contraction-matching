"""Tune KCM inside a Pipeline with GridSearchCV, without poisoning the anchor set.

KCM is a one-class detector: it must be fitted on rows that are all normal, and it can
only be model-selected against rows that are labelled. Those are two different sets, so
a plain k-fold cross-validation is wrong twice over. The pattern that is right:

  PredefinedSplit  puts the clean training rows in the train fold (test_fold = -1) and
                   the labelled validation rows in the single test fold.
  scoring          is karkcm.anomaly_auroc. scoring="roc_auc" returns 1 - AUROC here,
                   with no error raised, and this script prints both columns so the
                   difference is visible rather than argued.
  refit=False      matters. GridSearchCV's refit fits the winner on train plus test
                   folds, and the validation rows contain anomalies, so refit=True
                   silently puts anomalies into the kernel anchor set. The script
                   measures what that costs.

Needs one ADBench file, 68 KB:

  python scripts/download_datasets.py --datasets 6_cardio
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

if importlib.util.find_spec("karkcm") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from karkcm import anomaly_auprc, anomaly_auroc, make_kcm_pipeline
from karkcm.data import load_split

GRID = {"kcm__bandwidth": ["loo", 0.25, 0.5, 1.0, 2.0, 4.0]}


def _search(X, y, split, scoring):
    gs = GridSearchCV(
        make_kcm_pipeline(), GRID, scoring=scoring, cv=split, refit=False, n_jobs=1
    )
    gs.fit(X, y)
    return gs.cv_results_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="6_cardio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        X_train, X_lab, y_lab = load_split(args.dataset, seed=args.seed)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    X_val, X_hold, y_val, y_hold = train_test_split(
        X_lab, y_lab, test_size=0.5, random_state=args.seed, stratify=y_lab
    )
    X = np.vstack([X_train, X_val])
    y = np.r_[np.zeros(len(X_train), dtype=int), y_val]
    split = PredefinedSplit(
        np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)]
    )

    print(f"dataset            {args.dataset}, seed {args.seed}")
    print(f"train fold         {len(X_train)} rows, all normal")
    print(f"validation fold    {len(X_val)} rows, {int(y_val.sum())} anomalies")
    print(f"held out           {len(X_hold)} rows, {int(y_hold.sum())} anomalies")
    print()

    right = _search(X, y, split, anomaly_auroc)
    wrong = _search(X, y, split, "roc_auc")

    print("grid, scored on the validation fold")
    print(f"{'kcm__bandwidth':>16}{'anomaly_auroc':>16}{'scoring=roc_auc':>17}"
          f"{'sum':>7}")
    for params, a, b in zip(
        right["params"], right["mean_test_score"], wrong["mean_test_score"]
    ):
        print(f"{str(params['kcm__bandwidth']):>16}{a:>16.4f}{b:>17.4f}{a + b:>7.2f}")

    best = right["params"][int(np.argmax(right["mean_test_score"]))]
    worst = wrong["params"][int(np.argmax(wrong["mean_test_score"]))]
    print()
    print(f"argmax of anomaly_auroc     {best['kcm__bandwidth']}")
    print(f"argmax of scoring='roc_auc' {worst['kcm__bandwidth']}   "
          "the two columns sum to 1, so this maximises the wrong thing")
    print()

    clean = make_kcm_pipeline(bandwidth=best["kcm__bandwidth"]).fit(X_train)
    dirty = make_kcm_pipeline(bandwidth=best["kcm__bandwidth"]).fit(X)
    print("the winner, refitted two ways, scored on the held-out rows")
    print(f"{'refit on':>32}{'anchors':>9}{'AUROC':>9}{'AUPRC':>9}")
    print(f"{'clean train rows (refit=False)':>32}{clean[-1].n_reference_:>9}"
          f"{anomaly_auroc(clean, X_hold, y_hold):>9.4f}"
          f"{anomaly_auprc(clean, X_hold, y_hold):>9.4f}")
    print(f"{'train + validation (refit=True)':>32}{dirty[-1].n_reference_:>9}"
          f"{anomaly_auroc(dirty, X_hold, y_hold):>9.4f}"
          f"{anomaly_auprc(dirty, X_hold, y_hold):>9.4f}")
    print()
    print(f"the second anchor set is {y.mean():.1%} anomalies, and a kernel smoother "
          "reconstructs them")
    print("as happily as it reconstructs normals, so their residual, and the score, "
          "collapses.")


if __name__ == "__main__":
    main()
