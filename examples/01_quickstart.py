"""Fit KCM on normal rows, score a contaminated test set, read the flags.

Run it with no arguments. It uses the ADBench dataset 6_cardio when that file is
present (one 68 KB download: python scripts/download_datasets.py --datasets 6_cardio)
and falls back to scikit-learn's breast cancer data otherwise, so the script always
runs straight after pip install -e .

The three things worth taking away:

  anomaly_score  is the paper's number and is HIGH for anomalies. make_kcm_pipeline
                 forwards it, along with reconstruct and kernel_mass, so a pipeline
                 never forces you back to writing the bare minus sign.
  score_samples  is its negation, because scikit-learn wants HIGH for normal. Passing
                 decision_function to roc_auc_score with y == 1 meaning anomaly gives
                 1 - AUROC, silently.
  predict        thresholds at contamination, calibrated on leave-one-out residuals of
                 the anchor set, so the flag rate holds up out of sample.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if importlib.util.find_spec("karkcm") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from karkcm import anomaly_auprc, anomaly_auroc, make_kcm_pipeline
from karkcm.data import load_split


def _breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    normal, anomaly = X[y == 1], X[y == 0]
    clean, held = train_test_split(normal, test_size=0.5, random_state=0)
    X_test = np.vstack([held, anomaly])
    y_test = np.r_[np.zeros(len(held), dtype=int), np.ones(len(anomaly), dtype=int)]
    return "breast_cancer", clean, X_test, y_test


def _data():
    try:
        return ("6_cardio",) + load_split("6_cardio", seed=0)
    except FileNotFoundError:
        print("[note] 6_cardio.npz not found, using sklearn's breast cancer data.")
        print("[note] for the ADBench version, 68 KB:")
        print("[note]   python scripts/download_datasets.py --datasets 6_cardio\n")
        return _breast_cancer()


def main() -> None:
    name, X_train, X_test, y_test = _data()

    detector = make_kcm_pipeline().fit(X_train)
    anomaly = detector.anomaly_score(X_test)
    flags = detector.predict(X_test)
    kcm = detector.named_steps["kcm"]

    print(f"dataset             {name}")
    print(f"train rows          {len(X_train)} normals, {X_train.shape[1]} features")
    print(f"test rows           {len(X_test)}, {int(y_test.sum())} of them anomalies")
    print(f"bandwidth_          {kcm.bandwidth_:.6f}  selected by leave-one-out")
    print(f"bandwidth at edge   {kcm.bandwidth_at_grid_edge_}")
    print(f"anchors             {kcm.n_reference_}")
    print()
    print(f"AUROC               {anomaly_auroc(detector, X_test, y_test):.4f}")
    print(f"AUPRC               {anomaly_auprc(detector, X_test, y_test):.4f}")
    print(f"AUROC, sign flipped {roc_auc_score(y_test, detector.decision_function(X_test)):.4f}"
          "   roc_auc_score on decision_function returns 1 - AUROC")
    print()

    normal = y_test == 0
    print(f"contamination       {kcm.contamination}")
    print(f"flagged             {int((flags == -1).sum())} of {len(flags)} test rows")
    print(f"false alarm rate    {(flags[normal] == -1).mean():.3f} on held-out normals")
    print(f"recall              {(flags[~normal] == -1).mean():.3f} on anomalies")
    print()

    print("five most anomalous test rows")
    print(f"{'row':>6}{'anomaly_score':>15}{'kernel_mass':>14}{'label':>10}")
    mass = detector.kernel_mass(X_test)
    for i in np.argsort(-anomaly)[:5]:
        print(f"{i:>6}{anomaly[i]:>15.4f}{mass[i]:>14.3e}"
              f"{'anomaly' if y_test[i] else 'normal':>10}")


if __name__ == "__main__":
    main()
