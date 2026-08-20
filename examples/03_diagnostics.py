"""Read the three things a closed-form detector can tell you that a trained one cannot.

  loo_curve_               the leave-one-out objective at every candidate bandwidth,
                           i.e. the whole model-selection surface, not just its argmin.
  bandwidth_at_grid_edge_  True when that argmin sat at an endpoint, so bandwidth_ is
                           an artefact of the grid rather than a choice it made.
  kernel_mass(X)           the total kernel weight a query row receives. Exactly 0.0
                           means the Gaussian underflowed, NW(z) is the coordinate
                           origin, and the score is exactly ||z||. Try --dataset
                           24_mnist, where 48 of 4152 test rows are in that state.

It then answers "why was this row flagged" per feature, out of reconstruct(X): the
score is ||x - NW(x)||, so the per-feature terms of that norm are the whole
explanation, with no attribution method in between.

A PNG is written when matplotlib is importable; otherwise every panel is printed as
text and nothing else changes. matplotlib is not a dependency of this package.

  python scripts/download_datasets.py --datasets 6_cardio
  python examples/03_diagnostics.py --out kcm_diagnostics.png
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

if importlib.util.find_spec("karkcm") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from karkcm import KCM
from karkcm.data import load_split


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bar(value, lo, hi, width=32):
    if hi <= lo:
        return ""
    return "#" * max(1, int(round(width * (value - lo) / (hi - lo))))


def _print_curve(m):
    print("[bandwidth selection]")
    print(f"  anchors                  {m.n_reference_}")
    print(f"  bandwidth_               {m.bandwidth_:.6f}")
    print(f"  grid                     {m.bandwidth_grid_[0]:.4f} to "
          f"{m.bandwidth_grid_[-1]:.4f}, {len(m.bandwidth_grid_)} points")
    print(f"  argmin index             {int(np.argmin(m.loo_curve_))} of "
          f"{len(m.loo_curve_) - 1}")
    print(f"  bandwidth_at_grid_edge_  {m.bandwidth_at_grid_edge_}")
    print()
    lo, hi = float(m.loo_curve_.min()), float(m.loo_curve_.max())
    print(f"{'h':>10}{'loo objective':>16}   curve")
    for h, loss in zip(m.bandwidth_grid_, m.loo_curve_):
        mark = " <- selected" if h == m.bandwidth_ else ""
        print(f"{h:>10.4f}{loss:>16.6f}   {_bar(loss, lo, hi)}{mark}")
    print()


def _print_mass(m, X_te):
    mass = m.kernel_mass(X_te)
    score = m.anomaly_score(X_te)
    dead = mass == 0.0
    print("[kernel mass on the test set]")
    print(f"  rows                     {len(X_te)}")
    print(f"  zero mass rows           {int(dead.sum())}")
    print(f"  mass below 1e-12         {int((mass < 1e-12).sum())}   these already "
          "reconstruct to the origin")
    print(f"  min                      {mass.min():.3e}")
    print(f"  median                   {np.median(mass):.3e}")
    if dead.any():
        norm = np.linalg.norm(X_te[dead], axis=1)
        agree = bool(np.allclose(score[dead], norm, atol=1e-9))
        print(f"  score == ||z|| there     {agree}")
        print(f"  their mean score         {score[dead].mean():.4f}, versus "
              f"{score[~dead].mean():.4f} elsewhere")
    print()
    return mass, score


def _print_why(m, X_te, y_te, score, mass, k):
    recon = m.reconstruct(X_te)
    resid = X_te - recon
    order = np.argsort(-score)[:k]
    print(f"[why they were flagged: top {k} by anomaly_score]")
    print("  per-feature terms of ||x - NW(x)||, largest three")
    for rank, i in enumerate(order, 1):
        label = "anomaly" if y_te[i] else "normal"
        print(f"  {rank}. row {i:<6} score {score[i]:>8.4f}  mass {mass[i]:>10.3e}  "
              f"{label}")
        for j in np.argsort(-np.abs(resid[i]))[:3]:
            print(f"       feature {j:<3} x {X_te[i, j]:>9.3f}   "
                  f"NW(x) {recon[i, j]:>9.3f}   residual {resid[i, j]:>+9.3f}")
    print()
    return recon, resid, order


def _plot(plt, m, X_te, resid, order, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(m.bandwidth_grid_, m.loo_curve_, marker="o", ms=3)
    ax.axvline(m.bandwidth_, color="crimson", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("bandwidth h")
    ax.set_ylabel("leave-one-out objective")
    ax.set_title(f"selected h = {m.bandwidth_:.4f}")

    i = order[0]
    ax = axes[1]
    top = np.argsort(-np.abs(resid[i]))[:12][::-1]
    ax.barh(np.arange(len(top)), resid[i, top], color="steelblue")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"f{j}" for j in top], fontsize=8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("x - NW(x)")
    ax.set_title(f"row {i}, score {np.linalg.norm(resid[i]):.3f}")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="6_cardio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default="kcm_diagnostics.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    try:
        X_tr, X_te, y_te = load_split(args.dataset, seed=args.seed)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    m = KCM(random_state=args.seed, assume_scaled=True).fit(X_tr)
    print(f"dataset {args.dataset}, seed {args.seed}, "
          f"{len(X_tr)} train rows, {X_tr.shape[1]} features\n")

    _print_curve(m)
    mass, score = _print_mass(m, X_te)
    recon, resid, order = _print_why(m, X_te, y_te, score, mass, args.top)

    if args.no_plot:
        return
    plt = _pyplot()
    if plt is None:
        print("[plot] matplotlib is not installed, so the two panels above stayed "
              "text. pip install matplotlib")
        return
    _plot(plt, m, X_te, resid, order, args.out)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
