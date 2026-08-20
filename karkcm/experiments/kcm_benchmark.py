import argparse
import time

import pandas as pd

from ..data import ADBENCH_47, MAX_REF, load_split, subsample
from ..kcm import KCMAnchor
from ..metrics import auprc, auroc
from ..paths import SEC4, prepare


def evaluate(dataset: str, seed: int, max_ref: int, n_grid: int, trim_frac: float) -> dict:
    t0 = time.time()
    X_tr_full, X_te, y_te = load_split(dataset, seed=seed)
    X_tr = subsample(X_tr_full, seed, max_ref)

    t_fit = time.time()
    anchor = KCMAnchor.fit(X_tr, n_grid=n_grid, trim_frac=trim_frac)
    fit_time = time.time() - t_fit

    t_score = time.time()
    sc = anchor.score_np(X_te)
    score_time = time.time() - t_score

    return {
        "dataset": dataset,
        "seed": seed,
        "n_train_full": int(len(X_tr_full)),
        "n_train_ref": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "D": int(X_tr.shape[1]),
        "anom_frac": float(y_te.mean()),
        "h": float(anchor.h),
        "auroc": auroc(y_te, sc),
        "auprc": auprc(y_te, sc),
        "fit_time": float(fit_time),
        "score_time": float(score_time),
        "wallclock_s": float(time.time() - t0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ADBENCH_47)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-ref", type=int, default=MAX_REF)
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument("--trim-frac", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = prepare(args.out or SEC4 / "kcm_adbench47.csv", args.overwrite)

    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            try:
                row = evaluate(ds, s, args.max_ref, args.n_grid, args.trim_frac)
            except FileNotFoundError as exc:
                print(f"[skip] {ds} s={s}: {exc}", flush=True)
                continue
            rows.append(row)
            print(f"[{ds:<22} s={s}] auroc={row['auroc']:.4f} "
                  f"({row['wallclock_s']:.1f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
