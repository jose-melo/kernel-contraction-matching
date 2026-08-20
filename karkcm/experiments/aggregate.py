import collections
import csv
import json

import numpy as np

from ..baselines import DETECTORS
from ..data import KAR_DATASETS
from ..paths import SEC3, SEC6, prepare
from .collapse_gap import UNREGULARIZED_LAMBDA


def load_json_glob(pattern_root, pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pattern_root.glob(pattern))]


def write_csv(path, header, rows, overwrite=True) -> None:
    path = prepare(path, overwrite)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[ok] {path}")


def table2_collapse_gap() -> None:
    """Aggregate the Section 3 collapse-gap runs into Table 2.

    Reads the unregularized runs only, ``<dataset>_s<seed>_l<lambda>*.json``. The
    trailing wildcard is load-bearing: the published deepsvdd records carry a
    ``_nobias`` suffix, recording the bias-free linear layers Deep SVDD needs to
    exclude the trivial constant solution, and they are that detector's only runs
    rather than an ablation beside them. Dropping them empties the Deep SVDD row.
    Duplicate keys raise rather than being resolved by filename order, so a re-run
    written beside a published record is reported instead of silently winning.
    """
    pattern = f"*/*_s[0-9]*_l{UNREGULARIZED_LAMBDA}*.json"
    records = [
        (p, json.loads(p.read_text()))
        for p in sorted((SEC3 / "table2_source").glob(pattern))
    ]
    if not records:
        print(f"[warn] no runs matching {pattern} under {SEC3 / 'table2_source'}")
        return
    seen = {}
    for path, r in records:
        key = (r["detector"], r["dataset"], r["seed"])
        if key in seen:
            raise ValueError(
                f"two runs claim {key}: {seen[key].name} and {path.name}. Table 2 is "
                "a mean over seeds, so a duplicate key would be resolved by filename "
                "order. Remove one, or move the ablation out of table2_source."
            )
        seen[key] = path
    g = collections.defaultdict(list)
    for _, r in records:
        g[(r["detector"], r["dataset"])].append(r["gap"])
    rows = []
    for det in DETECTORS:
        row = [det]
        for ds in KAR_DATASETS:
            vals = g.get((det, ds), [])
            row.append(f"{np.mean(vals):.3f}" if vals else "")
        rows.append(row)
    write_csv(SEC3 / "table2_summary.csv",
              ["detector"] + [d.replace("_", "-") for d in KAR_DATASETS], rows)


def table4_rho_ablation() -> None:
    records = load_json_glob(SEC6 / "rho_ablation", "*.json")
    if not records:
        print(f"[warn] no runs under {SEC6 / 'rho_ablation'}")
        return
    g = collections.defaultdict(list)
    for r in records:
        g[(r["dataset"], r["rho"])].append(r)
    rows = []
    for key in sorted(g):
        rs = g[key]
        rows.append([
            key[0], key[1],
            f"{np.mean([x['final_auroc'] for x in rs]):.4f}",
            f"{np.mean([x['gap'] for x in rs]):+.4f}",
            f"{np.mean([x['kcm_auroc'] for x in rs]):.4f}",
        ])
    write_csv(SEC6 / "rho_ablation_summary.csv",
              ["dataset", "rho", "final_auroc", "gap", "kcm_auroc"], rows)


def table5_backbone() -> None:
    records = load_json_glob(SEC6 / "backbone_collapse", "*.json")
    if not records:
        print(f"[warn] no runs under {SEC6 / 'backbone_collapse'}")
        return
    g = collections.defaultdict(list)
    for r in records:
        g[r["dataset"]].append(r)
    rows = []
    for ds in sorted(g):
        rs = g[ds]
        rows.append([
            ds,
            *(f"{np.mean([x['final_auroc'][bb] for x in rs]):.4f}"
              for bb in ("tccm", "ae", "deepsvdd")),
            f"{np.max([x['final_spread'] for x in rs]):.2e}",
        ])
    write_csv(SEC6 / "backbone_collapse_summary.csv",
              ["dataset", "via_tccm", "via_ae", "via_deepsvdd", "max_spread"], rows)


def main() -> None:
    table2_collapse_gap()
    table4_rho_ablation()
    table5_backbone()


if __name__ == "__main__":
    main()
