# Examples

Three scripts, each self-contained and runnable from a clone (`python examples/<file>`)
or after `pip install -e .`. Only `01_quickstart.py` runs with no data at all; the other
two want one ADBench file, 68 KB:

```bash
python scripts/download_datasets.py --datasets 6_cardio
```

| script | what it shows | runtime |
|---|---|---|
| `01_quickstart.py` | fit on normals, score a contaminated test set, read `predict` flags, and see `roc_auc_score(y, decision_function(X))` return `1 - AUROC` | 0.3 s |
| `02_pipeline_gridsearch.py` | `Pipeline` + `GridSearchCV` done correctly for a one-class detector: `PredefinedSplit`, the `anomaly_auroc` scorer, and `refit=False` | 1.4 s |
| `03_diagnostics.py` | the leave-one-out bandwidth curve, `bandwidth_at_grid_edge_`, `kernel_mass` underflow, and a per-feature answer to "why was this row flagged" out of `reconstruct` | 0.6 s |

Measured on 6_cardio, seed 0:

- `01_quickstart.py` prints AUROC `0.9614`, which is the paper's per-run value for this
  dataset and seed, and a false-alarm rate of `0.093` on held-out normals at
  `contamination=0.1`.
- `02_pipeline_gridsearch.py` prints the same six bandwidths scored twice. With
  `anomaly_auroc` the best is `0.9795`; with `scoring="roc_auc"` the two columns sum to
  1.00 on every row and the argmax lands on the worst bandwidth in the grid. It then
  refits the winner both ways: on the clean training rows it scores `0.9536` on
  held-out data, and on train plus the labelled validation fold, which is what
  `refit=True` would do, it scores `0.5681`.
- `03_diagnostics.py --dataset 24_mnist` reports 48 of 4152 test rows with
  `kernel_mass == 0`, whose score is then exactly `||z||`, distance from the coordinate
  origin.

## matplotlib

`03_diagnostics.py` writes a two-panel PNG when matplotlib is importable and prints the
same content as text when it is not. matplotlib is deliberately not a dependency of
this package, in any extra.

```bash
python examples/03_diagnostics.py --out kcm_diagnostics.png
python examples/03_diagnostics.py --no-plot
```

## Benchmark

`../benchmarks/compare_sklearn.py` puts KCM next to `IsolationForest`,
`LocalOutlierFactor` and `OneClassSVM` on the same splits and prints AUROC, AUPRC,
wall clock and mean rank. Twelve datasets and three seeds, 144 runs, about 15 s.

```bash
python benchmarks/compare_sklearn.py
python benchmarks/compare_sklearn.py --datasets 6_cardio 29_Pima --seeds 0 --out runs.csv
```

The paper's own comparison is larger and already committed: 47 datasets against 46
detectors in `results/section4_kcm/leaderboard.csv`.
