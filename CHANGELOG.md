# Changelog

## 1.0.0 - CIKM 2026 camera-ready release

First public release, matching the paper (doi:10.1145/3799682.3841143).

- `karkcm.KCM`: KCM as a scikit-learn outlier detector.
- **API change: `KCM` takes a `novelty` parameter, defaulting to `True`.** It splits the
  labelling API the way `LocalOutlierFactor` does, and it exists because the two halves
  are not interchangeable on the same rows. `novelty=True` exposes `score_samples`,
  `decision_function` and `predict`, for rows the estimator was NOT fitted on, and hides
  `fit_predict`; `novelty=False` exposes `fit_predict`, which labels the fitted rows by
  their leave-one-out residuals, and hides the other three. Implemented with
  `sklearn.utils.metaestimators.available_if`, so the hidden half raises `AttributeError`
  at attribute access, before and after `fit`, and `hasattr` answers correctly.
  Measured justification: a row that was fitted on sits inside its own reference set,
  where `K(z, z) = 1` dominates its own Nadaraya-Watson reconstruction, so its residual is
  near zero and it ranks as maximally NORMAL. On 400 standard normals plus 8 points at
  radius 20, `KCM(contamination=0.02).fit(X).predict(X)` caught 0 of 8 planted anomalies
  at in-sample AUROC 0.1247, worse than chance, where `IsolationForest.fit_predict` caught
  8 of 8. The same estimator fitted on the 400 clean normals alone and scoring those 8
  points out of sample gets AUROC 1.0000 and catches 8 of 8, so the deficit is the
  self-inclusion, not the detector. `fit_predict` at `novelty=False` recovers AUROC 0.844,
  0.878 and 0.995 on three such draws (seeds 11, 3, 7) where the in-sample score gives
  0.125, 0.062 and 0.130.
  **`novelty` guards the mode, not the rows.** It removes the `fit_predict`-shaped route
  into the in-sample ranking and makes a `fit_predict` / `fit(X).predict(X)` disagreement
  impossible on one unmodified object. It does **not** make the inverted path unreachable:
  the `KCM(contamination=0.02).fit(X).predict(X)` above still runs at the default and still
  catches 0 of 8 at AUROC 0.1247. Nothing in the estimator knows which rows it was fitted on.
  **The default is INVERTED with respect to `LocalOutlierFactor`, whose default is
  `novelty=False`.** Deliberate: the paper's protocol fits on clean normals and scores
  held-out rows, so a reader who copies a LOF snippet must not land on the in-sample path
  by default. `anomaly_score`, `reconstruct` and `kernel_mass` are diagnostics on arbitrary
  rows and stay available in BOTH modes; the guard is on the label-producing API, not on
  the arithmetic.
- `check_estimator` on scikit-learn 1.8.0: `KCM()` runs 46 checks, 44 passed, 2 expected
  failures, both of them `check_outliers_train` (scikit-learn runs it twice, once
  memmap-backed). `KCM(novelty=False)` runs 44 and passes all of them with nothing
  excused: three checks scikit-learn generates only for an estimator exposing `predict`
  are not generated (`check_outliers_train`, twice, and
  `check_classifier_data_not_an_array`), while `check_outliers_fit_predict` is, and passes,
  because it compares against `fit(X).predict(X)` only `if hasattr(estimator, "predict")`.
  `46 - 3 + 1 = 44`. The deviation is bandwidth-dependent, so the excuse
  is pinned per configuration rather than blanket: `KCM(assume_scaled=True, bandwidth=0.5)`,
  `KCM(assume_scaled=True, trim_frac=0.1, n_grid=5)` and
  `KCM(assume_scaled=True, chunk_size=7, contamination=0.25)` pass all 46 with nothing
  excused, and a blanket excuse would have turned those three into stale cover
  (`tests/test_sklearn_conformance.py`, `CONFIGURATIONS`). The check counts are derived,
  not pinned, because scikit-learn adds and removes checks between minor releases.
- `import karkcm` no longer imports PyTorch. KCM is numpy, scipy and scikit-learn only;
  `pip install "kar-kcm[kar]"` adds torch for KAR.
- `predict` thresholds on leave-one-out anchor residuals, so `contamination` is honoured
  out of sample.
- `fit_predict`, at `novelty=False`, is **transductive**, like
  `LocalOutlierFactor.fit_predict`: it labels the anchors by their leave-one-out
  residuals, which are not self-reconstructed. It is not comparable with
  `fit(X).predict(X)`, because no single `KCM` exposes both halves at once.
- `make_kcm_pipeline()` returns a `KCMPipeline`, a `Pipeline` subclass that also forwards
  `anomaly_score`, `reconstruct` and `kernel_mass` to the final step. A plain `Pipeline`
  forwards none of the three.
- The reproduction path is unchanged: `KCMAnchor` and `karkcm/experiments/*` produce the same
  numbers as before. Both ship in the source distribution; the wheel carries the importable
  package only.
- `results/` is now write-protected. Every experiment module routes its output through
  `karkcm.paths.prepare`, which raises `WouldOverwritePublished` rather than replacing a file
  that ships with the paper. `KARKCM_RESULTS=/some/dir` or `--overwrite` opt out. The committed
  per-run JSONs carry fields the current scripts do not emit (`auprc`, `h_kcm`, `kcm_auroc`,
  `spearman_history`, per-phase timings), so an unguarded re-run kept the arithmetic and lost
  the record.
- `experiments.aggregate` reads `<dataset>_s<seed>_l0.0*.json` and **raises** on a duplicate
  `(detector, dataset, seed)` instead of resolving it by filename order. The trailing wildcard
  matters: the committed `deepsvdd` records carry a `_nobias` suffix for the bias-free layers
  Deep SVDD requires, and are that detector's only runs. On the committed tree the glob matches
  108 files for 108 keys and every Table 2 row is populated.
- The float64 KCM path is bitwise reproducible; the float32 KAR path is not. Of the six
  Table 4 configurations, one reproduces bit-exactly on a re-run and one does so trivially
  at `rho = 0`, and the rest move by at most `9.3e-5`.
