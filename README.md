<p align="center">
  <img src="https://raw.githubusercontent.com/jose-melo/kernel-contraction-matching/main/docs/figures/logo.png" alt="KCM" width="180">
</p>

<h1 align="center">Kernel Contraction Matching (KCM)</h1>

<p align="center">
  Anomaly detection with no optimizer, no epoch axis, and no collapse.
  A drop-in scikit-learn estimator that ranks first by mean AUROC
  across the 47 ADBench datasets while training nothing.
</p>

<p align="center">
  <a href="#the-problem-fixed-target-detectors-get-worse-as-they-train">Problem</a> &bull;
  <a href="#install">Install</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#using-the-estimator">API</a> &bull;
  <a href="#reproducing-the-paper">Paper</a>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3799682.3841143"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1145%2F3799682.3841143-1f6feb"></a>
  <a href="https://pypi.org/project/kar-kcm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/kar-kcm"></a>
  <a href="https://pypi.org/project/kar-kcm/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/kar-kcm"></a>
  <a href="https://github.com/jose-melo/kernel-contraction-matching/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jose-melo/kernel-contraction-matching/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-compatible-f89939">
</p>

Reference implementation for *Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors
via Kernel-Anchored Locality Regularization*, CIKM 2026.

## Install

```bash
pip install kar-kcm                 # KCM only, no PyTorch
pip install "kar-kcm[kar]"          # adds KAR, which needs PyTorch
```

Reproducing the paper needs the repository itself, which carries the per-run records and the
experiment modules:

```bash
git clone https://github.com/jose-melo/kernel-contraction-matching
cd kernel-contraction-matching && pip install -e ".[paper]"
```

## The problem: fixed-target detectors get worse as they train

A *fixed-target* detector trains a network to match a target that carries no information about how
far a point lies from the data: a constant centre (Deep SVDD), the input itself (autoencoders), a
frozen random network (RDP), a noise endpoint (flow-matching). The anomaly score is the residual
that is left over.

As training converges that residual goes to zero **everywhere the model can reach, anomalies
included**. The score field flattens, and detection decays while the training loss keeps falling.

![Convergence collapse across fixed-target anomaly detectors](https://raw.githubusercontent.com/jose-melo/kernel-contraction-matching/main/docs/figures/fig1_collapse.png)

The score field is sharp early and flat late; the anomalies (stars) stop being separated from the
normal cluster (dots). Nine detectors on ADBench `cardio` all reach an early peak and then decay
toward convergence. Early stopping does not rescue this: the best epoch moves by two orders of
magnitude across seeds, and the loss you would monitor decreases monotonically the whole time.

## What KCM does

Replace the trained map with the kernel smoother a wide network approaches in its lazy regime. For
a query `z` and training set `X`, KCM predicts the kernel-weighted average of the training points
and scores the residual:

```
NW_h(z) = sum_i k_h(z, x_i) x_i / sum_i k_h(z, x_i)        Nadaraya-Watson reconstruction
score(z) = || z - NW_h(z) ||                                higher means more anomalous
```

Every training point's influence decays with distance, so as `z` leaves the data the prediction
levels off and the residual **grows**. The bandwidth `h` is chosen by leave-one-out reconstruction
error over a 21-point grid. There is no optimizer, no epoch axis, and nothing to early-stop.

## Quickstart

```python
from sklearn.datasets import make_blobs
from karkcm import make_kcm_pipeline

X_train, _ = make_blobs(n_samples=500, centers=1, random_state=0)
X_test, _ = make_blobs(n_samples=20, centers=[[8, 8]], random_state=0)

detector = make_kcm_pipeline().fit(X_train)

detector.predict(X_test)          # -1 outlier, +1 inlier
detector.anomaly_score(X_test)    # higher means more anomalous
```

`make_kcm_pipeline()` is `StandardScaler` plus `KCM`. KCM uses one isotropic bandwidth for every
feature, so scaled input is what you want; the pipeline handles it and forwards KCM's own methods.

## Results

Across the 47 ADBench datasets, against the 44 baselines ADBench ships:

| | mean AUROC | mean rank | neural training |
|---|---|---|---|
| **KCM** | **0.8710** | 7.06 | **none** |
| DTE-NP | 0.8652 | 6.81 | yes |
| LUNAR | 0.8603 | 8.62 | yes |
| KDE | 0.8452 | 10.49 | none |

![Nemenyi critical-difference diagram over the 47 datasets](https://raw.githubusercontent.com/jose-melo/kernel-contraction-matching/main/docs/figures/fig3_cd.png)

KCM sits in the leading group of the critical-difference diagram, with no trained parameters.

![Fit and inference time per detector](https://raw.githubusercontent.com/jose-melo/kernel-contraction-matching/main/docs/figures/fig4_speed.png)

Fit is a bandwidth search and a matrix product: median **0.40 s** per dataset on one CPU core,
where the neural detectors sit one to two orders of magnitude higher.

## Using the estimator

`KCM` follows the scikit-learn outlier-detector contract, so it clones, pickles, and drops into
`Pipeline` and `GridSearchCV`.

```python
from karkcm import KCM

KCM(bandwidth="loo", n_grid=21, trim_frac=0.0, max_reference=2000,
    contamination=0.1, chunk_size=4096, assume_scaled=False,
    novelty=True, random_state=None)
```

| method | returns |
|---|---|
| `anomaly_score(X)` | `\|\|z - NW(z)\|\|`, higher means more anomalous |
| `score_samples(X)` | `-anomaly_score(X)`, the scikit-learn orientation |
| `decision_function(X)` | `score_samples(X) - offset_`, negative means outlier |
| `predict(X)` | `-1` outlier, `+1` inlier |
| `reconstruct(X)` | the Nadaraya-Watson reconstruction, per feature |
| `kernel_mass(X)` | total kernel weight a query receives |

`reconstruct` is the explanation channel: it returns what the model considers a normal version of
each row, so the per-feature difference tells you *why* a point scored the way it did.

**Two modes, following `LocalOutlierFactor`.** `novelty=True` (the default) scores rows the model
was not fitted on, which is the setting the paper uses. `novelty=False` exposes `fit_predict` for
labelling the rows you fit on, using leave-one-out residuals so that no row is scored against
itself.

```python
detector = KCM().fit(X_train)          # novelty=True: score new data
labels = KCM(novelty=False).fit_predict(X)   # label X itself
```

`anomaly_auroc` and `anomaly_auprc` are scorers with the sign already handled, for use with
`GridSearchCV`.

## KAR, for a trainable backbone

When a trained network is a requirement rather than a choice, `KAR` keeps one and constrains it to
stay within a fixed radius `rho` of the KCM anchor, in KCM-residual units. That floor is what
removes the collapse: `rho = 0.5` is used unchanged on every dataset in the paper.

![KAR under prolonged training](https://raw.githubusercontent.com/jose-melo/kernel-contraction-matching/main/docs/figures/fig5_kar_curves.png)

KAR needs PyTorch, which KCM does not: `pip install "kar-kcm[kar]"`.

## Reproducing the paper

```bash
pip install -e ".[paper]"
python scripts/download_datasets.py

python -m karkcm.experiments.kcm_benchmark      # Sec. 5, KCM over 47 datasets
python -m karkcm.experiments.collapse_gap       # Sec. 3, peak-to-final gap
python -m karkcm.experiments.backbone           # Sec. 6, backbone independence
python -m karkcm.experiments.aggregate          # per-section summary tables
python -m karkcm.experiments.kar_ablation --dataset 6_cardio --rho 0.5
```

`results/` holds the per-run JSON and CSV records behind the paper's tables and figures, and
`results/figures/` the figures themselves. The experiment modules will not overwrite it: point
`KARKCM_RESULTS` at a directory of your own, or pass `--overwrite` deliberately.

`KCMAnchor` in `karkcm/kcm.py` is the frozen object the paper's experiments call. `KCM` is the
estimator-shaped front door onto the same arithmetic, and `tests/test_estimator_parity.py` locks
the two together.

The 47 ADBench `.npz` files are fetched from [ADBench](https://github.com/Minqi824/ADBench) and are
not redistributed here.

## Layout

```
karkcm/
  kcm.py            the kernel, the bandwidth search, the frozen anchor
  estimator.py      the scikit-learn estimator, pipeline and scorers
  kar.py            bounded correction and the KAR training loop
  nets.py           MLP, time-conditioned MLP, autoencoder
  baselines.py      the six fixed-target detectors of Sec. 3
  data.py           ADBench loading and the semi-supervised split
  experiments/      one runnable module per paper artifact
benchmarks/         KCM against scikit-learn's detectors
examples/           quickstart, pipeline and grid search, diagnostics
results/            the per-run records behind every table and figure
tests/
```

## Citing

```bibtex
@inproceedings{demelocosta2026kar,
  author    = {De Melo Costa, José Lucas and Popineau, Fabrice and
               Rimmel, Arpad and Doan, Bich-Liên},
  title     = {Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors
               via Kernel-Anchored Locality Regularization},
  booktitle = {Proceedings of the 35th ACM International Conference on Information
               and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3841143}
}
```

## License

MIT, see [LICENSE](LICENSE).
