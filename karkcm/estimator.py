"""scikit-learn estimator front door for Kernel Contraction Matching."""

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from .kcm import build_h_grid, rbf_gram, trimmed_loo_bandwidth


class NotStandardizedWarning(UserWarning):
    """Warning that `X` is not standardized and KCM uses one isotropic bandwidth.

    Its own class so it can be silenced on its own with
    ``warnings.filterwarnings("ignore", category=NotStandardizedWarning)`` rather than
    by muting every :class:`UserWarning`.
    """


class KCM(OutlierMixin, BaseEstimator):
    """Kernel Contraction Matching anomaly detector.

    KCM replaces the trained map of a fixed-target detector with the kernel smoother a
    wide network approaches in its lazy regime. It has no optimizer and no epoch axis,
    so it cannot suffer the convergence collapse of Section 3 of the paper. Fitting
    selects one isotropic RBF bandwidth by trimmed leave-one-out reconstruction error
    over a log-spaced grid; scoring a row `z` returns ``||z - NW(z)||``, its distance
    from the Nadaraya-Watson reconstruction built out of the reference set.

    Parameters
    ----------
    bandwidth : {"loo"} or float, default="loo"
        ``"loo"`` selects the bandwidth by trimmed leave-one-out reconstruction error
        over the log grid. A positive float is used directly and skips selection.
    n_grid : int, default=21
        Number of candidate bandwidths. Ignored when `bandwidth` is a float.
    trim_frac : float, default=0.0
        Fraction of worst leave-one-out residuals dropped before averaging. ``0.0`` is
        the published setting; ``0.10`` moves 39_vertebral to 0.5345 against a
        published 0.550.
    max_reference : int or None, default=2000
        Cap on the kernel anchor set. ``None`` uses every training row. This is the
        cap the paper's protocol applies through :func:`karkcm.data.subsample`. Rows
        beyond the cap are not used at all: a fit on 500k rows is a fit on 2000 of
        them, silently and by design.
    contamination : float, default=0.1
        Target false-alarm rate on data from the training distribution. Sets
        `offset_`, and therefore affects only `decision_function`, `predict` and
        `fit_predict` - at ``novelty=False`` it is the sole knob on the threshold
        `fit_predict` applies.
        ``"auto"`` is NOT accepted, unlike ``IsolationForest`` (where it is the
        default) and ``LocalOutlierFactor``: those normalise their score by
        construction and can therefore pick a fixed offset, while a kernel residual
        norm has no such reference point, so the quantile has to be named.
    chunk_size : int or None, default=4096
        Query rows scored per block. A memory knob with no statistical meaning: it
        appears in ``get_params()``, so a grid search will happily search a parameter
        that cannot change the reported metric. It does reorder the BLAS reduction, so
        the scores themselves move: over the 47 ADBench datasets at seed 0 the default
        differs from ``None`` on the two whose test set just exceeds one block, by at
        most 8e-15 (24_mnist) and 2e-15 (30_satellite), with the AUROC identical to
        every digit float64 stores. ``None`` scores in one block, which is bitwise
        identical to :meth:`karkcm.KCMAnchor.score_np`.
    assume_scaled : bool, default=False
        ``True`` silences :class:`NotStandardizedWarning`. A warnings-only flag: it
        changes no arithmetic, KCM never rescales `X`, and nothing checks the
        assertion it records, so ``assume_scaled=True`` in a ``get_params()`` dump is
        not evidence that the data was scaled. Silencing the warning on data that is
        scaled but not CENTERED is a measured way to invert the detector; see
        Warnings.
    novelty : bool, default=True
        Which half of the API is exposed, following ``LocalOutlierFactor``. ``True``
        exposes :meth:`score_samples`, :meth:`decision_function` and :meth:`predict`
        for rows the estimator was NOT fitted on, which is the protocol of the paper,
        and hides :meth:`fit_predict`. ``False`` exposes :meth:`fit_predict` for
        labelling the rows you fitted on, by their leave-one-out residuals, and hides
        the other three. The default is inverted with respect to
        ``LocalOutlierFactor``, whose default is ``novelty=False``.
    random_state : int, RandomState instance or None, default=None
        Seeds the anchor subsample. Consumed only when
        ``n_samples > max_reference``, so a smaller dataset is deterministic anyway.
        Above the cap, ``None`` draws fresh entropy and two fits of the same `X`
        differ: measured on 11_donors (291308 training rows) over six processes,
        `bandwidth_` ranged over 0.462 to 0.733 and the test AUROC over 0.9882 to
        0.9910. Pass an integer for a reproducible fit, and the SPLIT seed to
        reproduce a published row.

    Attributes
    ----------
    bandwidth_ : float
        The selected bandwidth.
    bandwidth_grid_ : ndarray of shape (n_grid,)
        ``base * logspace(-1, 1, n_grid)``. Shape ``(0,)`` when `bandwidth` is a float.
    loo_curve_ : ndarray of shape (n_grid,)
        The leave-one-out objective per candidate. Shape ``(0,)`` when `bandwidth` is
        a float.
    bandwidth_at_grid_edge_ : bool
        ``True`` when the minimum of `loo_curve_` sits at an endpoint, i.e. the grid
        did not bracket the optimum and `bandwidth_` is an artefact of the grid.
        ``False`` when `bandwidth` is a float.
    reference_ : ndarray of shape (n_reference_, n_features_in_)
        The kernel anchor set, an owned C-contiguous float64 copy.
    reference_indices_ : ndarray of shape (n_reference_,)
        Row indices into the `X` passed to :meth:`fit`, in subsample order.
    n_reference_ : int
        ``len(reference_)``.
    loo_score_samples_ : ndarray of shape (n_reference_,)
        Leave-one-out ``score_samples`` of the anchors, all <= 0. The analogue of
        ``LocalOutlierFactor.negative_outlier_factor_``.
    offset_ : float
        ``percentile(loo_score_samples_, 100 * contamination)``, always <= 0.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of str
        Feature names seen during :meth:`fit`, when `X` carries column names.

    Warnings
    --------
    **Do not score rows that were in the `X` passed to :meth:`fit`. In sample the
    ranking is inverted, not merely optimistic.** A training row sits inside its own
    reference set, so ``K(z, z) = 1`` dominates its own Nadaraya-Watson
    reconstruction, and the more isolated the row the more completely it reconstructs
    itself. The planted anomalies therefore get the SMALLEST residuals. Measured on
    400 standard normals plus 8 points at radius 20, over three draws (seeds 11, 3
    and 7): ``roc_auc_score(y, m.anomaly_score(X))`` is 0.125, 0.062 and 0.130 where
    ``IsolationForest`` scores 1.000, and the planted points rank among the most
    NORMAL rows in the set.

    The `novelty` parameter narrows that mistake rather than merely documenting it.
    At the default ``novelty=True`` there is no :meth:`fit_predict` to call, and at
    ``novelty=False`` there is no :meth:`predict`, :meth:`score_samples` or
    :meth:`decision_function`. :meth:`fit_predict` scores the anchors by their
    leave-one-out residuals, which are not self-reconstructed, and recovers AUROC
    0.844, 0.878 and 0.995 on those same three draws, in that order. This is the
    ``LocalOutlierFactor`` design and it is adopted for the same reason.

    **`novelty` guards the MODE, not the rows.** It removes the `fit_predict`-shaped
    route into the in-sample ranking and makes a `fit_predict` / `fit(X).predict(X)`
    disagreement impossible on one unmodified object, because no single instance
    exposes both halves at once. It does NOT stop you passing the training rows back
    into :meth:`predict` at ``novelty=True``: on the draw above,
    ``KCM(contamination=0.02).fit(X).predict(X)`` still catches 0 of the 8 planted
    points at an in-sample AUROC of 0.1247, exactly as it did before the parameter
    existed. Nothing in the estimator knows which rows it was fitted on.

    :meth:`anomaly_score`, :meth:`reconstruct` and :meth:`kernel_mass` are diagnostics
    on arbitrary rows and are available in both modes, so they remain callable on the
    training rows and remain inverted there. The guard is on the label-producing API,
    not on the arithmetic.

    **A query far from every anchor scores its distance from the coordinate ORIGIN,
    so an uncentered `X` inverts the detector.** The Gaussian underflows, `NW(z)`
    collapses to zero, and the score becomes ``||z||``. When the data is centred that
    is a large number and the ranking survives, which is why 46 of the 48 underflowed
    24_mnist rows are true anomalies. When the data is not centred it is the opposite:
    fitting on a blob at ``(50, 50)`` with ``assume_scaled=True`` and querying 8 points
    at the origin gives ``anomaly_score == 0.0`` exactly, ``predict == +1`` on all
    eight, and AUROC 0.0. The same data through :func:`make_kcm_pipeline` gives AUROC
    1.0. Centering is what makes the fallback conservative; `assume_scaled=True`
    silences the message that would have told you, without fixing anything.

    Notes
    -----
    Sign conventions, in one chain::

        anomaly_score(X)     = ||z - NW(z)||   >= 0   HIGHER = MORE ANOMALOUS
        score_samples(X)     = -anomaly_score(X)  <= 0   HIGHER = MORE NORMAL
        offset_              = percentile(loo_score_samples_, 100 * contamination)
        decision_function(X) = score_samples(X) - offset_   NEGATIVE = OUTLIER
        predict(X)           = -1 outlier, +1 inlier

    ``roc_auc_score(y, m.decision_function(X))`` therefore returns ``1 - AUROC`` when
    ``y == 1`` means anomaly: 0.2957 instead of 0.7043 on 29_Pima at seed 0. Use
    :meth:`anomaly_score` or the :func:`anomaly_auroc` scorer.

    `offset_` is calibrated on leave-one-out anchor residuals, not on in-sample
    scores, because a training row sits inside its own reference set and its
    in-sample residual is near zero. At ``contamination=0.1``, over all 47 ADBench
    datasets at seed 0, the in-sample threshold flags 0.108 to 1.000 of clean held-out
    rows (median 0.822; 0.873 on 6_cardio, 1.000 on 24_mnist) where the leave-one-out
    threshold flags 0.062 to 0.225 (median 0.100). The precedent is
    ``LocalOutlierFactor``, which thresholds its own leave-one-out scores for the same
    reason. The price is one documented expected failure of ``check_estimator``,
    ``check_outliers_train``, whose unguarded
    ``assert_array_equal(unique(fit(X).predict(X)), [-1, 1])`` requires both labels on
    the TRAINING set. On the check's 300-row blob the in-sample scores span
    ``[-0.223, -0.0002]`` while the leave-one-out scores span ``[-1.115, -0.0048]``,
    so `offset_` at the check's ``contamination=0.1`` is -0.2649, below the whole
    in-sample range, and every row comes back an inlier; the crossing is between
    ``contamination=0.14`` and ``0.15``. ``LocalOutlierFactor`` escapes the same check
    only because its ``contamination="auto"`` offset is the scale-free constant -1.5
    rather than a percentile. Making this check pass would mean calibrating `offset_`
    on in-sample scores, which is the tighter-threshold failure measured in the
    paragraph above. ``check_estimator(KCM(novelty=False))`` passes with nothing
    excused.

    ``random_state=RandomState(0)`` and ``random_state=0`` give different anchor sets.
    This is contrary to the usual scikit-learn expectation and is forced by the
    published subsample, which uses ``numpy.random.default_rng``;
    ``KCM(random_state=s)`` reproduces ``karkcm.data.subsample(X, s)`` only for an
    integer `s` equal to the split seed.

    The Gaussian underflows in the far field and is not stabilised, so a query far
    from every anchor has ``kernel_mass(X) == 0`` exactly, ``NW(z) == 0``, and a score
    of exactly ``||z||``, its distance from the coordinate origin. On 24_mnist that is
    48 of 4152 test rows. Use :meth:`kernel_mass` to detect it and put a scaler in
    front, for instance with :func:`make_kcm_pipeline`. See Warnings for what this
    costs on data that is not centred.

    The bandwidth grid is centred on the median pairwise distance of at most 500
    reference rows drawn BY POSITION, so `bandwidth_` depends on the row order of `X`
    whenever ``n_reference_ > 500``. Measured over 20 permutations of the paper's own
    corpus, `bandwidth_` moves by up to 4.6 percent (2_annthyroid) and the test AUROC
    by up to 0.0024 (23_mammography). Fixing `random_state` is not enough for a
    bit-identical fit; the row order has to be fixed too.

    `X` is validated to float64, so float32 input is upcast: `reference_` and every
    returned array are float64, and a float32 caller pays twice the memory of their
    input. The frozen path behaves the same way, because ``scipy.spatial.distance``
    promotes to double. `X` must also be dense and finite: ``__sklearn_tags__``
    declares ``input_tags.sparse = False`` and ``input_tags.allow_nan = False``, so a
    sparse matrix or a ``NaN`` is rejected by validation rather than scored.

    Fitting costs ``O(n_grid * n_reference_ ** 2)`` time and
    ``O(n_reference_ ** 2)`` memory, so `max_reference` is load-bearing: a caller with
    500k rows is fitting on 2000 of them.

    References
    ----------
    Jose Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel and Bich-Lien Doan.
    Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via
    Kernel-Anchored Locality Regularization. CIKM 2026.
    doi:10.1145/3799682.3841143

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.metrics import roc_auc_score
    >>> from sklearn.model_selection import train_test_split
    >>> from karkcm import make_kcm_pipeline
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> normal, anomaly = X[y == 1], X[y == 0]
    >>> clean, held = train_test_split(normal, test_size=0.5, random_state=0)
    >>> detector = make_kcm_pipeline().fit(clean)
    >>> X_test = np.vstack([held, anomaly])
    >>> y_test = np.r_[np.zeros(len(held)), np.ones(len(anomaly))]
    >>> round(float(roc_auc_score(y_test, -detector.score_samples(X_test))), 4)
    0.953
    """

    _parameter_constraints = {
        "bandwidth": [StrOptions({"loo"}), Interval(Real, 0, None, closed="neither")],
        "n_grid": [Interval(Integral, 2, None, closed="left")],
        "trim_frac": [Interval(Real, 0, 1, closed="left")],
        "max_reference": [Interval(Integral, 1, None, closed="left"), None],
        "contamination": [Interval(Real, 0, 0.5, closed="right")],
        "chunk_size": [Interval(Integral, 1, None, closed="left"), None],
        "assume_scaled": ["boolean"],
        "novelty": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        bandwidth="loo",
        n_grid=21,
        trim_frac=0.0,
        max_reference=2000,
        contamination=0.1,
        chunk_size=4096,
        assume_scaled=False,
        novelty=True,
        random_state=None,
    ):
        self.bandwidth = bandwidth
        self.n_grid = n_grid
        self.trim_frac = trim_frac
        self.max_reference = max_reference
        self.contamination = contamination
        self.chunk_size = chunk_size
        self.assume_scaled = assume_scaled
        self.novelty = novelty
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags

    def _check_novelty_fit_predict(self):
        if self.novelty:
            raise AttributeError(
                "fit_predict is not available when novelty=True. Use novelty=False "
                "if you want to label the rows you fit on."
            )
        return True

    def _check_novelty_scoring(self):
        if not self.novelty:
            raise AttributeError(
                "score_samples, decision_function and predict are not available "
                "when novelty=False, because a row that was fitted on sits inside "
                "its own reference set and scores as maximally normal. Use "
                "novelty=True to score rows the estimator was not fitted on, or "
                "fit_predict to label the rows you fit on."
            )
        return True

    def _rng(self):
        rs = self.random_state
        if isinstance(rs, np.random.RandomState):
            rs = int(rs.randint(np.iinfo(np.int32).max))
        return np.random.default_rng(rs)

    def _warn_unscaled(self, X):
        scale = X.std(axis=0)
        pos = scale[scale > 0]
        if not pos.size:
            return
        ratio = float(pos.max() / pos.min())
        centre = float(np.max(np.abs(X.mean(axis=0)[scale > 0]) / pos))
        if ratio <= 10.0 and centre <= 3.0:
            return
        warnings.warn(
            "KCM uses one isotropic RBF bandwidth for all features; this X is not "
            f"standardized (largest/smallest feature sd = {ratio:.1f}, largest "
            f"|mean|/sd = {centre:.1f}). Wrap it as karkcm.make_kcm_pipeline(). "
            "Passing assume_scaled=True silences this message without changing the "
            "data, which is only correct if X is already centered as well as scaled: "
            "on an uncentered X the kernel-underflow fallback score is the distance "
            "from the coordinate origin and can invert the ranking. Standardizing "
            "changes AUROC by -0.001 in the median over 36 ADBench datasets, by up "
            "to +0.211 on 2_annthyroid and -0.196 on 44_Wilt, and raw features win "
            "on 20 of the 36.",
            NotStandardizedWarning,
            stacklevel=3,
        )

    def _loo_residual(self):
        R = self.reference_
        K = rbf_gram(R, R, self.bandwidth_)
        np.fill_diagonal(K, 0.0)
        wsum = K.sum(axis=1, keepdims=True) + 1e-12
        return np.linalg.norm(R - (K @ R) / wsum, axis=1)

    def _validate_query(self, X):
        check_is_fitted(self)
        return validate_data(self, X, dtype=np.float64, reset=False)

    def _blocks(self, X):
        step = len(X) if self.chunk_size is None else self.chunk_size
        for start in range(0, len(X), step):
            yield start, X[start : start + step]

    def fit(self, X, y=None):
        """Select the bandwidth and store the kernel anchor set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, assumed to be from the normal class. At least two rows are
            required: at ``n_samples == 1`` the leave-one-out objective has no
            neighbours and every candidate bandwidth ties.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : KCM
            The fitted estimator.
        """
        self._validate_params()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        if not self.assume_scaled:
            self._warn_unscaled(X)

        n = len(X)
        cap = self.max_reference
        if cap is None or n <= cap:
            idx = np.arange(n)
        else:
            idx = self._rng().permutation(n)[:cap]
        self.reference_indices_ = idx
        self.reference_ = np.ascontiguousarray(X[idx], dtype=np.float64)
        self.n_reference_ = int(len(self.reference_))

        if self.bandwidth == "loo":
            grid = build_h_grid(self.reference_, n_points=self.n_grid)
            h, curve = trimmed_loo_bandwidth(
                self.reference_, grid, trim_frac=self.trim_frac
            )
            self.bandwidth_grid_ = grid
            self.loo_curve_ = curve
            self.bandwidth_ = float(h)
            k = int(np.argmin(curve))
            self.bandwidth_at_grid_edge_ = bool(k == 0 or k == len(grid) - 1)
        else:
            self.bandwidth_grid_ = np.empty(0, dtype=np.float64)
            self.loo_curve_ = np.empty(0, dtype=np.float64)
            self.bandwidth_ = float(self.bandwidth)
            self.bandwidth_at_grid_edge_ = False

        self.loo_score_samples_ = -self._loo_residual()
        self.offset_ = float(
            np.percentile(self.loo_score_samples_, 100.0 * self.contamination)
        )
        return self

    def anomaly_score(self, X):
        """Return ``||z - NW(z)||``. HIGHER MEANS MORE ANOMALOUS.

        This is the number reported in the paper. It is the negation of
        :meth:`score_samples`, which follows the opposite scikit-learn convention.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residual norms, all >= 0.
        """
        X = self._validate_query(X)
        out = np.empty(len(X), dtype=np.float64)
        for start, block in self._blocks(X):
            K = rbf_gram(block, self.reference_, self.bandwidth_)
            wsum = K.sum(axis=1, keepdims=True) + 1e-12
            x_hat = (K @ self.reference_) / wsum
            out[start : start + len(block)] = np.linalg.norm(block - x_hat, axis=1)
        return out

    def reconstruct(self, X):
        """Return ``NW(X)``, what the fitted normal manifold expected each row to be.

        Not named ``transform``: an estimator carrying that name is treated as a
        transformer by ``check_estimator``, which then aborts check generation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples, n_features_in_)
            The Nadaraya-Watson reconstruction.
        """
        X = self._validate_query(X)
        out = np.empty((len(X), self.n_features_in_), dtype=np.float64)
        for start, block in self._blocks(X):
            K = rbf_gram(block, self.reference_, self.bandwidth_)
            wsum = K.sum(axis=1, keepdims=True) + 1e-12
            out[start : start + len(block)] = (K @ self.reference_) / wsum
        return out

    def kernel_mass(self, X):
        """Return ``sum_j K(z, x_j)``, the total kernel weight each row receives.

        Where this is exactly ``0.0`` the Gaussian has underflowed, ``NW(z)`` is the
        coordinate origin, and :meth:`anomaly_score` returns exactly ``||z||``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples,)
            The kernel masses.
        """
        X = self._validate_query(X)
        out = np.empty(len(X), dtype=np.float64)
        for start, block in self._blocks(X):
            K = rbf_gram(block, self.reference_, self.bandwidth_)
            out[start : start + len(block)] = K.sum(axis=1)
        return out

    @available_if(_check_novelty_scoring)
    def score_samples(self, X):
        """Return ``-anomaly_score(X)``. HIGHER MEANS MORE NORMAL, all values <= 0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples,)
            The negated residual norms.
        """
        return -self.anomaly_score(X)

    @available_if(_check_novelty_scoring)
    def decision_function(self, X):
        """Return ``score_samples(X) - offset_``. NEGATIVE MEANS OUTLIER.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples,)
            The shifted scores.
        """
        return self.score_samples(X) - self.offset_

    @available_if(_check_novelty_scoring)
    def predict(self, X):
        """Return ``+1`` for inliers and ``-1`` for outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_samples,)
            Integer labels, ``-1`` where :meth:`decision_function` is negative.
        """
        return np.where(self.decision_function(X) < 0, -1, 1).astype(int)

    @available_if(_check_novelty_fit_predict)
    def fit_predict(self, X, y=None):
        """Fit on `X` and label those same rows. ``+1`` inlier, ``-1`` outlier.

        Transductive, and not comparable with ``fit(X).predict(X)`` on one unmodified
        object, because no single `KCM` exposes both halves at once. Every anchor is
        inside its own reference set, so its in-sample residual is near zero and its
        in-sample ranking is inverted; this method scores the anchors by their
        leave-one-out residuals (`loo_score_samples_`) instead, and scores any row the
        subsample left out normally, since such a row is genuinely out of sample. The
        two therefore disagree where it matters: flipping `novelty` on this fitted
        object and calling :meth:`predict` on the same `X` recovers the inverted
        labels. The precedent is ``LocalOutlierFactor.fit_predict``, which is
        transductive for the same reason.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to fit on and label. May be contaminated.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        ndarray of shape (n_samples,)
            Integer labels.
        """
        self.fit(X)
        scores = -self.anomaly_score(X)
        scores[self.reference_indices_] = self.loo_score_samples_
        return np.where(scores - self.offset_ < 0, -1, 1).astype(int)


def _final_step_has(attr):
    def check(self):
        getattr(self.steps[-1][1], attr)
        return True

    return check


class KCMPipeline(Pipeline):
    """A :class:`sklearn.pipeline.Pipeline` that also forwards KCM's extra methods.

    ``Pipeline`` forwards only the methods scikit-learn knows about, so on a plain
    pipeline ``anomaly_score``, ``reconstruct`` and ``kernel_mass`` are unreachable
    and a caller is pushed back to writing ``-pipe.score_samples(X)``, the bare minus
    sign :meth:`karkcm.KCM.anomaly_score` exists to remove. This subclass adds the
    three, each available only when the final step has it. It is what
    :func:`make_kcm_pipeline` returns; it is otherwise an ordinary ``Pipeline`` and
    clones, pickles and grid-searches like one.

    Examples
    --------
    >>> import numpy as np
    >>> from karkcm import make_kcm_pipeline
    >>> scale = np.array([1.0, 1000.0, 1.0])
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3)) * scale
    >>> query = rng.normal(size=(2, 3)) * scale
    >>> detector = make_kcm_pipeline().fit(X)
    >>> detector.kernel_mass(query).round(3)
    array([1.07 , 0.016])
    """

    def _to_final(self, X):
        check_is_fitted(self)
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return Xt

    @available_if(_final_step_has("anomaly_score"))
    def anomaly_score(self, X):
        """Transform `X` through every step, then return the final step's score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows, in the units the first step expects.

        Returns
        -------
        ndarray of shape (n_samples,)
            ``||z - NW(z)||``. HIGHER MEANS MORE ANOMALOUS.
        """
        return self.steps[-1][1].anomaly_score(self._to_final(X))

    @available_if(_final_step_has("reconstruct"))
    def reconstruct(self, X):
        """Transform `X` through every step, then return the final step's ``NW(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows, in the units the first step expects.

        Returns
        -------
        ndarray of shape (n_samples, n_features_in_)
            The reconstruction, in the SCALED units the final step works in.
        """
        return self.steps[-1][1].reconstruct(self._to_final(X))

    @available_if(_final_step_has("kernel_mass"))
    def kernel_mass(self, X):
        """Transform `X` through every step, then return the final step's kernel mass.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query rows, in the units the first step expects.

        Returns
        -------
        ndarray of shape (n_samples,)
            ``sum_j K(z, x_j)``. Exactly ``0.0`` means the Gaussian underflowed.
        """
        return self.steps[-1][1].kernel_mass(self._to_final(X))


def make_kcm_pipeline(**kcm_kwargs) -> "KCMPipeline":
    """Return ``StandardScaler`` followed by ``KCM(assume_scaled=True)``.

    The steps are named ``"standardscaler"`` and ``"kcm"``, so a grid searches
    ``kcm__bandwidth``. The returned object is a :class:`KCMPipeline`, which forwards
    ``anomaly_score``, ``reconstruct`` and ``kernel_mass`` to the final step on top of
    everything an ordinary ``Pipeline`` forwards.

    Centering matters as much as scaling here: see the second paragraph of
    :class:`KCM`'s Warnings for what an uncentered `X` does to the score.

    Parameters
    ----------
    **kcm_kwargs : dict
        Keyword arguments forwarded to :class:`KCM`. Passing ``assume_scaled``
        explicitly raises :class:`TypeError` on the duplicate keyword.

    Returns
    -------
    KCMPipeline
        The two-step pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from karkcm import make_kcm_pipeline
    >>> scale = np.array([1.0, 1000.0, 1.0, 1.0])
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 4)) * scale
    >>> query = rng.normal(size=(3, 4)) * scale
    >>> detector = make_kcm_pipeline().fit(X)
    >>> detector.predict(query)
    array([1, 1, 1])
    >>> detector.anomaly_score(query).round(4)
    array([0.0843, 0.1995, 0.1381])
    """
    return KCMPipeline(
        [
            ("standardscaler", StandardScaler()),
            ("kcm", KCM(assume_scaled=True, **kcm_kwargs)),
        ]
    )


def anomaly_auroc(estimator, X, y) -> float:
    """Scorer callable returning AUROC with ``y == 1`` meaning anomaly.

    ``scoring="roc_auc"`` returns ``1 - AUROC`` for an outlier detector, silently,
    because scikit-learn scores `decision_function` in the normal-is-high direction.
    Pass this instead.

    A plain callable, so it has no ``set_score_request`` and cannot receive routed
    metadata: under ``set_config(enable_metadata_routing=True)``,
    ``GridSearchCV(scoring=anomaly_auroc).fit(X, y, sample_weight=w)`` raises
    ``TypeError``. Wrap the metric with :func:`sklearn.metrics.make_scorer` if you
    need routing. Ordinary routing-enabled use without extra metadata works.

    Parameters
    ----------
    estimator : KCM
        A fitted estimator exposing ``score_samples``.
    X : array-like of shape (n_samples, n_features)
        Query rows.
    y : array-like of shape (n_samples,)
        Binary labels, ``1`` for anomaly.

    Returns
    -------
    float
        The area under the ROC curve.
    """
    return float(roc_auc_score(y, -estimator.score_samples(X)))


def anomaly_auprc(estimator, X, y) -> float:
    """Scorer callable returning average precision with ``y == 1`` meaning anomaly.

    A plain callable, so it has no ``set_score_request``; see :func:`anomaly_auroc`.

    Parameters
    ----------
    estimator : KCM
        A fitted estimator exposing ``score_samples``.
    X : array-like of shape (n_samples, n_features)
        Query rows.
    y : array-like of shape (n_samples,)
        Binary labels, ``1`` for anomaly.

    Returns
    -------
    float
        The average precision.
    """
    return float(average_precision_score(y, -estimator.score_samples(X)))
