"""Kernel Contraction Matching: the frozen arithmetic of the CIKM 2026 paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    import torch


def silverman_scale(N: int, D: int) -> float:
    """Silverman's rule-of-thumb bandwidth scale for `N` points in `D` dimensions.

    Parameters
    ----------
    N : int
        Number of points.
    D : int
        Number of features.

    Returns
    -------
    float
        ``(4 / (D + 2)) ** (1 / (D + 4)) * N ** (-1 / (D + 4))``.
    """
    return float((4.0 / (D + 2.0)) ** (1.0 / (D + 4.0)) * N ** (-1.0 / (D + 4.0)))


def build_h_grid(X: np.ndarray, n_points: int = 21) -> np.ndarray:
    """Log-spaced grid of candidate bandwidths, centred on the data scale.

    The centre is ``sqrt(silverman_scale(N, D) * median_pairwise_distance)`` and the
    grid spans two decades around it. The median is taken over at most 500 rows drawn
    by position with a fixed generator, so the grid depends on the ORDER of `X`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Reference rows. Must be a numpy array; a DataFrame raises ``KeyError`` on the
        positional subsample when ``len(X) > 500``.
    n_points : int, default=21
        Number of grid points.

    Returns
    -------
    ndarray of shape (n_points,)
        ``base * numpy.logspace(-1, 1, n_points)``.
    """
    N, D = X.shape
    sig0 = max(silverman_scale(N, D), 1e-3)
    sub = (
        X
        if len(X) <= 500
        else X[np.random.default_rng(0).choice(len(X), 500, replace=False)]
    )
    d = cdist(sub, sub)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    base = float(np.sqrt(sig0 * med))
    return base * np.logspace(-1.0, 1.0, n_points)


def rbf_gram(X: np.ndarray, Y: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel matrix ``exp(-||x - y||^2 / (2 h^2))``.

    Materialises a dense ``len(X) x len(Y)`` float64 array; see
    :class:`karkcm.KCM` and its ``chunk_size`` parameter for bounded-memory scoring.
    No max-subtraction stabilisation is applied, so the kernel underflows to exactly
    zero in the far field.

    Parameters
    ----------
    X : ndarray of shape (n_queries, n_features)
        Query rows.
    Y : ndarray of shape (n_references, n_features)
        Reference rows.
    h : float
        Bandwidth.

    Returns
    -------
    ndarray of shape (n_queries, n_references)
        The kernel matrix.
    """
    d2 = cdist(X, Y, "sqeuclidean")
    return np.exp(-d2 / (2.0 * h * h))


def trimmed_loo_bandwidth(X: np.ndarray, h_grid: np.ndarray, trim_frac: float = 0.0):
    """Select a bandwidth by trimmed leave-one-out reconstruction error.

    For each candidate the kernel diagonal is zeroed, every row is reconstructed from
    the others by Nadaraya-Watson, and the mean of the smallest
    ``round((1 - trim_frac) * n_samples)`` squared errors is the objective.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Reference rows.
    h_grid : ndarray of shape (n_grid,)
        Candidate bandwidths.
    trim_frac : float, default=0.0
        Fraction of worst residuals dropped before averaging. ``0.0`` is the published
        setting; ``0.10`` moves 39_vertebral to 0.5345 against a published 0.550.

    Returns
    -------
    h : float
        The minimising candidate.
    losses : ndarray of shape (n_grid,)
        The objective at every candidate.
    """
    d2 = cdist(X, X, "sqeuclidean")
    keep = max(1, int(round((1.0 - trim_frac) * len(X))))
    losses = np.empty_like(h_grid, dtype=np.float64)
    for i, h in enumerate(h_grid):
        K = np.exp(-d2 / (2.0 * h * h))
        np.fill_diagonal(K, 0.0)
        wsum = K.sum(axis=1, keepdims=True) + 1e-12
        x_hat = (K @ X) / wsum
        per_pt = np.sum((X - x_hat) ** 2, axis=1)
        losses[i] = float(per_pt[np.argsort(per_pt)[:keep]].mean())
    return float(h_grid[int(np.argmin(losses))]), losses


def nw_reconstruction(X_train: np.ndarray, Z: np.ndarray, h: float) -> np.ndarray:
    """Nadaraya-Watson reconstruction of `Z` from the reference rows `X_train`.

    Parameters
    ----------
    X_train : ndarray of shape (n_references, n_features)
        Reference rows.
    Z : ndarray of shape (n_queries, n_features)
        Query rows.
    h : float
        Bandwidth.

    Returns
    -------
    ndarray of shape (n_queries, n_features)
        ``K @ X_train / (K.sum(axis=1) + 1e-12)``. Rows whose kernel mass underflows
        to zero reconstruct to the coordinate origin.
    """
    K = rbf_gram(Z, X_train, h)
    wsum = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ X_train) / wsum


def nw_residual_score(X_train: np.ndarray, Z: np.ndarray, h: float) -> np.ndarray:
    """KCM anomaly score ``||z - NW(z)||``. Higher means more anomalous.

    Parameters
    ----------
    X_train : ndarray of shape (n_references, n_features)
        Reference rows.
    Z : ndarray of shape (n_queries, n_features)
        Query rows.
    h : float
        Bandwidth.

    Returns
    -------
    ndarray of shape (n_queries,)
        The residual norms.
    """
    return np.linalg.norm(Z - nw_reconstruction(X_train, Z, h), axis=1)


@dataclass
class KCMAnchor:
    """Frozen reproduction object for the paper's KCM detector.

    This is the exact code path behind ``karkcm/experiments/*`` and the
    committed ``results/`` tree, and its arithmetic must not change. It applies NO cap
    on the reference set; callers pass the output of :func:`karkcm.data.subsample`.
    :class:`karkcm.KCM` is the estimator-shaped entry point and applies the cap itself.

    Parameters
    ----------
    X_train : ndarray of shape (n_references, n_features)
        Reference rows, cast to float64.
    h : float
        Selected bandwidth.
    h_grid : ndarray of shape (n_grid,)
        The candidate grid.
    loo_curve : ndarray of shape (n_grid,)
        The leave-one-out objective at every candidate.

    References
    ----------
    Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via
    Kernel-Anchored Locality Regularization, CIKM 2026,
    doi:10.1145/3799682.3841143.
    """

    X_train: np.ndarray
    h: float
    h_grid: np.ndarray
    loo_curve: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray, n_grid: int = 21, trim_frac: float = 0.0) -> "KCMAnchor":
        """Build the bandwidth grid and select `h` by trimmed leave-one-out error.

        Parameters
        ----------
        X_train : ndarray of shape (n_references, n_features)
            Reference rows, already subsampled and scaled by the caller.
        n_grid : int, default=21
            Number of candidate bandwidths.
        trim_frac : float, default=0.0
            Trimming fraction; ``0.0`` is the published setting.

        Returns
        -------
        KCMAnchor
            The fitted anchor.
        """
        h_grid = build_h_grid(X_train, n_points=n_grid)
        h_star, loo = trimmed_loo_bandwidth(X_train, h_grid, trim_frac=trim_frac)
        return cls(
            X_train=X_train.astype(np.float64), h=h_star, h_grid=h_grid, loo_curve=loo
        )

    def reconstruct_np(self, Z: np.ndarray) -> np.ndarray:
        """Nadaraya-Watson reconstruction of `Z` at the selected bandwidth.

        Parameters
        ----------
        Z : ndarray of shape (n_queries, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_queries, n_features)
            The reconstruction.
        """
        return nw_reconstruction(self.X_train, Z, self.h)

    def score_np(self, Z: np.ndarray) -> np.ndarray:
        """Anomaly score of `Z`. Higher means more anomalous.

        Parameters
        ----------
        Z : ndarray of shape (n_queries, n_features)
            Query rows.

        Returns
        -------
        ndarray of shape (n_queries,)
            ``||z - NW(z)||``.
        """
        return nw_residual_score(self.X_train, Z, self.h)

    def torch_target(self, Z: "torch.Tensor") -> "torch.Tensor":
        """Negated reconstruction of `Z` as a torch tensor, for the KAR training loop.

        Importing this module does not import torch; the import happens here, on first
        call.

        Parameters
        ----------
        Z : torch.Tensor of shape (n_queries, n_features)
            Query rows.

        Returns
        -------
        torch.Tensor of shape (n_queries, n_features)
            ``-NW(Z)``, so that ``Z + torch_target(Z)`` is the KCM residual.
        """
        import torch

        X = torch.as_tensor(self.X_train, dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            d2 = torch.cdist(Z, X, p=2.0).pow_(2)
            K = torch.exp(-d2 / (2.0 * self.h * self.h))
            wsum = K.sum(dim=1, keepdim=True) + 1e-12
            x_hat = (K @ X) / wsum
        return -x_hat
