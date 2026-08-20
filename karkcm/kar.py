"""Kernel-Anchored Regularization: a bounded learned correction on the KCM residual."""

import time

import numpy as np
import torch

from .data import load_split, subsample
from .kcm import KCMAnchor
from .metrics import auprc, auroc
from .nets import MLP, set_seed


def bounded_correction(u: torch.Tensor, rho: float) -> torch.Tensor:
    """Squash a network output into a ball of radius `rho`.

    ``g = rho * u / (1 + ||u||)``, so ``||g|| < rho`` for every `u` and the learned
    correction can never move a score further than the radius allows. This is what
    bounds KAR's departure from the closed-form KCM residual.

    Parameters
    ----------
    u : torch.Tensor of shape (n_samples, n_features)
        Raw network output.
    rho : float
        Correction radius. ``rho <= 0`` returns exact zeros, which reduces KAR to KCM
        and switches the optimizer step off in :func:`run_kar`.

    Returns
    -------
    torch.Tensor of shape (n_samples, n_features)
        The bounded correction.
    """
    if rho <= 0.0:
        return torch.zeros_like(u)
    norm = torch.linalg.norm(u, dim=-1, keepdim=True)
    return rho * u / (1.0 + norm)


def kar_score(net: MLP, anchor: KCMAnchor, x: torch.Tensor, rho: float) -> torch.Tensor:
    """KAR anomaly score ``||r_kcm + g||``. Higher means more anomalous.

    `r_kcm` is the KCM residual ``x - NW(x)`` taken from `anchor`, and `g` is the
    bounded correction of the network output. At ``rho = 0`` this is exactly the KCM
    score.

    Parameters
    ----------
    net : karkcm.nets.MLP
        The correction network.
    anchor : karkcm.KCMAnchor
        Fitted anchor supplying ``-NW(x)`` through
        :meth:`karkcm.KCMAnchor.torch_target`.
    x : torch.Tensor of shape (n_samples, n_features)
        Query rows, on the same device and dtype as `net`.
    rho : float
        Correction radius.

    Returns
    -------
    torch.Tensor of shape (n_samples,)
        The scores.
    """
    with torch.no_grad():
        r_kcm = x + anchor.torch_target(x)
        g = bounded_correction(net(x), rho)
    return torch.linalg.norm(r_kcm + g, dim=-1)


def run_kar(
    dataset: str,
    seed: int = 0,
    rho: float = 0.5,
    epochs: int = 100,
    batch: int = 256,
    hidden: int = 256,
    depth: int = 3,
    lr: float = 1e-3,
    h_grid_n: int = 21,
    max_ref: int = 0,
    device: str = "cpu",
) -> dict:
    """Run one KAR experiment end to end and return its record.

    Loads the semi-supervised split, fits a :class:`karkcm.KCMAnchor` on the training
    rows, then trains an MLP for `epochs` epochs on the bounded-correction objective
    ``mean_i sum_d (r_kcm + g)^2``, scoring the test set after every epoch. This is
    the function behind ``karkcm/experiments/kar_ablation.py``, and
    it needs PyTorch; KCM itself does not.

    Parameters
    ----------
    dataset : str
        ADBench dataset name.
    seed : int, default=0
        Split seed, also the network seed and the :func:`karkcm.data.subsample` seed.
    rho : float, default=0.5
        Correction radius, used unchanged on every dataset in the paper. ``0.0``
        disables the optimizer step and reduces the run to KCM.
    epochs : int, default=100
        Training epochs.
    batch : int, default=256
        Batch size, clipped to the training set size.
    hidden : int, default=256
        Hidden width of the correction MLP.
    depth : int, default=3
        Number of linear layers.
    lr : float, default=1e-3
        Adam learning rate.
    h_grid_n : int, default=21
        Number of candidate bandwidths for the anchor.
    max_ref : int, default=0
        Reference cap passed to :func:`karkcm.data.subsample`; ``0`` means no cap,
        which is the published KAR setting.
    device : str, default="cpu"
        Torch device string.

    Returns
    -------
    dict
        One flat record: ``auroc`` and ``auprc`` of the final scores, ``peak_auroc``
        and ``final_auroc`` over the epoch axis and their ``gap``, the KCM baseline
        ``kcm_auroc``, the selected ``h_kcm``, ``mean_correction_norm``, the full
        ``auroc_per_epoch`` list, ``final_loss``, and the two phase timings.

    References
    ----------
    Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via
    Kernel-Anchored Locality Regularization, CIKM 2026,
    doi:10.1145/3799682.3841143.
    """
    set_seed(seed)
    dev = torch.device(device)

    X_tr_np, X_te_np, y_te = load_split(dataset, seed)
    X_tr_np = subsample(X_tr_np, seed, max_ref)
    N, D = X_tr_np.shape
    X_tr = torch.from_numpy(X_tr_np).float().to(dev)
    X_te = torch.from_numpy(X_te_np).float().to(dev)

    t0 = time.time()
    anchor = KCMAnchor.fit(X_tr_np, n_grid=h_grid_n)
    t_phase0 = time.time() - t0

    net = MLP(d_in=D, d_out=D, hidden=hidden, depth=depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    bsize = min(batch, N)
    n_batches = max(1, (N + bsize - 1) // bsize)
    auroc_per_epoch: list[float] = []
    final_loss = float("nan")
    t1 = time.time()

    for _ in range(epochs):
        net.train()
        perm = torch.randperm(N, device=dev)
        losses = []
        for b in range(n_batches):
            x_b = X_tr[perm[b * bsize : (b + 1) * bsize]]
            with torch.no_grad():
                r_kcm = x_b + anchor.torch_target(x_b)
            g = bounded_correction(net(x_b), rho)
            loss = ((r_kcm + g) ** 2).sum(dim=-1).mean()
            if rho > 0.0:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses))

        net.eval()
        auroc_per_epoch.append(auroc(y_te, kar_score(net, anchor, X_te, rho).cpu().numpy()))
    t_phase1 = time.time() - t1

    net.eval()
    s_te = kar_score(net, anchor, X_te, rho).cpu().numpy()
    peak = float(np.nanmax(auroc_per_epoch))
    final = float(auroc_per_epoch[-1])

    with torch.no_grad():
        g_te = bounded_correction(net(X_te), rho)
        mean_g_norm = float(torch.linalg.norm(g_te, dim=-1).mean().item())

    return {
        "detector": "kar",
        "dataset": dataset,
        "seed": seed,
        "rho": rho,
        "epochs": epochs,
        "N_train": int(N),
        "D": int(D),
        "h_kcm": float(anchor.h),
        "auroc": auroc(y_te, s_te),
        "auprc": auprc(y_te, s_te),
        "peak_auroc": peak,
        "final_auroc": final,
        "gap": peak - final,
        "kcm_auroc": auroc(y_te, anchor.score_np(X_te_np)),
        "mean_correction_norm": mean_g_norm,
        "auroc_per_epoch": auroc_per_epoch,
        "final_loss": final_loss,
        "time_phase0_s": t_phase0,
        "time_phase1_s": t_phase1,
    }
