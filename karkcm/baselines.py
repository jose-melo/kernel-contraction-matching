from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import auroc
from .nets import MLP, Autoencoder, TimeConditionedMLP, set_seed


@dataclass
class BaselineConfig:
    epochs: int = 100
    batch: int = 256
    hidden: int = 256
    latent: int = 8
    lr: float = 1e-3
    mask_rate: float = 0.2
    beta: float = 0.5


DEFAULT_HIDDEN = {
    "dae": 128,
    "toll": 128,
    "deepsvdd": 256,
    "rdp": 256,
    "reflect": 256,
    "wtflow": 256,
}


def config_for(detector: str, **overrides) -> "BaselineConfig":
    fields = {"hidden": DEFAULT_HIDDEN[detector]}
    fields.update({k: v for k, v in overrides.items() if v is not None})
    return BaselineConfig(**fields)


def init_center(net: nn.Module, X: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        c = net(X).mean(dim=0)
        small = c.abs() < eps
        c[small & (c < 0)] = -eps
        c[small & (c >= 0)] = eps
    return c


def mask_corrupt(x: torch.Tensor, rate: float) -> torch.Tensor:
    keep = (torch.rand_like(x) >= rate).to(x.dtype)
    return x * keep


def reflect_corrupt(x_1: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    B, D = x_1.shape
    device, dtype = x_1.device, x_1.dtype
    p_mask = 0.1 + 0.4 * torch.rand(B, 1, device=device, dtype=dtype)
    mask = (torch.rand(B, D, device=device, dtype=dtype) < p_mask).to(dtype)
    use_gaussian = (torch.rand(B, 1, device=device, dtype=dtype) < 0.5).to(dtype)
    gaussian_R = torch.randn(B, D, device=device, dtype=dtype) * sigma
    shuffle_R = x_1[torch.randperm(B, device=device)]
    R = use_gaussian * gaussian_R + (1.0 - use_gaussian) * shuffle_R
    return (1.0 - mask) * x_1 + mask * R


def wt_normalize(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mu) / sigma


def _trajectory(cfg, step, score_fn, params, X_tr, X_te, y_te, device):
    opt = torch.optim.Adam(params, lr=cfg.lr)
    N = len(X_tr)
    bsize = min(cfg.batch, N)
    n_batches = max(1, (N + bsize - 1) // bsize)
    auroc_per_epoch = []
    for _ in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        for b in range(n_batches):
            loss = step(X_tr[perm[b * bsize : (b + 1) * bsize]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        auroc_per_epoch.append(auroc(y_te, score_fn(X_te).cpu().numpy()))
    return auroc_per_epoch


def run_dae(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    model = Autoencoder(d_in=D, hidden=cfg.hidden, latent=max(2, min(cfg.latent, D)),
                        depth=2).to(device)

    def step(x_b):
        return F.mse_loss(model(mask_corrupt(x_b, cfg.mask_rate)), x_b)

    def score(x):
        with torch.no_grad():
            x_hat = model(x)
        return torch.linalg.norm(x - x_hat, dim=-1)

    return _trajectory(cfg, step, score, model.parameters(), X_tr, X_te, y_te, device)


def run_toll(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    model = Autoencoder(d_in=D, hidden=cfg.hidden, latent=max(2, min(cfg.latent, D)),
                        depth=2).to(device)

    def step(x_b):
        z = model.encoder(x_b)
        x_hat = model.decoder(z)
        return (torch.linalg.norm(x_hat - x_b, dim=-1).mean()
                + cfg.beta * torch.linalg.norm(z, dim=-1).mean())

    def score(x):
        with torch.no_grad():
            z = model.encoder(x)
            x_hat = model.decoder(z)
        return torch.linalg.norm(x_hat - x, dim=-1) + cfg.beta * torch.linalg.norm(z, dim=-1)

    return _trajectory(cfg, step, score, model.parameters(), X_tr, X_te, y_te, device)


def run_deepsvdd(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    net = MLP(d_in=D, d_out=D, hidden=cfg.hidden, depth=3, bias=False).to(device)
    c = init_center(net, X_tr).to(device)

    def step(x_b):
        return F.mse_loss(net(x_b), c.expand(len(x_b), -1))

    def score(x):
        with torch.no_grad():
            f = net(x)
        return torch.linalg.norm(f - c, dim=-1)

    return _trajectory(cfg, step, score, net.parameters(), X_tr, X_te, y_te, device)


def run_rdp(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    phi = MLP(d_in=D, d_out=D, hidden=cfg.hidden, depth=3).to(device)
    eta = MLP(d_in=D, d_out=D, hidden=cfg.hidden, depth=3).to(device)
    for p in eta.parameters():
        p.requires_grad_(False)
    eta.eval()

    def step(x_b):
        with torch.no_grad():
            T_fix = eta(x_b)
        return F.mse_loss(phi(x_b), T_fix)

    def score(x):
        with torch.no_grad():
            r = phi(x) - eta(x)
        return torch.linalg.norm(r, dim=-1)

    return _trajectory(cfg, step, score, phi.parameters(), X_tr, X_te, y_te, device)


def run_reflect(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    sigma = X_tr.std(dim=0, keepdim=True).clamp(min=1e-3)
    model = TimeConditionedMLP(d_in=D, d_out=D, hidden=cfg.hidden, t_embed=32,
                               depth=3).to(device)

    def step(y1):
        y0 = reflect_corrupt(y1, sigma)
        t = torch.rand(len(y1), 1, device=device)
        return F.mse_loss(model((1.0 - t) * y0 + t * y1, t), y1 - y0)

    def score(x):
        t0 = torch.zeros(len(x), 1, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            v = model(x, t0)
        return torch.linalg.norm(v, dim=-1)

    return _trajectory(cfg, step, score, model.parameters(), X_tr, X_te, y_te, device)


def run_wtflow(cfg, X_tr, X_te, y_te, device, seed):
    set_seed(seed)
    D = X_tr.shape[1]
    model = TimeConditionedMLP(d_in=D, d_out=D, hidden=cfg.hidden, t_embed=32,
                               depth=3).to(device)

    def step(x0):
        x1 = wt_normalize(torch.randn_like(x0))
        t = torch.rand(len(x0), 1, device=device)
        return F.mse_loss(model((1.0 - t) * x0 + t * x1, t), x1 - x0)

    def score(x):
        t0 = torch.zeros(len(x), 1, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            v = model(x, t0)
        return torch.linalg.norm(v, dim=-1)

    return _trajectory(cfg, step, score, model.parameters(), X_tr, X_te, y_te, device)


DETECTORS = {
    "dae": run_dae,
    "toll": run_toll,
    "deepsvdd": run_deepsvdd,
    "rdp": run_rdp,
    "reflect": run_reflect,
    "wtflow": run_wtflow,
}
