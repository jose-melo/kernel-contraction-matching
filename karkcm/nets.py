import math
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ntk_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def mlp_stack(d_in: int, d_out: int, hidden: int, depth: int, bias: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(last, hidden, bias=bias), nn.ReLU(inplace=True)]
        last = hidden
    layers += [nn.Linear(last, d_out, bias=bias)]
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half, 1)
        )
        ang = t * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, depth: int = 3,
                 bias: bool = True):
        super().__init__()
        self.net = mlp_stack(d_in, d_out, hidden, depth, bias=bias)
        ntk_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeConditionedMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, t_embed: int = 32,
                 depth: int = 3):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbedding(t_embed)
        self.net = mlp_stack(d_in + t_embed, d_out, hidden, depth)
        ntk_init(self)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.t_emb(t)], dim=-1))


class Autoencoder(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, latent: int = 32, depth: int = 2):
        super().__init__()
        self.encoder = mlp_stack(d_in, latent, hidden, depth)
        self.decoder = mlp_stack(latent, d_in, hidden, depth)
        ntk_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
