import argparse
import json

import numpy as np
import torch

from ..data import KAR_DATASETS, load_split
from ..kar import bounded_correction
from ..kcm import KCMAnchor
from ..metrics import auroc
from ..nets import MLP, set_seed
from ..paths import SEC6, prepare

BACKBONES = ("tccm", "ae", "deepsvdd")


def run_one_backbone(bb, X_tr, X_te, y_te, anchor, rho, epochs, seed, device):
    set_seed(seed)
    N, D = X_tr.shape
    net = MLP(d_in=D, d_out=D, hidden=256, depth=3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    center = torch.zeros(D, device=device)

    def anchor_AB(p):
        nw_neg = anchor.torch_target(p)
        if bb == "tccm":
            return nw_neg
        if bb == "ae":
            return -nw_neg
        return p + nw_neg + center

    def target_TB(p):
        if bb == "tccm":
            return -p
        if bb == "ae":
            return p
        return center.expand(len(p), -1)

    def score(x):
        with torch.no_grad():
            m = anchor_AB(x) + bounded_correction(net(x), rho)
            if bb == "tccm":
                s = torch.linalg.norm(m + x, dim=-1)
            elif bb == "ae":
                s = torch.linalg.norm(x - m, dim=-1)
            else:
                s = torch.linalg.norm(m - center, dim=-1)
        return s.cpu().numpy()

    bsize = min(256, N)
    n_batches = max(1, (N + bsize - 1) // bsize)
    curve = []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(N, device=device)
        for b in range(n_batches):
            x_b = X_tr[perm[b * bsize : (b + 1) * bsize]]
            m = anchor_AB(x_b) + bounded_correction(net(x_b), rho)
            loss = ((m - target_TB(x_b)) ** 2).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        curve.append(auroc(y_te, score(X_te)))
    return curve


def run(dataset: str, seed: int, rho: float, epochs: int, device: str) -> dict:
    dev = torch.device(device)
    X_tr_np, X_te_np, y_te = load_split(dataset, seed)
    X_tr = torch.from_numpy(X_tr_np).float().to(dev)
    X_te = torch.from_numpy(X_te_np).float().to(dev)
    anchor = KCMAnchor.fit(X_tr_np, n_grid=21)

    traj = {bb: run_one_backbone(bb, X_tr, X_te, y_te, anchor, rho, epochs, seed, dev)
            for bb in BACKBONES}

    finals = {bb: traj[bb][-1] for bb in BACKBONES}
    fvals = np.array(list(finals.values()))
    arr = np.array([traj[bb] for bb in BACKBONES])

    return {
        "dataset": dataset,
        "seed": seed,
        "rho": rho,
        "epochs": epochs,
        "final_auroc": finals,
        "peak_auroc": {bb: float(np.nanmax(traj[bb])) for bb in BACKBONES},
        "final_spread": float(np.nanmax(fvals) - np.nanmin(fvals)),
        "traj_spread": float(np.nanmax(np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0))),
        "tccm_eq_deepsvdd": bool(np.allclose(traj["tccm"], traj["deepsvdd"],
                                             atol=1e-9, equal_nan=True)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=KAR_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = SEC6 / "backbone_collapse"
    for ds in args.datasets:
        for s in args.seeds:
            res = run(ds, s, args.rho, args.epochs, args.device)
            out = prepare(out_dir / f"{ds}_s{s}_r{args.rho}.json", args.overwrite)
            out.write_text(json.dumps(res, indent=2))
            f = res["final_auroc"]
            print(f"[backbone] {ds} s{s} rho={args.rho} tccm={f['tccm']:.4f} "
                  f"ae={f['ae']:.4f} deepsvdd={f['deepsvdd']:.4f} "
                  f"final-spread={res['final_spread']:.2e}", flush=True)


if __name__ == "__main__":
    main()
