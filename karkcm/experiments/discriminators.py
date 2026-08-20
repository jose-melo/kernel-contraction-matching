import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..data import load_split
from ..metrics import auroc
from ..nets import TimeConditionedMLP, set_seed
from ..paths import SEC3, prepare

DATASETS = ["7_Cardiotocography", "2_annthyroid"]
SEEDS = [0, 1, 2]
CONFIGS = [(1, 1.0), (3, 0.25), (3, 0.5), (3, 0.75), (3, 1.0)]
EPOCHS = 2000


def tccm_score(model: TimeConditionedMLP, x: torch.Tensor) -> torch.Tensor:
    t1 = torch.ones(len(x), 1, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        v = model(x, t1)
    return torch.linalg.norm(v + x, dim=-1)


def run_one(dataset: str, seed: int, depth: int, n_frac: float, epochs: int,
            device: str = "cpu") -> dict:
    set_seed(seed)
    dev = torch.device(device)

    X_tr_np, X_te_np, y_te = load_split(dataset, seed)
    N_full = X_tr_np.shape[0]
    if n_frac < 1.0:
        n_keep = max(8, int(round(n_frac * N_full)))
        idx = np.random.default_rng(1000 + seed).choice(N_full, n_keep, replace=False)
        X_tr_np = X_tr_np[idx]
    N, D = X_tr_np.shape

    X_tr = torch.from_numpy(X_tr_np).float().to(dev)
    X_te = torch.from_numpy(X_te_np).float().to(dev)
    y_te = np.asarray(y_te)

    model = TimeConditionedMLP(d_in=D, d_out=D, hidden=256, t_embed=32, depth=depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    bsize = min(256, N)
    n_batches = max(1, (N + bsize - 1) // bsize)
    auroc_pe, loss_pe = [], []
    s_train_pe, s_tn_pe, s_ta_pe = [], [], []

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        losses = []
        for b in range(n_batches):
            x_b = X_tr[perm[b * bsize : (b + 1) * bsize]]
            t = torch.rand(len(x_b), 1, device=dev)
            eps = torch.randn_like(x_b)
            z = (1.0 - t) * eps + t * x_b
            loss = F.mse_loss(model(z, t), -z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        loss_pe.append(float(np.mean(losses)))

        model.eval()
        s_te = tccm_score(model, X_te).cpu().numpy()
        s_tr = tccm_score(model, X_tr).cpu().numpy()
        auroc_pe.append(auroc(y_te, s_te))
        s_train_pe.append(float(np.mean(s_tr)))
        s_tn_pe.append(float(np.mean(s_te[y_te == 0])))
        s_ta_pe.append(float(np.mean(s_te[y_te == 1])))

    return {
        "detector": "tccm_vanilla",
        "dataset": dataset,
        "seed": seed,
        "depth": depth,
        "linear": depth == 1,
        "n_frac": n_frac,
        "N_train": int(N),
        "N_train_full": int(N_full),
        "D": int(D),
        "n_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "auroc_per_epoch": auroc_pe,
        "train_loss_per_epoch": loss_pe,
        "score_train_per_epoch": s_train_pe,
        "score_test_normal_per_epoch": s_tn_pe,
        "score_test_anom_per_epoch": s_ta_pe,
        "wall_s": time.time() - t0,
    }


def tag(dataset: str, seed: int, depth: int, n_frac: float) -> str:
    return f"{dataset}_s{seed}_d{depth}_f{n_frac:g}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = SEC3 / "discriminators"
    jobs = [(ds, s, d, f) for ds in args.datasets for s in args.seeds for d, f in CONFIGS]
    print(f"[grid] {len(jobs)} runs -> {out_dir}")

    for i, (ds, s, d, f) in enumerate(jobs, 1):
        out = out_dir / f"{tag(ds, s, d, f)}.json"
        if args.skip_existing and out.exists():
            print(f"[{i}/{len(jobs)}] skip {out.name}")
            continue
        res = run_one(ds, s, d, f, args.epochs, args.device)
        prepare(out, args.overwrite).write_text(json.dumps(res, indent=2))
        pk = float(np.nanmax(res["auroc_per_epoch"]))
        fn = float(res["auroc_per_epoch"][-1])
        print(f"[{i}/{len(jobs)}] {out.name} peak={pk:.4f} final={fn:.4f} "
              f"gap={pk - fn:+.4f} ({res['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
