import argparse
import json
import time

import numpy as np
import torch

from ..baselines import DETECTORS, config_for
from ..data import KAR_DATASETS, load_split
from ..paths import SEC3, prepare

UNREGULARIZED_LAMBDA = 0.0


def run(detector: str, dataset: str, seed: int, cfg, device: str) -> dict:
    dev = torch.device(device)
    X_tr_np, X_te_np, y_te = load_split(dataset, seed)
    X_tr = torch.from_numpy(X_tr_np).float().to(dev)
    X_te = torch.from_numpy(X_te_np).float().to(dev)

    t0 = time.time()
    curve = DETECTORS[detector](cfg, X_tr, X_te, y_te, dev, seed)
    peak = float(np.nanmax(curve))
    final = float(curve[-1])

    return {
        "detector": detector,
        "dataset": dataset,
        "seed": seed,
        "lambda": UNREGULARIZED_LAMBDA,
        "epochs": cfg.epochs,
        "N_train": int(X_tr.shape[0]),
        "D": int(X_tr.shape[1]),
        "peak_auroc": peak,
        "final_auroc": final,
        "gap": peak - final,
        "auroc_per_epoch": curve,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    p.add_argument("--datasets", nargs="+", default=KAR_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--latent", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mask-rate", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for det in args.detectors:
        cfg = config_for(det, epochs=args.epochs, batch=args.batch, hidden=args.hidden,
                         latent=args.latent, lr=args.lr, mask_rate=args.mask_rate,
                         beta=args.beta)
        out_dir = SEC3 / "table2_source" / det
        for ds in args.datasets:
            for s in args.seeds:
                out = out_dir / f"{ds}_s{s}_l{UNREGULARIZED_LAMBDA}.json"
                if args.skip_existing and out.exists():
                    print(f"[skip] {det}/{out.name}")
                    continue
                res = run(det, ds, s, cfg, args.device)
                prepare(out, args.overwrite).write_text(json.dumps(res, indent=2))
                print(f"[{det}] {ds} s{s} peak={res['peak_auroc']:.4f} "
                      f"final={res['final_auroc']:.4f} gap={res['gap']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
