import argparse
import json
from pathlib import Path

from ..kar import run_kar
from ..paths import SEC6, prepare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--h-grid-n", type=int, default=21)
    p.add_argument("--max-ref", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    res = run_kar(
        dataset=args.dataset, seed=args.seed, rho=args.rho, epochs=args.epochs,
        batch=args.batch, hidden=args.hidden, depth=args.depth, lr=args.lr,
        h_grid_n=args.h_grid_n, max_ref=args.max_ref, device=args.device,
    )

    if args.out is None:
        out = SEC6 / "rho_ablation" / f"{args.dataset}_s{args.seed}_r{args.rho}.json"
    else:
        out = Path(args.out)
    prepare(out, args.overwrite).write_text(json.dumps(res, indent=2))

    print(f"[kar] {args.dataset} seed={args.seed} rho={args.rho} "
          f"AUROC={res['auroc']:.4f} peak={res['peak_auroc']:.4f} "
          f"gap={res['gap']:+.4f} KCM={res['kcm_auroc']:.4f}")


if __name__ == "__main__":
    main()
