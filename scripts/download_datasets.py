import argparse
import importlib.util
import sys
import urllib.error
import urllib.request
from pathlib import Path

if importlib.util.find_spec("karkcm") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from karkcm.data import ADBENCH_47
from karkcm.paths import DATA_ROOT

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"


def fetch(name: str, dest: Path, force: bool) -> str:
    out = dest / f"{name}.npz"
    if out.exists() and not force:
        return "have"
    try:
        with urllib.request.urlopen(f"{BASE}/{name}.npz", timeout=120) as r:
            payload = r.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[fail] {name}: {exc}")
        return "fail"
    out.write_bytes(payload)
    return "new"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ADBENCH_47)
    ap.add_argument("--dest", default=str(DATA_ROOT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tally = {"have": 0, "new": 0, "fail": 0}
    for i, name in enumerate(args.datasets, 1):
        status = fetch(name, dest, args.force)
        tally[status] += 1
        print(f"[{i:>2}/{len(args.datasets)}] {name:<22} {status}", flush=True)

    print(f"\n{tally['new']} downloaded, {tally['have']} already present, "
          f"{tally['fail']} failed -> {dest}")
    sys.exit(1 if tally["fail"] else 0)


if __name__ == "__main__":
    main()
