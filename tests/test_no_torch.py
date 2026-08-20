import importlib.util
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

PROGRAM = '''
import sys


class TorchBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name=fullname)
        return None


sys.meta_path.insert(0, TorchBlocker())

import numpy as np

import karkcm
from karkcm import KCM, KCMAnchor

assert "torch" not in sys.modules, "importing karkcm dragged torch in"

X = np.random.default_rng(0).normal(size=(80, 4))
m = KCM(assume_scaled=True).fit(X)
assert m.anomaly_score(X).shape == (80,)
assert set(np.unique(m.predict(X))) <= {-1, 1}
assert m.reconstruct(X).shape == (80, 4)
assert m.kernel_mass(X).shape == (80,)
assert KCMAnchor.fit(X).score_np(X).shape == (80,)

try:
    karkcm.run_kar
except ImportError as exc:
    assert "kar-kcm[kar]" in str(exc), str(exc)
    print("KAR error:", exc)
else:
    raise AssertionError("touching karkcm.run_kar should have raised ImportError")

assert "torch" not in sys.modules, "scoring dragged torch in"
print("OK")
'''


OTHER_MISSING = '''
import sys


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn.model_selection":
            raise ModuleNotFoundError("No module named 'sklearn.model_selection'",
                                      name=fullname)
        return None


sys.meta_path.insert(0, Blocker())

import importlib.util

import karkcm

assert importlib.util.find_spec("torch") is not None, "this case needs torch installed"

try:
    karkcm.run_kar
except ModuleNotFoundError as exc:
    assert exc.name == "sklearn.model_selection", exc.name
    print("propagated:", exc)
except ImportError as exc:
    raise AssertionError(f"blamed torch for a missing sklearn: {exc}")
else:
    raise AssertionError("touching karkcm.run_kar should have raised")

print("OK")
'''


def test_import_without_torch():
    proc = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "kar-kcm[kar]" in proc.stdout


def test_a_missing_non_torch_dependency_is_not_blamed_on_torch():
    """The lazy loader must not send a user to install a 367 MB package they already have.

    karkcm.kar transitively imports karkcm.data, karkcm.nets, scipy and several
    sklearn modules. Catching every ModuleNotFoundError and rewriting it as
    'needs PyTorch' turns any one of those going missing into wrong advice.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("this case only exists when torch IS installed")
    proc = subprocess.run(
        [sys.executable, "-c", OTHER_MISSING],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "kar-kcm[kar]" not in proc.stdout
