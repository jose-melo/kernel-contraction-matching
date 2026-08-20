"""Kernel Contraction Matching and Kernel-Anchored Regularization."""

from .estimator import (
    KCM,
    KCMPipeline,
    NotStandardizedWarning,
    anomaly_auprc,
    anomaly_auroc,
    make_kcm_pipeline,
)
from .kcm import KCMAnchor

__version__ = "1.0.0"

_LAZY = {"bounded_correction": "kar", "kar_score": "kar", "run_kar": "kar"}

__all__ = [
    "KCM",
    "KCMAnchor",
    "KCMPipeline",
    "NotStandardizedWarning",
    "anomaly_auprc",
    "anomaly_auroc",
    "bounded_correction",
    "kar_score",
    "make_kcm_pipeline",
    "run_kar",
    "__version__",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        loaded = importlib.import_module(f".{module}", __name__)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "torch":
            raise
        raise ImportError(
            f"karkcm.{name} needs PyTorch, which KCM itself does not. "
            'Install it with: pip install "kar-kcm[kar]"'
        ) from exc
    return getattr(loaded, name)


def __dir__():
    return sorted(__all__)
