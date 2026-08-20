"""Dataset and results roots, and the guard that keeps the published tree intact."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("KARKCM_DATASETS", REPO_ROOT / "datasets"))
RESULTS_ROOT = Path(os.environ.get("KARKCM_RESULTS", REPO_ROOT / "results"))

SEC3 = RESULTS_ROOT / "section3_collapse"
SEC4 = RESULTS_ROOT / "section4_kcm"
SEC6 = RESULTS_ROOT / "section6_kar"

PUBLISHED = RESULTS_ROOT == REPO_ROOT / "results"


class WouldOverwritePublished(RuntimeError):
    """Raised when a re-run would replace a file shipped with the paper.

    ``results/`` is the record behind the published tables, not a scratch
    directory. A re-run reproduces most of it but not all of it: the shipped
    per-run JSONs carry fields the current scripts do not emit, so overwriting
    silently degrades the artifact even when every number agrees.
    """


def prepare(path, overwrite=False):
    """Create the parent directory and refuse to clobber a published file.

    Parameters
    ----------
    path : path-like
        The file about to be written.
    overwrite : bool, default=False
        Allow replacing an existing file under the repository's own
        ``results/``. Ignored when `KARKCM_RESULTS` points elsewhere.

    Returns
    -------
    pathlib.Path
        `path`, with its parent directory created.

    Raises
    ------
    WouldOverwritePublished
        When the file exists under the repository's ``results/`` and
        `overwrite` is false.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if PUBLISHED and path.exists() and not overwrite:
        raise WouldOverwritePublished(
            f"{path} ships with the paper. Write somewhere else with "
            f"KARKCM_RESULTS=/some/dir, or pass --overwrite to replace it."
        )
    return path
