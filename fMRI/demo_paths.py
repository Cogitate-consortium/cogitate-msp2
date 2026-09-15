"""Shared BIDS / code paths for production and demo mode.

When DEMO=1 (or after sourcing demo_setup.sh), paths point at data_demo and
the demo subject CSV. Otherwise defaults match the cluster layout.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_BIDS_ROOT = Path("/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids")
SESSION = "ses-V2"
DEMO_SESSION_ON_DISK = "ses-2"


def is_demo() -> bool:
    return os.environ.get("DEMO", "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _resolve_paths() -> tuple[Path, Path, Path]:
    if is_demo():
        code = Path(os.environ.get("CODE_PATH", str(_REPO_ROOT)))
        bids = Path(os.environ.get("BIDS_ROOT", str(code / "data_demo")))
        subject_csv = Path(
            os.environ.get(
                "SUBJECT_CSV",
                str(code / "ses-v2-analysis-subs-fmri_demo.csv"),
            )
        )
        return bids, code, subject_csv

    bids = Path(os.environ.get("BIDS_ROOT", str(DEFAULT_BIDS_ROOT)))
    code = Path(os.environ.get("CODE_PATH", str(bids / "code")))
    subject_csv = Path(
        os.environ.get("SUBJECT_CSV", str(code / "ses-v2-analysis-subs-fmri.csv"))
    )
    return bids, code, subject_csv


BIDS_ROOT, CODE_PATH, SUBJECT_CSV = _resolve_paths()
