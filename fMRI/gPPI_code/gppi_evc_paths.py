"""Shared paths for gPPI 2nd-level EVC ROI workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gppi_3rd_level import ANALYSIS_COPE

SESSION = "ses-V2"
SPACE = "MNI152NLin2009cAsym"
DEFAULT_FILTER_COL = "SYNCHRONY_min_seen"

# Stat maps used by average_gppi_in_evc (inner stats inside cope3 / cope4 .feat)
EVC_STAT_FILES = ("cope1", "zstat1")
EVC_GLM_ANALYSES = ("PPI_FFA", "PPI_LOC")


def load_subjects(csv_path: Path, filter_col: str) -> list[str]:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(csv_path, sep=";")
    if filter_col not in df.columns:
        raise ValueError(f"missing column {filter_col} in {csv_path}")
    mask = df[filter_col].astype(str).str.upper().eq("TRUE")
    return [str(x) for x in df.loc[mask, "sub_code"].tolist()]


def resolve_2nd_stat(
    feat_root: Path,
    sub_code: str,
    glm_analysis: str,
    cope_num: int,
    stat_name: str,
) -> Path | None:
    subj_id = f"sub-{sub_code}"
    stem = f"{subj_id}_{SESSION}_task-VG_analysis-2ndGLM_{glm_analysis}-{SPACE}"
    base = feat_root / "gPPI" / subj_id
    for ext in (".gfeat", ".feat"):
        nii = base / f"{stem}{ext}" / f"cope{cope_num}.feat" / "stats" / f"{stat_name}.nii.gz"
        if nii.is_file():
            return nii
    return None


def dest_basename(glm_analysis: str, cope_num: int, stat_name: str) -> str:
    return f"{glm_analysis}_cope{cope_num}_{stat_name}.nii.gz"
