#!/usr/bin/env python3
"""
Voxel-wise Bayes Factor maps from group-level FEAT t-statistics.

Uses Pingouin's JZS Bayes Factor (BF10) for a one-sample t-test against zero,
applied separately to the FFA and LOC gPPI group FEAT directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pingouin as pg

# Match BIDS root used by other gPPI scripts (override with $BIDS_ROOT if needed).
DEFAULT_BIDS_ROOT = Path(
    os.environ.get(
        "BIDS_ROOT",
        "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids",
    )
)
BIDS_ROOT = Path(os.environ.get("BIDS_ROOT", str(DEFAULT_BIDS_ROOT)))

# 3rd-level group FEAT outputs (see 09_make_fsf_files_3rdlevel.sh / fsf template).
# Directory names are N{n}_ses-V2_... where n = included subjects at FSF build time.
FSLFEAT_GROUP = BIDS_ROOT / "derivatives" / "fslFeat" / "gPPI" / "group"

# Stem patterns relative to FSLFEAT_GROUP (N* filled by glob; cope indices from gppi_3rd_level).
FFA_GFEAT_GLOB = "N*_ses-V2_task-VG_analysis-3rdGLM_PPI_FFA-MNI152NLin2009cAsym_desc-cope3.gfeat"
LOC_GFEAT_GLOB = "N*_ses-V2_task-VG_analysis-3rdGLM_PPI_LOC-MNI152NLin2009cAsym_desc-cope4.gfeat"

# Paths relative to each .gfeat directory
TSTAT_REL = "cope1.feat/stats/tstat1.nii.gz"
DOF_REL = "cope1.feat/stats/dof"
MASK_REL = "cope1.feat/mask.nii.gz"

# Cauchy prior scale (Pingouin / Rouder et al. default)
CAUCHY_R = 0.707

# Output root (relative to BIDS). Adjust if BF maps should live elsewhere.
OUT_DIR = BIDS_ROOT / "derivatives" / "bf"


def resolve_gfeat(group_dir: Path, pattern: str, label: str) -> Path | None:
    """
    Find the unique 3rd-level .gfeat matching pattern under group_dir.

    Returns None if missing. Raises if multiple matches (ambiguous N / naming).
    """
    matches = sorted(p for p in group_dir.glob(pattern) if p.is_dir())
    if not matches:
        print(f"[{label}] SKIP — no directory matching {group_dir / pattern}")
        # If your group FEAT used a nonstandard suffix (e.g. legacy _GUI / _GUI+),
        # set FFA_GFEAT_GLOB / LOC_GFEAT_GLOB above or point to an absolute path.
        return None
    if len(matches) > 1:
        raise FileExistsError(
            f"[{label}] ambiguous gfeat ({len(matches)} matches for {pattern}): "
            + ", ".join(str(m) for m in matches)
            + " — set an explicit path or remove extras"
        )
    return matches[0]


def read_dof(dof_path: str | Path) -> float:
    """Read FSL dof file (single float)."""
    text = Path(dof_path).read_text().strip().split()
    if not text:
        raise ValueError(f"Empty dof file: {dof_path}")
    return float(text[0])


def compute_bf_maps(
    tstat_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image | None,
    n: int,
    r: float = CAUCHY_R,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bf10, log10_bf10) arrays in the same shape as the t-stat volume.

    BF10 is computed voxel-by-voxel with pingouin.bayesfactor_ttest.
    Voxels outside the mask (or with non-finite t) are set to NaN.
    """
    t = np.asanyarray(tstat_img.get_fdata(), dtype=np.float64)
    if mask_img is not None:
        mask = np.asanyarray(mask_img.get_fdata()) != 0
    else:
        mask = np.isfinite(t) & (t != 0)

    bf10 = np.full(t.shape, np.nan, dtype=np.float64)
    idx = np.where(mask & np.isfinite(t))
    n_vox = len(idx[0])
    print(f"  computing BF10 for {n_vox} voxels...")

    for i, (x, y, z) in enumerate(zip(*idx)):
        bf10[x, y, z] = pg.bayesfactor_ttest(float(t[x, y, z]), nx=n, paired=False, r=r)
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1}/{n_vox}")

    with np.errstate(divide="ignore", invalid="ignore"):
        log10_bf10 = np.log10(bf10)

    return bf10, log10_bf10


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    """Save data with the same affine/header as ref_img."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))
    print(f"  wrote {out_path}")


def process_feat(feat_dir: str | Path, label: str, out_dir: Path) -> Path:
    """
    Compute and save BF10 / log10(BF10) maps for one group FEAT directory.

    Returns the output directory used.
    """
    feat_dir = Path(feat_dir)
    tstat_path = feat_dir / TSTAT_REL
    dof_path = feat_dir / DOF_REL
    mask_path = feat_dir / MASK_REL

    if not tstat_path.is_file():
        raise FileNotFoundError(f"[{label}] missing t-stat: {tstat_path}")
    if not dof_path.is_file():
        raise FileNotFoundError(f"[{label}] missing dof: {dof_path}")

    dof = read_dof(dof_path)
    n = int(round(dof + 1))  # one-sample group t-test: df = n - 1
    print(f"[{label}] feat={feat_dir}")
    print(f"[{label}] dof={dof:g} -> n={n}")

    tstat_img = nib.load(str(tstat_path))
    mask_img = nib.load(str(mask_path)) if mask_path.is_file() else None
    if mask_img is None:
        print(f"[{label}] warning: no mask at {mask_path}; using finite non-zero t voxels")

    bf10, log10_bf10 = compute_bf_maps(tstat_img, mask_img, n=n, r=CAUCHY_R)

    out_dir = Path(out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(bf10, tstat_img, out_dir / "bf10.nii.gz")
    save_nifti(log10_bf10, tstat_img, out_dir / "log10_bf10.nii.gz")

    # Evidence for null (BF01), convenient for "support H0" maps
    with np.errstate(divide="ignore", invalid="ignore"):
        bf01 = 1.0 / bf10
        log10_bf01 = np.log10(bf01)
    save_nifti(bf01, tstat_img, out_dir / "bf01.nii.gz")
    save_nifti(log10_bf01, tstat_img, out_dir / "log10_bf01.nii.gz")

    finite = np.isfinite(bf10)
    if np.any(finite):
        print(
            f"[{label}] BF10 in-mask: "
            f"min={np.nanmin(bf10):.4g}, median={np.nanmedian(bf10):.4g}, "
            f"max={np.nanmax(bf10):.4g} ({int(finite.sum())} voxels)"
        )
    return out_dir


def main() -> None:
    analyses = (
        ("FFA", FFA_GFEAT_GLOB),
        ("LOC", LOC_GFEAT_GLOB),
    )
    if not FSLFEAT_GROUP.is_dir():
        print(f"ERROR: group FEAT root not found: {FSLFEAT_GROUP}")
        print("  Set $BIDS_ROOT or edit DEFAULT_BIDS_ROOT if your data live elsewhere.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, pattern in analyses:
        feat = resolve_gfeat(FSLFEAT_GROUP, pattern, label)
        if feat is None:
            continue
        process_feat(feat, label=label, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
