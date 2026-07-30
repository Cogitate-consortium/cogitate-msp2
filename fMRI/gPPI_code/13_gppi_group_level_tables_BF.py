#!/usr/bin/env python3
"""
Creates summary CSV tables for Bayes Factor results from gppi_group_analysis_BF.py.

For each seed (FFA, LOC) and each BF map (bf10, then bf01):
  - per ROI in roi_definitions.roi_list: % voxels with BF > 3 and BF value range
  - save additional tables restricted to GNW and IIT theory ROI sets
  - append a combined (union) theory row to each theory table and print it
  - print % and BF ranges across GNW and IIT constituent ROIs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from roi_definitions import GNW_roi_list, IIT_roi_list, roi_list

# Match BIDS root used by other gPPI scripts (override with $BIDS_ROOT if needed).
DEFAULT_BIDS_ROOT = Path(
    os.environ.get(
        "BIDS_ROOT",
        "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids",
    )
)
BIDS_ROOT = Path(os.environ.get("BIDS_ROOT", str(DEFAULT_BIDS_ROOT)))

# Bayes Factor maps from 12_gppi_group_analysis_BF.py
BF_ROOT = BIDS_ROOT / "derivatives" / "bf"
FFA_BF_path = BF_ROOT / "FFA"
LOC_BF_path = BF_ROOT / "LOC"

# Group-level theory ROI masks (bilateral *bh_{roi}_space-*.nii.gz).
# This pack is not referenced elsewhere in this repo — if your masks live under a
# different derivatives folder or atlas name, adjust MASKS_PATH (or set $MASKS_PATH).
MASKS_PATH = Path(
    os.environ.get(
        "MASKS_PATH",
        str(BIDS_ROOT / "derivatives" / "masks" / "ICBM2009c_asym_nlin"),
    )
)
masks_path = MASKS_PATH

# Where to write summary tables (same tree as BF maps)
OUT_DIR = BF_ROOT

EVIDENCE_THRESH = 3.0
BF_MAPS = ("bf10", "bf01")
FLOAT_FMT = "%.2f"

THEORY_ROIS = {
    "GNW": GNW_roi_list,  # GNWT / global neuronal workspace
    "IIT": IIT_roi_list,
}


def find_mask(roi: str) -> Path | None:
    """Locate the bilateral mask NIfTI for an ROI under masks_path."""
    preferred = sorted(masks_path.glob(f"*bh_{roi}_space-*.nii.gz"))
    if preferred:
        return preferred[0]

    # Fallback: any file whose name contains the ROI token between underscores/dots
    matches = []
    for p in masks_path.glob("*.nii.gz"):
        stem = p.name
        if f"_{roi}_" in stem or stem.startswith(f"{roi}_") or f"_{roi}." in stem:
            matches.append(p)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer bilateral (bh) if present
        bh = [m for m in matches if "_bh_" in m.name]
        return sorted(bh or matches)[0]
    return None


def load_volume(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).get_fdata(), dtype=np.float64)


def roi_bf_stats(bf_data: np.ndarray, mask_data: np.ndarray) -> dict:
    """
    Percent of ROI voxels with BF > threshold, plus BF value range.

    Only finite BF values inside the mask are counted.
    """
    roi = mask_data != 0
    vals = bf_data[roi]
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {
            "n_voxels": 0,
            "n_gt3": 0,
            "pct_gt3": np.nan,
            "bf_min": np.nan,
            "bf_max": np.nan,
            "bf_range": "",
        }

    n_gt = int(np.sum(vals > EVIDENCE_THRESH))
    bf_min = round(float(np.min(vals)), 2)
    bf_max = round(float(np.max(vals)), 2)
    return {
        "n_voxels": n,
        "n_gt3": n_gt,
        "pct_gt3": round(100.0 * n_gt / n, 2),
        "bf_min": bf_min,
        "bf_max": bf_max,
        "bf_range": f"{bf_min:.2f} – {bf_max:.2f}",
    }


TABLE_COLS = ["roi", "pct_gt3", "bf_range", "n_gt3", "n_voxels"]


def combine_theory_mask(members: list[str], ref_shape: tuple[int, ...]) -> np.ndarray:
    """Binary union of all available masks for a theory ROI set."""
    combined = np.zeros(ref_shape, dtype=bool)
    n_loaded = 0
    for roi in members:
        mask_file = find_mask(roi)
        if mask_file is None:
            print(f"  ! mask not found for theory ROI: {roi}")
            continue
        mask_data = load_volume(mask_file)
        if mask_data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for {roi}: expected {ref_shape}, got {mask_data.shape}"
            )
        combined |= mask_data != 0
        n_loaded += 1
    if n_loaded == 0:
        return np.zeros(ref_shape, dtype=np.float64)
    return combined.astype(np.float64)


def theory_combined_row(
    theory: str, members: list[str], bf_data: np.ndarray
) -> dict:
    """Stats treating all theory ROIs as one combined (union) mask."""
    combined_mask = combine_theory_mask(members, bf_data.shape)
    stats = roi_bf_stats(bf_data, combined_mask)
    return {
        "roi": theory,
        "pct_gt3": stats["pct_gt3"],
        "bf_range": stats["bf_range"],
        "bf_min": stats["bf_min"],
        "bf_max": stats["bf_max"],
        "n_gt3": stats["n_gt3"],
        "n_voxels": stats["n_voxels"],
    }


def summarize_seed(seed: str, bf_dir: Path) -> None:
    if not bf_dir.is_dir():
        print(f"[{seed}] SKIP — BF directory not found: {bf_dir}")
        return

    for bf_name in BF_MAPS:
        bf_file = bf_dir / f"{bf_name}.nii.gz"
        if not bf_file.is_file():
            print(f"[{seed}] SKIP — missing {bf_file}")
            continue

        print(f"\n=== {seed} / {bf_name} ===")
        bf_data = load_volume(bf_file)
        rows = []

        for roi in roi_list:
            mask_file = find_mask(roi)
            if mask_file is None:
                print(f"  ! mask not found for ROI: {roi}")
                rows.append(
                    {
                        "roi": roi,
                        "pct_gt3": np.nan,
                        "bf_range": "",
                        "bf_min": np.nan,
                        "bf_max": np.nan,
                        "n_gt3": 0,
                        "n_voxels": 0,
                        "mask_file": "",
                    }
                )
                continue

            mask_data = load_volume(mask_file)
            if mask_data.shape != bf_data.shape:
                raise ValueError(
                    f"Shape mismatch for {roi}: BF {bf_data.shape} vs mask {mask_data.shape} "
                    f"({mask_file})"
                )

            stats = roi_bf_stats(bf_data, mask_data)
            rows.append(
                {
                    "roi": roi,
                    "pct_gt3": stats["pct_gt3"],
                    "bf_range": stats["bf_range"],
                    "bf_min": stats["bf_min"],
                    "bf_max": stats["bf_max"],
                    "n_gt3": stats["n_gt3"],
                    "n_voxels": stats["n_voxels"],
                    "mask_file": mask_file.name,
                }
            )

        df = pd.DataFrame(rows)
        table = df[TABLE_COLS].copy()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / f"{seed}_{bf_name}_roi_summary.csv"
        table.to_csv(out_csv, index=False, float_format=FLOAT_FMT)
        print(f"  wrote {out_csv}")

        save_theory_tables(seed, bf_name, df, bf_data)
        print_theory_ranges(seed, bf_name, df)


def save_theory_tables(
    seed: str, bf_name: str, df: pd.DataFrame, bf_data: np.ndarray
) -> None:
    """Save CSV tables restricted to each theory's ROI set (GNW, IIT)."""
    print(f"\n  Combined theory ROI stats ({seed} / {bf_name}):")
    for theory, members in THEORY_ROIS.items():
        sub = df[df["roi"].isin(members)].copy()
        # Preserve theory list order
        order = {roi: i for i, roi in enumerate(members)}
        sub["_order"] = sub["roi"].map(order)
        sub = sub.sort_values("_order").drop(columns="_order")

        combined = theory_combined_row(theory, members, bf_data)
        theory_table = pd.concat(
            [sub[TABLE_COLS], pd.DataFrame([combined])[TABLE_COLS]],
            ignore_index=True,
        )

        if np.isfinite(combined["pct_gt3"]):
            print(
                f"    {theory} (all ROIs combined): "
                f"pct_gt3={combined['pct_gt3']:.2f}%; "
                f"bf_range={combined['bf_range']}; "
                f"n_gt3={combined['n_gt3']}/{combined['n_voxels']}"
            )
        else:
            print(f"    {theory} (all ROIs combined): no valid voxels")

        out_csv = OUT_DIR / f"{seed}_{bf_name}_{theory}_roi_summary.csv"
        theory_table.to_csv(out_csv, index=False, float_format=FLOAT_FMT)
        print(f"  wrote {out_csv}")


def print_theory_ranges(seed: str, bf_name: str, df: pd.DataFrame) -> None:
    """Print % and BF ranges across ROIs that make up each theory set."""
    print(f"\n  Theory ROI summaries across constituent ROIs ({seed} / {bf_name}):")
    for theory, members in THEORY_ROIS.items():
        sub = df[df["roi"].isin(members)].copy()
        sub = sub[np.isfinite(sub["pct_gt3"])]
        if sub.empty:
            print(f"    {theory}: no valid ROI rows")
            continue

        pct_lo = float(sub["pct_gt3"].min())
        pct_hi = float(sub["pct_gt3"].max())
        bf_lo = float(sub["bf_min"].min())
        bf_hi = float(sub["bf_max"].max())
        print(
            f"    {theory} (n={len(sub)} ROIs): "
            f"% voxels with BF>{EVIDENCE_THRESH:g} in [{pct_lo:.2f}, {pct_hi:.2f}]; "
            f"BF values in [{bf_lo:.2f}, {bf_hi:.2f}]"
        )


def main() -> None:
    if not masks_path.is_dir():
        raise FileNotFoundError(
            f"masks_path not found: {masks_path}\n"
            "  Set $MASKS_PATH or $BIDS_ROOT, or edit MASKS_PATH if your ROI pack "
            "is not under derivatives/masks/ICBM2009c_asym_nlin."
        )

    analyses = (
        ("FFA", FFA_BF_path),
        ("LOC", LOC_BF_path),
    )
    for seed, bf_dir in analyses:
        summarize_seed(seed, bf_dir)


if __name__ == "__main__":
    main()
