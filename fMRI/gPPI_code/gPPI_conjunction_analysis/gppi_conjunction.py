#!/usr/bin/env python3
"""
Descriptive gPPI conjunction across dAT-seen ∩ AT-seen-go ∩ AT-seen-no-go.

Preregistration (ver 5.9): after group random-effects gPPI with GRF cluster
correction (cluster-forming one-sided p < 0.001 / z ≥ 3.1; cluster p < 0.05),
perform a descriptive conjunction to test whether similar brain regions appear
in all three seen conditions (esp. PFC for GNW). Run separately for FFA and LOC
seeds (PPI interaction maps: cope3 / cope4 at 3rd level).

Inputs: each condition's 3rd-level FEAT ``thresh_zstat1.nii.gz`` (cluster-
thresholded). Falls back to ``cluster_mask_zstat1.nii.gz`` if needed.

Outputs (under derivatives/gppi/conjunction/ by default):
  - gppi_conjunction_{FFA,LOC}_overlap_count.nii.gz   (0–3)
  - gppi_conjunction_{FFA,LOC}_dAT-seen_AT-seen-go_AT-seen-no-go.nii.gz
  - gppi_conjunction_{FFA,LOC}_summary.csv
  - gppi_conjunction_report.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo_paths import BIDS_ROOT as _DP_BIDS, CODE_PATH as _DP_CODE  # noqa: E402

# Theory ROI lists live with the dAT-seen pipeline (shared definitions).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dAT-seen"))
from roi_definitions import GNW_roi_list, IIT_roi_list  # noqa: E402

DEFAULT_BIDS_ROOT = Path(os.environ.get("BIDS_ROOT", str(_DP_BIDS)))
DEFAULT_CODE_ROOT = Path(os.environ.get("CODE_PATH", str(_DP_CODE)))

SESSION = "ses-V2"
SPACE = "MNI152NLin2009cAsym"
CONDITIONS = ("dAT-seen", "AT-seen-go", "AT-seen-no-go")
SEEDS = ("FFA", "LOC")

# Relative paths under each .gfeat
THRESH_CANDIDATES = (
    "cope1.feat/thresh_zstat1.nii.gz",
    "cope1.feat/cluster_mask_zstat1.nii.gz",
)

CONDITION_SPECS: dict[str, dict[str, str]] = {
    "dAT-seen": {
        "group_rel": "gPPI/group",
        "FFA": f"N*_{SESSION}_task-VG_analysis-3rdGLM_PPI_FFA-{SPACE}_desc-cope3.gfeat",
        "LOC": f"N*_{SESSION}_task-VG_analysis-3rdGLM_PPI_LOC-{SPACE}_desc-cope4.gfeat",
    },
    "AT-seen-go": {
        "group_rel": "gPPI_AT-seen-go/group",
        "FFA": f"N*_{SESSION}_task-Replay_analysis-3rdGLM_PPI_FFA-{SPACE}_desc-cope3.gfeat",
        "LOC": f"N*_{SESSION}_task-Replay_analysis-3rdGLM_PPI_LOC-{SPACE}_desc-cope4.gfeat",
    },
    "AT-seen-no-go": {
        "group_rel": "gPPI_AT-seen-no-go/group",
        "FFA": f"N*_{SESSION}_task-Replay_analysis-3rdGLM_PPI_FFA-{SPACE}_desc-cope3.gfeat",
        "LOC": f"N*_{SESSION}_task-Replay_analysis-3rdGLM_PPI_LOC-{SPACE}_desc-cope4.gfeat",
    },
}

THEORY_ROIS = {"GNW": GNW_roi_list, "IIT": IIT_roi_list}


@dataclass
class ResolvedMap:
    condition: str
    seed: str
    gfeat: Path
    thresh_path: Path


def resolve_unique_gfeat(group_dir: Path, pattern: str, label: str) -> Path | None:
    if not group_dir.is_dir():
        print(f"[{label}] SKIP — group dir missing: {group_dir}")
        return None
    matches = sorted(p for p in group_dir.glob(pattern) if p.is_dir())
    if not matches:
        print(f"[{label}] SKIP — no gfeat matching {group_dir / pattern}")
        return None
    if len(matches) > 1:
        raise FileExistsError(
            f"[{label}] ambiguous gfeat ({len(matches)} matches for {pattern}): "
            + ", ".join(str(m) for m in matches)
        )
    return matches[0]


def resolve_thresh_map(gfeat: Path) -> Path | None:
    for rel in THRESH_CANDIDATES:
        p = gfeat / rel
        if p.is_file():
            return p
    return None


def resolve_condition_map(
    feat_root: Path, condition: str, seed: str
) -> ResolvedMap | None:
    spec = CONDITION_SPECS[condition]
    group_dir = feat_root / spec["group_rel"]
    label = f"{condition}/{seed}"
    gfeat = resolve_unique_gfeat(group_dir, spec[seed], label)
    if gfeat is None:
        return None
    thresh = resolve_thresh_map(gfeat)
    if thresh is None:
        print(
            f"[{label}] SKIP — no thresh/cluster map under {gfeat} "
            f"(tried: {', '.join(THRESH_CANDIDATES)})"
        )
        return None
    return ResolvedMap(condition=condition, seed=seed, gfeat=gfeat, thresh_path=thresh)


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata(), dtype=np.float64)
    binary = (np.isfinite(data) & (data != 0)).astype(np.float64)
    return img, binary


def find_mask(masks_path: Path, roi: str) -> Path | None:
    preferred = sorted(masks_path.glob(f"*bh_{roi}_space-*.nii.gz"))
    if preferred:
        return preferred[0]
    matches = []
    for p in masks_path.glob("*.nii.gz"):
        stem = p.name
        if f"_{roi}_" in stem or stem.startswith(f"{roi}_") or f"_{roi}." in stem:
            matches.append(p)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        bh = [m for m in matches if "_bh_" in m.name]
        return sorted(bh or matches)[0]
    return None


def theory_overlap_rows(
    conjunction: np.ndarray, masks_path: Path
) -> list[dict]:
    rows: list[dict] = []
    conj = conjunction != 0
    n_conj = int(conj.sum())
    for theory, members in THEORY_ROIS.items():
        union = np.zeros(conjunction.shape, dtype=bool)
        n_loaded = 0
        for roi in members:
            mask_file = find_mask(masks_path, roi)
            if mask_file is None:
                continue
            mask = np.asanyarray(nib.load(str(mask_file)).get_fdata()) != 0
            if mask.shape != conjunction.shape:
                raise ValueError(
                    f"Shape mismatch for {roi}: conj {conjunction.shape} vs mask {mask.shape}"
                )
            union |= mask
            n_loaded += 1
        n_overlap = int((conj & union).sum()) if n_loaded else 0
        rows.append(
            {
                "roi_set": theory,
                "n_masks_loaded": n_loaded,
                "n_conjunction_voxels": n_conj,
                "n_overlap_voxels": n_overlap,
                "pct_conjunction_in_set": (
                    round(100.0 * n_overlap / n_conj, 2) if n_conj else np.nan
                ),
            }
        )
    return rows


def run_seed(
    seed: str,
    *,
    feat_root: Path,
    out_dir: Path,
    masks_path: Path | None,
) -> tuple[bool, list[str]]:
    lines: list[str] = [f"## seed={seed}"]
    resolved: list[ResolvedMap] = []
    for cond in CONDITIONS:
        r = resolve_condition_map(feat_root, cond, seed)
        if r is None:
            lines.append(f"missing: {cond}")
            return False, lines
        resolved.append(r)
        lines.append(f"input[{cond}]={r.thresh_path}")

    binaries: list[np.ndarray] = []
    ref_img: nib.Nifti1Image | None = None
    for r in resolved:
        img, binary = load_binary(r.thresh_path)
        if ref_img is None:
            ref_img = img
        elif img.shape != ref_img.shape:
            raise ValueError(
                f"Shape mismatch {r.condition}/{seed}: "
                f"{img.shape} vs ref {ref_img.shape}"
            )
        binaries.append(binary)
        n_sig = int(binary.sum())
        lines.append(f"n_sig_voxels[{r.condition}]={n_sig}")
        # Keep per-condition binaries for inspection
        out_bin = out_dir / f"gppi_conjunction_{seed}_{r.condition}_thresh_bin.nii.gz"
        nib.save(nib.Nifti1Image(binary, ref_img.affine, ref_img.header), str(out_bin))

    assert ref_img is not None
    overlap_count = np.sum(np.stack(binaries, axis=0), axis=0)
    conjunction = (overlap_count >= 3).astype(np.float64)
    n_conj = int(conjunction.sum())
    lines.append(f"n_conjunction_voxels={n_conj}")

    count_path = out_dir / f"gppi_conjunction_{seed}_overlap_count.nii.gz"
    conj_path = (
        out_dir
        / f"gppi_conjunction_{seed}_dAT-seen_AT-seen-go_AT-seen-no-go.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(overlap_count.astype(np.float64), ref_img.affine, ref_img.header),
        str(count_path),
    )
    nib.save(
        nib.Nifti1Image(conjunction, ref_img.affine, ref_img.header),
        str(conj_path),
    )
    lines.append(f"wrote={count_path}")
    lines.append(f"wrote={conj_path}")

    summary_rows = [
        {
            "seed": seed,
            "metric": "n_sig",
            "condition": r.condition,
            "n_voxels": int(b.sum()),
        }
        for r, b in zip(resolved, binaries)
    ]
    summary_rows.append(
        {
            "seed": seed,
            "metric": "conjunction_3way",
            "condition": "dAT-seen∩AT-seen-go∩AT-seen-no-go",
            "n_voxels": n_conj,
        }
    )
    for k in (1, 2, 3):
        summary_rows.append(
            {
                "seed": seed,
                "metric": f"overlap_count_eq_{k}",
                "condition": "any",
                "n_voxels": int((overlap_count == k).sum()),
            }
        )

    if masks_path is not None and masks_path.is_dir():
        for row in theory_overlap_rows(conjunction, masks_path):
            summary_rows.append(
                {
                    "seed": seed,
                    "metric": f"conjunction_in_{row['roi_set']}",
                    "condition": "theory_ROI",
                    "n_voxels": row["n_overlap_voxels"],
                    "pct_conjunction_in_set": row["pct_conjunction_in_set"],
                    "n_masks_loaded": row["n_masks_loaded"],
                }
            )
            lines.append(
                f"theory[{row['roi_set']}]: overlap={row['n_overlap_voxels']} "
                f"({row['pct_conjunction_in_set']}% of conjunction; "
                f"masks={row['n_masks_loaded']})"
            )
    else:
        lines.append("theory_ROI: skipped (masks path missing)")

    summary_csv = out_dir / f"gppi_conjunction_{seed}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    lines.append(f"wrote={summary_csv}")
    return True, lines


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Descriptive gPPI conjunction: dAT-seen ∩ AT-seen-go ∩ AT-seen-no-go "
            "(FFA and LOC seeds; cluster-thresholded group maps)"
        )
    )
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", str(DEFAULT_BIDS_ROOT)))
    p.add_argument("--feat-root", default=os.environ.get("FSLFEAT_ROOT", ""))
    p.add_argument("--out-dir", default=os.environ.get("GPPI_CONJ_OUT_DIR", ""))
    p.add_argument(
        "--masks-path",
        default=os.environ.get(
            "MASKS_PATH",
            "",
        ),
    )
    args = p.parse_args()

    bids_root = Path(args.bids_root)
    feat_root = Path(args.feat_root) if args.feat_root else bids_root / "derivatives/fslFeat"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else bids_root / "derivatives" / "gppi" / "conjunction"
    )
    masks_path = (
        Path(args.masks_path)
        if args.masks_path
        else bids_root / "derivatives" / "masks" / "ICBM2009c_asym_nlin"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# gPPI descriptive conjunction report",
        f"# date={datetime.now(timezone.utc).isoformat()}",
        f"# feat_root={feat_root}",
        f"# out_dir={out_dir}",
        f"# masks_path={masks_path}",
        f"# conditions={','.join(CONDITIONS)}",
        "# method=intersection of nonzero thresh_zstat1 (cluster-corrected group maps)",
        "",
    ]

    ok_any = False
    for seed in SEEDS:
        ok, lines = run_seed(
            seed, feat_root=feat_root, out_dir=out_dir, masks_path=masks_path
        )
        report_lines.extend(lines)
        report_lines.append("")
        ok_any = ok_any or ok
        status = "OK" if ok else "INCOMPLETE"
        print(f">> {seed}: {status}")

    report_path = out_dir / "gppi_conjunction_report.txt"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f">> report: {report_path}")

    if not ok_any:
        print(
            "ERROR: no complete three-way conjunction (missing one or more condition maps)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
