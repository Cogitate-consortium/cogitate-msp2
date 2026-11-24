#!/usr/bin/env bash
# Compare new vs old gPPI seed masks (FFA/LOC NIfTI) per subject.
#
# Overlap = min(100*|intersect|/|new|, 100*|intersect|/|old|) on binary voxels.
# Writes ALL pairs (overlap descending) to low_overlap.txt and all_overlap.txt.
#
# Uses Python/nibabel (no FSL). Run inside cogitate_fmri:
#   python3 -u compare_old_vs_new.sh
#   bash compare_old_vs_new.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NEW_ROOT="${NEW_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/new/gppi_seeds}"
export OLD_ROOT="${OLD_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/old/gppi_seeds}"
export ALL_OUT="${ALL_OUT:-${_SCRIPT_DIR}/all_overlap.txt}"
export LOW_OUT="${LOW_OUT:-${_SCRIPT_DIR}/low_overlap.txt}"
export THRESH="${THRESH:-95}"
export PYTHONUNBUFFERED=1

exec python3 -u - <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("ERROR: nibabel not found. Activate cogitate_fmri (or pip install nibabel).")

NEW_ROOT = Path(os.environ["NEW_ROOT"])
OLD_ROOT = Path(os.environ["OLD_ROOT"])
ALL_OUT = Path(os.environ["ALL_OUT"])
LOW_OUT = Path(os.environ["LOW_OUT"])
THRESH = float(os.environ["THRESH"])

SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def log(msg: str) -> None:
    print(msg, flush=True)


def stem_of(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def is_nifti(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if "timecourse" in name:
        return False
    return name.endswith(".nii.gz") or name.endswith(".nii")


def iter_niftis(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_nifti(p):
                found.append(p)
    found.sort()
    return found


def prefer_gz(paths: list[Path]) -> Path:
    for p in paths:
        if p.name.endswith(".nii.gz"):
            return p
    return paths[0]


def index_by_stem(root: Path) -> dict[str, Path]:
    buckets: dict[str, list[Path]] = {}
    for p in iter_niftis(root):
        buckets.setdefault(stem_of(p), []).append(p)
    return {stem: prefer_gz(ps) for stem, ps in buckets.items()}


def mask_label(stem: str) -> str:
    if "FFA" in stem:
        return "FFA"
    if "LOC" in stem:
        return "LOC"
    return stem


def load_binary(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return data != 0, tuple(int(x) for x in data.shape)


def tsv_line(fields: list[str]) -> str:
    return "\t".join(fields) + "\n"


HEADER = [
    "sub",
    "mask",
    "overlap_pct",
    "dice_pct",
    "n_new",
    "n_old",
    "n_intersect",
    "reason",
    "new_path",
    "old_path",
]


def main() -> int:
    if not NEW_ROOT.is_dir():
        log(f"ERROR: NEW_ROOT not found: {NEW_ROOT}")
        return 1
    if not OLD_ROOT.is_dir():
        log(f"ERROR: OLD_ROOT not found: {OLD_ROOT}")
        return 1

    new_index = index_by_stem(NEW_ROOT)
    old_index = index_by_stem(OLD_ROOT)
    log(f">> new: {NEW_ROOT}")
    log(f">> old: {OLD_ROOT}")
    log(f">> unique mask stems: new={len(new_index)}  old={len(old_index)}")

    by_sub: dict[str, list[tuple[str, Path]]] = {}
    for stem, path in sorted(new_index.items()):
        m = SUB_RE.search(path.name)
        if not m:
            log(f"WARN: skip (no sub-* in name): {path}")
            continue
        by_sub.setdefault(m.group(1), []).append((stem, path))

    subjects = sorted(by_sub)
    log(f">> subjects found: {len(subjects)}")
    if not subjects:
        log("ERROR: no sub-* NIfTI masks under NEW_ROOT")
        return 1

    rows: list[dict[str, str]] = []
    n_ok = n_low = n_missing = n_grid = n_fail = 0

    for i_sub, sub in enumerate(subjects, start=1):
        pairs = by_sub[sub]
        log(f">> [{i_sub}/{len(subjects)}] {sub}  masks={len(pairs)}")
        for stem, new_path in pairs:
            label = mask_label(stem)
            old_path = old_index.get(stem)
            if old_path is None:
                log(f"   {label}  MISSING_OLD  {new_path.name}")
                rows.append(
                    {
                        "sort": -1.0,
                        "sub": sub,
                        "mask": label,
                        "overlap_pct": "NA",
                        "dice_pct": "NA",
                        "n_new": "NA",
                        "n_old": "NA",
                        "n_intersect": "NA",
                        "reason": "MISSING_OLD",
                        "new_path": str(new_path),
                        "old_path": "",
                    }
                )
                n_missing += 1
                continue

            log(f"   {label}  loading {new_path.name}")
            try:
                new_bin, new_shape = load_binary(new_path)
                old_bin, old_shape = load_binary(old_path)
            except Exception as exc:
                log(f"   {label}  LOAD_FAILED  {exc}")
                rows.append(
                    {
                        "sort": -1.0,
                        "sub": sub,
                        "mask": label,
                        "overlap_pct": "NA",
                        "dice_pct": "NA",
                        "n_new": "NA",
                        "n_old": "NA",
                        "n_intersect": "NA",
                        "reason": f"LOAD_FAILED {exc}",
                        "new_path": str(new_path),
                        "old_path": str(old_path),
                    }
                )
                n_fail += 1
                continue

            if new_shape != old_shape:
                log(f"   {label}  GRID_MISMATCH  {new_shape} vs {old_shape}")
                rows.append(
                    {
                        "sort": -1.0,
                        "sub": sub,
                        "mask": label,
                        "overlap_pct": "NA",
                        "dice_pct": "NA",
                        "n_new": "NA",
                        "n_old": "NA",
                        "n_intersect": "NA",
                        "reason": f"GRID_MISMATCH {new_shape} vs {old_shape}",
                        "new_path": str(new_path),
                        "old_path": str(old_path),
                    }
                )
                n_grid += 1
                continue

            n_new = int(new_bin.sum())
            n_old = int(old_bin.sum())
            n_int = int(np.logical_and(new_bin, old_bin).sum())
            pct_new = (100.0 * n_int / n_new) if n_new else 0.0
            pct_old = (100.0 * n_int / n_old) if n_old else 0.0
            overlap = min(pct_new, pct_old)
            dice = (200.0 * n_int / (n_new + n_old)) if (n_new + n_old) else 0.0
            reason = "LOW_OVERLAP" if overlap < THRESH else "OK"
            if reason == "LOW_OVERLAP":
                n_low += 1
            else:
                n_ok += 1
            log(
                f"   {label}  {overlap:.2f}%  dice={dice:.2f}  "
                f"n={n_new}/{n_old}/{n_int}  {reason}"
            )
            rows.append(
                {
                    "sort": overlap,
                    "sub": sub,
                    "mask": label,
                    "overlap_pct": f"{overlap:.2f}",
                    "dice_pct": f"{dice:.2f}",
                    "n_new": str(n_new),
                    "n_old": str(n_old),
                    "n_intersect": str(n_int),
                    "reason": reason,
                    "new_path": str(new_path),
                    "old_path": str(old_path),
                }
            )

    rows.sort(key=lambda r: (-r["sort"], r["sub"], r["mask"]))

    meta = [
        "# new vs old gPPI seed overlap — ALL pairs, overlap descending (NA last)",
        "# overlap_pct = min(100*|intersect|/|new|, 100*|intersect|/|old|)",
        f"# reason=LOW_OVERLAP if overlap_pct < {THRESH:g}; OK otherwise",
        f"# subjects={len(subjects)}  pairs={len(rows)}  ok={n_ok}  "
        f"below_{THRESH:g}={n_low}  missing_old={n_missing}  "
        f"grid={n_grid}  load_fail={n_fail}",
        f"# NEW_ROOT={NEW_ROOT}",
        f"# OLD_ROOT={OLD_ROOT}",
        tsv_line(HEADER).rstrip("\n"),
    ]
    lines = list(meta)
    for r in rows:
        lines.append(tsv_line([r[k] for k in HEADER]).rstrip("\n"))
    text = "\n".join(lines) + "\n"
    ALL_OUT.write_text(text, encoding="utf-8")
    LOW_OUT.write_text(text, encoding="utf-8")

    log("")
    log(
        f">> subjects={len(subjects)}  pairs={len(rows)}  ok={n_ok}  "
        f"below_{THRESH:g}={n_low}  missing_old={n_missing}  "
        f"grid={n_grid}  load_fail={n_fail}"
    )
    log(f">> wrote ALL overlaps (desc) to:")
    log(f"   {LOW_OUT}")
    log(f"   {ALL_OUT}")
    log(f">> grep lows:  grep LOW_OVERLAP {LOW_OUT}")
    if not rows:
        log("ERROR: no comparison rows")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
