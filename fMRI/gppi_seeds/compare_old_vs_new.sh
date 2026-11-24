#!/usr/bin/env bash
# Compare new vs old gPPI seed masks (FFA/LOC NIfTI) per subject.
#
# Overlap = min(100*|intersect|/|new|, 100*|intersect|/|old|) on binary voxels
# in the same voxel grid (after squeezing trailing singleton dimensions).
# If several old files share a filename, each is compared and the LOWEST
# overlap is kept (avoids matching a copy of the new mask inside old/).
#
# Writes ALL pairs (overlap descending) to low_overlap.txt and all_overlap.txt.
#
# Uses Python/nibabel (no FSL). Inside cogitate_fmri:
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
AFFINE_ATOL = 1e-4


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


def unique_resolved(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def index_by_stem(root: Path) -> dict[str, list[Path]]:
    """All NIfTIs per filename stem. .nii and .nii.gz in the same folder count once."""
    groups: dict[tuple[str, str], list[Path]] = {}
    for p in iter_niftis(root):
        groups.setdefault((str(p.parent), stem_of(p)), []).append(p)
    buckets: dict[str, list[Path]] = {}
    for (_parent, stem), ps in groups.items():
        buckets.setdefault(stem, []).append(prefer_gz(ps))
    return {stem: unique_resolved(ps) for stem, ps in buckets.items()}


def mask_label(stem: str) -> str:
    if "FFA" in stem:
        return "FFA"
    if "LOC" in stem:
        return "LOC"
    return stem


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.squeeze(np.asanyarray(img.dataobj))
    binary = np.asarray(data != 0, dtype=bool)
    return binary, np.asarray(img.affine, dtype=np.float64)


def overlap_stats(new_bin: np.ndarray, old_bin: np.ndarray) -> tuple[float, float, int, int, int, int, int]:
    n_new = int(new_bin.sum())
    n_old = int(old_bin.sum())
    n_int = int(np.logical_and(new_bin, old_bin).sum())
    n_only_new = int(np.logical_and(new_bin, np.logical_not(old_bin)).sum())
    n_only_old = int(np.logical_and(old_bin, np.logical_not(new_bin)).sum())
    pct_new = (100.0 * n_int / n_new) if n_new else 0.0
    pct_old = (100.0 * n_int / n_old) if n_old else 0.0
    overlap = min(pct_new, pct_old)
    dice = (200.0 * n_int / (n_new + n_old)) if (n_new + n_old) else 0.0
    return overlap, dice, n_new, n_old, n_int, n_only_new, n_only_old


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
    "n_only_new",
    "n_only_old",
    "n_old_files",
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
    log(
        f">> unique mask stems: new={len(new_index)}  old={len(old_index)}  "
        f"old_files={sum(len(v) for v in old_index.values())}"
    )
    n_dup = sum(1 for v in old_index.values() if len(v) > 1)
    if n_dup:
        log(f">> WARN: {n_dup} old filename(s) appear in more than one folder; using lowest overlap")

    by_sub: dict[str, list[tuple[str, Path]]] = {}
    for stem, paths in sorted(new_index.items()):
        if len(paths) != 1:
            log(f"WARN: {len(paths)} new files for stem {stem}; using {paths[0]}")
        path = paths[0]
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

    rows: list[dict[str, object]] = []
    n_ok = n_low = n_missing = n_grid = n_fail = n_same = 0

    for i_sub, sub in enumerate(subjects, start=1):
        pairs = by_sub[sub]
        log(f">> [{i_sub}/{len(subjects)}] {sub}  masks={len(pairs)}")
        for stem, new_path in pairs:
            label = mask_label(stem)
            new_res = new_path.resolve()
            old_cands = [
                p
                for p in old_index.get(stem, [])
                if p.resolve() != new_res
            ]
            if not old_cands:
                reason = "MISSING_OLD"
                log(f"   {label}  {reason}  {new_path.name}")
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
                        "n_only_new": "NA",
                        "n_only_old": "NA",
                        "n_old_files": "0",
                        "reason": reason,
                        "new_path": str(new_path),
                        "old_path": "",
                    }
                )
                n_missing += 1
                continue

            try:
                new_bin, new_aff = load_mask(new_path)
            except Exception as exc:
                log(f"   {label}  LOAD_FAILED new  {exc}")
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
                        "n_only_new": "NA",
                        "n_only_old": "NA",
                        "n_old_files": str(len(old_cands)),
                        "reason": f"LOAD_FAILED {exc}",
                        "new_path": str(new_path),
                        "old_path": str(old_cands[0]),
                    }
                )
                n_fail += 1
                continue

            cand_rows: list[dict[str, object]] = []
            for old_path in old_cands:
                try:
                    old_bin, old_aff = load_mask(old_path)
                except Exception as exc:
                    log(f"   {label}  LOAD_FAILED old {old_path}  {exc}")
                    continue
                if new_bin.shape != old_bin.shape:
                    cand_rows.append(
                        {
                            "sort": -1.0,
                            "overlap_pct": "NA",
                            "dice_pct": "NA",
                            "n_new": str(int(new_bin.sum())),
                            "n_old": str(int(old_bin.sum())),
                            "n_intersect": "NA",
                            "n_only_new": "NA",
                            "n_only_old": "NA",
                            "reason": f"GRID_MISMATCH {tuple(new_bin.shape)} vs {tuple(old_bin.shape)}",
                            "old_path": str(old_path),
                            "identical": False,
                        }
                    )
                    continue
                if not np.allclose(new_aff, old_aff, atol=AFFINE_ATOL, rtol=0):
                    cand_rows.append(
                        {
                            "sort": -1.0,
                            "overlap_pct": "NA",
                            "dice_pct": "NA",
                            "n_new": str(int(new_bin.sum())),
                            "n_old": str(int(old_bin.sum())),
                            "n_intersect": "NA",
                            "n_only_new": "NA",
                            "n_only_old": "NA",
                            "reason": "AFFINE_MISMATCH",
                            "old_path": str(old_path),
                            "identical": False,
                        }
                    )
                    continue
                ov, dice, n_new, n_old, n_int, n_only_new, n_only_old = overlap_stats(
                    new_bin, old_bin
                )
                identical = n_only_new == 0 and n_only_old == 0
                cand_rows.append(
                    {
                        "sort": ov,
                        "overlap_pct": f"{ov:.2f}",
                        "dice_pct": f"{dice:.2f}",
                        "n_new": str(n_new),
                        "n_old": str(n_old),
                        "n_intersect": str(n_int),
                        "n_only_new": str(n_only_new),
                        "n_only_old": str(n_only_old),
                        "reason": "IDENTICAL" if identical else ("LOW_OVERLAP" if ov < THRESH else "OK"),
                        "old_path": str(old_path),
                        "identical": identical,
                    }
                )

            if not cand_rows:
                log(f"   {label}  LOAD_FAILED all old candidates")
                n_fail += 1
                continue

            numeric = [c for c in cand_rows if c["sort"] != -1.0]
            if numeric:
                best = min(numeric, key=lambda c: c["sort"])
            else:
                best = cand_rows[0]

            reason = str(best["reason"])
            if len(old_cands) > 1:
                ovs = ",".join(
                    ("IDENTICAL" if c["identical"] else str(c["overlap_pct"]))
                    for c in cand_rows
                )
                log(
                    f"   {label}  {len(old_cands)} old files for this name; "
                    f"keeping lowest overlap; candidates={ovs}"
                )
                if reason in {"OK", "IDENTICAL", "LOW_OVERLAP"}:
                    reason = f"{reason};DUPLICATE_OLD"

            if "GRID_MISMATCH" in str(best["reason"]):
                n_grid += 1
            elif "AFFINE_MISMATCH" in str(best["reason"]):
                n_grid += 1
            elif best["identical"]:
                n_same += 1
                n_ok += 1
            elif best["sort"] == -1.0:
                n_fail += 1
            elif float(best["sort"]) < THRESH:
                n_low += 1
                reason = reason.replace("OK;", "LOW_OVERLAP;")
                if reason == "OK":
                    reason = "LOW_OVERLAP"
            else:
                n_ok += 1

            log(
                f"   {label}  {best['overlap_pct']}%  dice={best['dice_pct']}  "
                f"n={best['n_new']}/{best['n_old']}/{best['n_intersect']}  "
                f"only_new={best['n_only_new']} only_old={best['n_only_old']}  "
                f"{reason}"
            )
            rows.append(
                {
                    "sort": best["sort"],
                    "sub": sub,
                    "mask": label,
                    "overlap_pct": best["overlap_pct"],
                    "dice_pct": best["dice_pct"],
                    "n_new": best["n_new"],
                    "n_old": best["n_old"],
                    "n_intersect": best["n_intersect"],
                    "n_only_new": best["n_only_new"],
                    "n_only_old": best["n_only_old"],
                    "n_old_files": str(len(old_cands)),
                    "reason": reason,
                    "new_path": str(new_path),
                    "old_path": best["old_path"],
                }
            )

    rows.sort(key=lambda r: (-float(r["sort"]), str(r["sub"]), str(r["mask"])))

    meta = [
        "# new vs old gPPI seed overlap — ALL pairs, overlap descending (NA last)",
        "# overlap_pct = min(100*|intersect|/|new|, 100*|intersect|/|old|)",
        "# n_only_new / n_only_old = voxels in one mask but not the other",
        "# if several old files share a name, the LOWEST overlap is kept",
        f"# reason=LOW_OVERLAP if overlap_pct < {THRESH:g}; IDENTICAL if arrays match",
        f"# subjects={len(subjects)}  pairs={len(rows)}  ok={n_ok}  identical={n_same}  "
        f"below_{THRESH:g}={n_low}  missing_old={n_missing}  grid={n_grid}  fail={n_fail}",
        f"# NEW_ROOT={NEW_ROOT}",
        f"# OLD_ROOT={OLD_ROOT}",
        tsv_line(HEADER).rstrip("\n"),
    ]
    lines = list(meta)
    for r in rows:
        lines.append(tsv_line([str(r[k]) for k in HEADER]).rstrip("\n"))
    text = "\n".join(lines) + "\n"
    ALL_OUT.write_text(text, encoding="utf-8")
    LOW_OUT.write_text(text, encoding="utf-8")

    log("")
    log(
        f">> subjects={len(subjects)}  pairs={len(rows)}  ok={n_ok}  "
        f"identical={n_same}  below_{THRESH:g}={n_low}  "
        f"missing_old={n_missing}  grid={n_grid}  fail={n_fail}"
    )
    log(">> wrote ALL overlaps (desc) to:")
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
