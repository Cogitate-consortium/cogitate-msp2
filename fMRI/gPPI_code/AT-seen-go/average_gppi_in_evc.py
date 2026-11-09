#!/usr/bin/env python3
"""
ROI-mean 2nd-level gPPI (FFA → cope3, LOC → cope4) inside subject EVC masks.

Writes one CSV from inner stats/cope1 and one from stats/zstat1 per subject.
Prefer .gfeat over .feat (consistent with 09_make / inspect_2nd_level_for_3rd).

Outputs (default under derivatives/gppi/):
  - gppi_evc_roi_means_cope_{filter_col}.csv
  - gppi_evc_roi_means_zstat_{filter_col}.csv
  - gppi_evc_averaging_inclusion_report.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gppi_evc_paths import (
    DEFAULT_FILTER_COL,
    load_subjects,
    resolve_2nd_stat,
)
from gppi_3rd_level import ANALYSIS_COPE

EVC_MASK_SUFFIX = "_evc_300_V1V2.nii.gz"

STAT_SPECS: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("cope1", "cope", ("FFA_cope", "LOC_cope")),
    ("zstat1", "zstat", ("FFA_zstat", "LOC_zstat")),
)


@dataclass
class RowResult:
    sub_code: str
    ffa_mean: str = ""
    loc_mean: str = ""
    status: str = "ok"
    reason: str = ""
    ffa_path: str = ""
    loc_path: str = ""
    evc_mask: str = ""


@dataclass
class TableRun:
    stat_file: str
    label: str
    col_ffa: str
    col_loc: str
    rows: list[RowResult] = field(default_factory=list)
    included: list[RowResult] = field(default_factory=list)
    excluded: list[RowResult] = field(default_factory=list)


def fsl_available() -> bool:
    return shutil.which("fslstats") is not None


def evc_mask_path(evc_root: Path, sub_code: str) -> Path:
    subj_id = f"sub-{sub_code}"
    return evc_root / subj_id / f"{subj_id}{EVC_MASK_SUFFIX}"


def fslstats_masked_mean(image: Path, mask: Path) -> float | None:
    if not fsl_available():
        return None
    try:
        out = subprocess.run(
            ["fslstats", str(image), "-k", str(mask), "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = out.stdout.strip()
        if not value:
            return None
        return float(value)
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def process_subject(
    sub_code: str,
    *,
    feat_root: Path,
    evc_root: Path,
    stat_name: str,
) -> RowResult:
    res = RowResult(sub_code=sub_code)
    mask = evc_mask_path(evc_root, sub_code)
    res.evc_mask = str(mask)

    if not mask.is_file():
        res.status = "excluded"
        res.reason = f"missing EVC mask: {mask}"
        return res

    ffa_nii = resolve_2nd_stat(feat_root, sub_code, "PPI_FFA", ANALYSIS_COPE["PPI_FFA"], stat_name)
    loc_nii = resolve_2nd_stat(feat_root, sub_code, "PPI_LOC", ANALYSIS_COPE["PPI_LOC"], stat_name)

    if ffa_nii is None:
        res.status = "excluded"
        res.reason = f"missing FFA 2nd-level map (cope{ANALYSIS_COPE['PPI_FFA']}/{stat_name})"
        return res
    if loc_nii is None:
        res.status = "excluded"
        res.reason = f"missing LOC 2nd-level map (cope{ANALYSIS_COPE['PPI_LOC']}/{stat_name})"
        res.ffa_path = str(ffa_nii)
        return res

    res.ffa_path = str(ffa_nii)
    res.loc_path = str(loc_nii)

    if not fsl_available():
        res.status = "excluded"
        res.reason = "fslstats not in PATH"
        return res

    ffa_mean = fslstats_masked_mean(ffa_nii, mask)
    loc_mean = fslstats_masked_mean(loc_nii, mask)

    if ffa_mean is None:
        res.status = "excluded"
        res.reason = "fslstats failed for FFA map"
        return res
    if loc_mean is None:
        res.status = "excluded"
        res.reason = "fslstats failed for LOC map"
        return res

    res.ffa_mean = f"{ffa_mean:g}"
    res.loc_mean = f"{loc_mean:g}"
    return res


def run_table(
    subjects: list[str],
    *,
    feat_root: Path,
    evc_root: Path,
    stat_file: str,
    label: str,
    col_ffa: str,
    col_loc: str,
) -> TableRun:
    run = TableRun(stat_file=stat_file, label=label, col_ffa=col_ffa, col_loc=col_loc)
    for sub_code in subjects:
        row = process_subject(
            sub_code,
            feat_root=feat_root,
            evc_root=evc_root,
            stat_name=stat_file,
        )
        run.rows.append(row)
        if row.status == "ok":
            run.included.append(row)
        else:
            run.excluded.append(row)
    return run


def write_csv(path: Path, run: TableRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["sub_code", run.col_ffa, run.col_loc])
        for row in run.included:
            writer.writerow([row.sub_code, row.ffa_mean, row.loc_mean])


def write_inclusion_report(
    path: Path,
    *,
    filter_col: str,
    subjects: list[str],
    runs: list[TableRun],
    feat_root: Path,
    evc_root: Path,
    fsl_ok: bool,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# gPPI EVC ROI averaging — inclusion report",
        f"# date={stamp}",
        f"# filter={filter_col}",
        f"# n_subjects_csv={len(subjects)}",
        f"# feat_root={feat_root}",
        f"# evc_rois_root={evc_root}",
        f"# fslstats_available={fsl_ok}",
        "",
    ]

    for run in runs:
        lines.extend(
            [
                f"## stat={run.label}\tfile={run.stat_file}",
                f"# included={len(run.included)}/{len(subjects)}",
                f"# excluded={len(run.excluded)}/{len(subjects)}",
                "",
                "### Included",
            ]
        )
        if not run.included:
            lines.append("(none)")
        else:
            for row in run.included:
                lines.append(
                    f"sub-{row.sub_code}\t{run.col_ffa}={row.ffa_mean}\t"
                    f"{run.col_loc}={row.loc_mean}"
                )
        lines.extend(["", "### Excluded"])
        if not run.excluded:
            lines.append("(none)")
        else:
            for row in sorted(run.excluded, key=lambda r: r.sub_code):
                extra = []
                if row.ffa_path:
                    extra.append(f"ffa={row.ffa_path}")
                if row.loc_path:
                    extra.append(f"loc={row.loc_path}")
                suffix = ("\t" + "\t".join(extra)) if extra else ""
                lines.append(f"sub-{row.sub_code}\t{row.reason}{suffix}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Mean 2nd-level gPPI inside EVC ROIs (cope + zstat tables)")
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p.add_argument("--filter-col", default=os.environ.get("FILTER_COL", DEFAULT_FILTER_COL))
    p.add_argument("--feat-root", default=os.environ.get("FSLFEAT_ROOT", ""))
    p.add_argument("--evc-rois-root", default=os.environ.get("EVC_ROIS_ROOT", ""))
    p.add_argument("--out-dir", default=os.environ.get("GPPI_EVC_OUT_DIR", ""))
    args = p.parse_args()

    if not args.bids_root:
        print("ERROR: --bids-root required", file=sys.stderr)
        return 1

    bids_root = Path(args.bids_root)
    code_path = bids_root / "derivatives/fMRI_exp2"

    subject_csv = Path(args.subject_csv) if args.subject_csv else code_path / "ses-v2-analysis-subs-fmri.csv"
    feat_root = Path(args.feat_root) if args.feat_root else bids_root / "derivatives/fslFeat"
    evc_root = Path(args.evc_rois_root) if args.evc_rois_root else bids_root / "derivatives/evc_rois"
    out_dir = Path(args.out_dir) if args.out_dir else bids_root / "derivatives/gppi"

    if not subject_csv.is_file():
        print(f"ERROR: subject CSV not found: {subject_csv}", file=sys.stderr)
        return 1

    fsl_ok = fsl_available()
    if not fsl_ok:
        print("WARN: fslstats not in PATH — load FSL before running", file=sys.stderr)

    subjects = load_subjects(subject_csv, args.filter_col)
    if not subjects:
        print(f"ERROR: no subjects after filter {args.filter_col}", file=sys.stderr)
        return 1

    print(f">> participants: {len(subjects)} (filter={args.filter_col})")

    runs: list[TableRun] = []
    for stat_file, label, (col_ffa, col_loc) in STAT_SPECS:
        run = run_table(
            subjects,
            feat_root=feat_root,
            evc_root=evc_root,
            stat_file=stat_file,
            label=label,
            col_ffa=col_ffa,
            col_loc=col_loc,
        )
        runs.append(run)
        out_csv = out_dir / f"gppi_evc_roi_means_{label}_{args.filter_col}.csv"
        write_csv(out_csv, run)
        print(
            f">> {label}: wrote {len(run.included)} rows to {out_csv} "
            f"(excluded {len(run.excluded)})"
        )

    report_path = out_dir / "gppi_evc_averaging_inclusion_report.txt"
    write_inclusion_report(
        report_path,
        filter_col=args.filter_col,
        subjects=subjects,
        runs=runs,
        feat_root=feat_root,
        evc_root=evc_root,
        fsl_ok=fsl_ok,
    )
    print(f">> inclusion report: {report_path}")

    if not fsl_ok:
        return 1
    if all(not run.included for run in runs):
        print("ERROR: no subjects included in any table", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
