#!/usr/bin/env python3
"""
Copy 2nd-level gPPI stat maps needed for EVC ROI tables (cope + zstat).

Per subject (SYNCHRONY_min_seen by default), copies inner stats from:
  PPI_FFA → cope3.feat/stats/cope1.nii.gz, zstat1.nii.gz
  PPI_LOC → cope4.feat/stats/cope1.nii.gz, zstat1.nii.gz

Default destination:
  .../derivatives/gppi/2nd_lvl/sub-{code}/{analysis}_cope{N}_{stat}.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gppi_3rd_level import ANALYSIS_COPE
from gppi_evc_paths import (
    DEFAULT_FILTER_COL,
    EVC_GLM_ANALYSES,
    EVC_STAT_FILES,
    dest_basename,
    load_subjects,
    resolve_2nd_stat,
)

DEFAULT_DEST = Path(
    "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
    "/derivatives/gppi/2nd_lvl"
)


@dataclass
class CopyRecord:
    sub_code: str
    glm_analysis: str
    cope_num: int
    stat_name: str
    src: str
    dest: str
    status: str
    message: str = ""


def iter_maps(feat_root: Path, sub_code: str):
    for glm_analysis in EVC_GLM_ANALYSES:
        cope_num = ANALYSIS_COPE[glm_analysis]
        for stat_name in EVC_STAT_FILES:
            yield glm_analysis, cope_num, stat_name, resolve_2nd_stat(
                feat_root, sub_code, glm_analysis, cope_num, stat_name
            )


def copy_maps(
    subjects: list[str],
    *,
    feat_root: Path,
    dest_root: Path,
    dry_run: bool,
) -> list[CopyRecord]:
    records: list[CopyRecord] = []
    dest_root.mkdir(parents=True, exist_ok=True)

    for sub_code in subjects:
        subj_id = f"sub-{sub_code}"
        sub_dest = dest_root / subj_id

        for glm_analysis, cope_num, stat_name, src in iter_maps(feat_root, sub_code):
            name = dest_basename(glm_analysis, cope_num, stat_name)
            dest = sub_dest / name

            if src is None:
                records.append(
                    CopyRecord(
                        sub_code=sub_code,
                        glm_analysis=glm_analysis,
                        cope_num=cope_num,
                        stat_name=stat_name,
                        src="",
                        dest=str(dest),
                        status="missing",
                        message="source not found",
                    )
                )
                continue

            if dest.is_file() and not dry_run:
                try:
                    if src.stat().st_ino == dest.stat().st_ino and src.resolve() == dest.resolve():
                        records.append(
                            CopyRecord(
                                sub_code=sub_code,
                                glm_analysis=glm_analysis,
                                cope_num=cope_num,
                                stat_name=stat_name,
                                src=str(src),
                                dest=str(dest),
                                status="skipped",
                                message="already same file",
                            )
                        )
                        continue
                except OSError:
                    pass

            if dry_run:
                records.append(
                    CopyRecord(
                        sub_code=sub_code,
                        glm_analysis=glm_analysis,
                        cope_num=cope_num,
                        stat_name=stat_name,
                        src=str(src),
                        dest=str(dest),
                        status="dry_run",
                    )
                )
                continue

            sub_dest.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dest)
            except OSError as exc:
                records.append(
                    CopyRecord(
                        sub_code=sub_code,
                        glm_analysis=glm_analysis,
                        cope_num=cope_num,
                        stat_name=stat_name,
                        src=str(src),
                        dest=str(dest),
                        status="error",
                        message=str(exc),
                    )
                )
            else:
                records.append(
                    CopyRecord(
                        sub_code=sub_code,
                        glm_analysis=glm_analysis,
                        cope_num=cope_num,
                        stat_name=stat_name,
                        src=str(src),
                        dest=str(dest),
                        status="copied",
                    )
                )

    return records


def write_manifest(path: Path, records: list[CopyRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sub_code",
                "glm_analysis",
                "cope_num",
                "stat_name",
                "status",
                "src",
                "dest",
                "message",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "sub_code": rec.sub_code,
                    "glm_analysis": rec.glm_analysis,
                    "cope_num": rec.cope_num,
                    "stat_name": rec.stat_name,
                    "status": rec.status,
                    "src": rec.src,
                    "dest": rec.dest,
                    "message": rec.message,
                }
            )


def write_summary(path: Path, records: list[CopyRecord], **meta: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    by_status: dict[str, int] = {}
    for rec in records:
        by_status[rec.status] = by_status.get(rec.status, 0) + 1

    n_subjects = len({r.sub_code for r in records})
    expected_per_sub = len(EVC_GLM_ANALYSES) * len(EVC_STAT_FILES)
    complete = sum(
        1
        for sub in {r.sub_code for r in records}
        if sum(1 for r in records if r.sub_code == sub and r.status in ("copied", "skipped", "dry_run"))
        == expected_per_sub
    )

    lines = [
        "# gPPI 2nd-level stat copy for EVC tables",
        f"# date={stamp}",
        *(f"# {k}={v}" for k, v in meta.items()),
        f"# n_subjects={n_subjects}",
        f"# maps_per_subject={expected_per_sub}",
        f"# subjects_all_maps_ok={complete}",
        "",
        "# status counts",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"{status}={count}")
    lines.append("")
    lines.append("# missing / error by subject")
    for sub in sorted({r.sub_code for r in records}):
        bad = [
            r
            for r in records
            if r.sub_code == sub and r.status in ("missing", "error")
        ]
        if bad:
            lines.append(f"sub-{sub}")
            for r in bad:
                lines.append(f"  {r.glm_analysis} cope{r.cope_num} {r.stat_name}: {r.status} {r.message}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Copy 2nd-level gPPI cope/zstat maps for EVC ROI averaging"
    )
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p.add_argument("--filter-col", default=os.environ.get("FILTER_COL", DEFAULT_FILTER_COL))
    p.add_argument("--feat-root", default=os.environ.get("FSLFEAT_ROOT", ""))
    p.add_argument("--dest-dir", default=os.environ.get("GPPI_2ND_LVL_DEST", ""))
    p.add_argument("--dry-run", action="store_true", help="list copies without writing files")
    args = p.parse_args()

    if not args.bids_root:
        print("ERROR: --bids-root required", file=sys.stderr)
        return 1

    bids_root = Path(args.bids_root)
    code_path = bids_root / "derivatives/fMRI_exp2"
    subject_csv = Path(args.subject_csv) if args.subject_csv else code_path / "ses-v2-analysis-subs-fmri.csv"
    feat_root = Path(args.feat_root) if args.feat_root else bids_root / "derivatives/fslFeat"
    dest_root = Path(args.dest_dir) if args.dest_dir else DEFAULT_DEST

    if not subject_csv.is_file():
        print(f"ERROR: subject CSV not found: {subject_csv}", file=sys.stderr)
        return 1

    subjects = load_subjects(subject_csv, args.filter_col)
    if not subjects:
        print(f"ERROR: no subjects for filter {args.filter_col}", file=sys.stderr)
        return 1

    mode = "dry-run" if args.dry_run else "copy"
    print(f">> {mode}: {len(subjects)} subjects → {dest_root}")

    records = copy_maps(subjects, feat_root=feat_root, dest_root=dest_root, dry_run=args.dry_run)

    manifest = dest_root / "copy_manifest.csv"
    summary = dest_root / "copy_summary.txt"
    write_manifest(manifest, records)
    write_summary(
        summary,
        records,
        filter_col=args.filter_col,
        feat_root=str(feat_root),
        dest_dir=str(dest_root),
    )

    n_copied = sum(1 for r in records if r.status == "copied")
    n_missing = sum(1 for r in records if r.status == "missing")
    n_error = sum(1 for r in records if r.status == "error")
    print(f">> copied={n_copied} missing={n_missing} error={n_error}")
    print(f">> manifest: {manifest}")
    print(f">> summary: {summary}")

    if n_missing or n_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
