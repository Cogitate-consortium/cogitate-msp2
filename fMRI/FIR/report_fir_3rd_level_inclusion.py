#!/usr/bin/env python3
"""
Inclusion report for FIR 3rd-level group FEAT (08/09).

Lists subjects included per variant/cope and traces missing derivatives for exclusions.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from report_fir_roi_inclusion import (
    ROI_VARIANT_CHECKS,
    FirVariantCheck,
    append_structural_trace,
    load_subject_csv,
    report_ok,
    second_level_parent,
    subjects_for_filter,
    trace_first_level,
)

SESSION = "ses-V2"

VARIANT_CFG: dict[str, FirVariantCheck] = {
    cfg.fslfeat_subdir: cfg for cfg in ROI_VARIANT_CHECKS
}


@dataclass
class InclusionSection:
    analysis: str
    cope_tag: str
    included: list[tuple[str, str]] = field(default_factory=list)
    excluded: list[tuple[str, str, str]] = field(default_factory=list)


def parse_inclusion_report(path: Path) -> dict[tuple[str, str], InclusionSection]:
    if not path.is_file():
        return {}
    sections: dict[tuple[str, str], InclusionSection] = {}
    current: InclusionSection | None = None
    mode: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## analysis="):
            parts = dict(p.split("=", 1) for p in line[3:].split("\t") if "=" in p)
            analysis = parts.get("analysis", "")
            cope_tag = parts.get("cope", "")
            current = InclusionSection(analysis=analysis, cope_tag=cope_tag)
            sections[(analysis, cope_tag)] = current
            mode = None
            continue
        if current is None:
            continue
        if line.startswith("# Included subjects"):
            mode = "included"
            continue
        if line.startswith("# Excluded subjects"):
            mode = "excluded"
            continue
        if not line.strip() or line.startswith("#"):
            continue
        if mode == "included":
            parts = line.split("\t", 1)
            subj = parts[0].strip()
            feat_path = parts[1].strip() if len(parts) > 1 else ""
            current.included.append((subj, feat_path))
        elif mode == "excluded":
            parts = line.split("\t", 2)
            if len(parts) >= 2:
                subj, reason = parts[0], parts[1]
                feat_path = parts[2] if len(parts) > 2 else ""
                current.excluded.append((subj, reason, feat_path))

    return sections


def runs_report_path(fir_root: Path, variant: str) -> Path:
    return fir_root / "job_files/run_FIR_2nd_level" / f"included_runs_report_{variant}.txt"


def trace_second_level(
    feat_root: Path,
    bids_root: Path,
    fir_root: Path,
    sub_code: str,
    variant: str,
    cope: int,
) -> list[str]:
    cfg = VARIANT_CFG.get(variant)
    if cfg is None:
        return [f"[config] unknown variant: {variant}"]

    sub_id = f"sub-{sub_code}"
    lines: list[str] = []
    parent, ext = second_level_parent(feat_root, cfg, sub_code)
    cope_path = (parent / f"cope{cope}.feat") if parent else None

    run_marker = (
        fir_root
        / "job_files/run_FIR_2nd_level"
        / variant
        / sub_code
        / f"{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}.run"
    )

    if parent is None:
        lines.append(
            f"[2nd level {variant}] missing: "
            f"{feat_root / cfg.fslfeat_subdir / sub_id}/{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}.gfeat"
        )
        if run_marker.is_file():
            lines.append(f"  2nd-level job marker exists: {run_marker}")
        else:
            lines.append("  2nd-level job marker missing (06_run_FIR_2nd_level_VG.sh?)")
        lines.append("  structural:")
        append_structural_trace(lines, bids_root, sub_code)
        lines.append("  1st-level run status:")
        lines.extend(
            trace_first_level(
                feat_root,
                bids_root,
                fir_root,
                cfg,
                sub_code,
                runs_report_path(fir_root, variant),
            )
        )
        return lines

    if not report_ok(parent):
        lines.append(f"[2nd level {variant}] bad/missing report.html ({ext}): {parent}")
    if cope_path is None or not cope_path.is_dir():
        lines.append(f"[2nd level {variant}] missing cope{cope}.feat under {parent}")
        lines.append("  1st-level run status:")
        lines.extend(
            trace_first_level(
                feat_root,
                bids_root,
                fir_root,
                cfg,
                sub_code,
                runs_report_path(fir_root, variant),
            )
        )
    else:
        lines.append(f"[2nd level {variant}] cope{cope} OK: {cope_path}")
        reg_mat = parent / "reg" / "example_func2standard.mat"
        if not reg_mat.is_file():
            lines.append(
                "[2nd level] missing identity reg at .gfeat root "
                "(07_create_identity_reg_for_3rd_lvl.sh?)"
            )
    return lines


def trace_excluded_subject(
    feat_root: Path,
    bids_root: Path,
    fir_root: Path,
    sub_code: str,
    variant: str,
    cope: int,
    reason_08: str,
    path_08: str,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines = [
        f"### {sub_id}",
        f"08_make_exclusion_reason: {reason_08}",
        f"08_make_path: {path_08}",
        "",
    ]
    lines.extend(trace_second_level(feat_root, bids_root, fir_root, sub_code, variant, cope))
    lines.append("")
    return lines


def submission_status_for(
    submission_report: Path | None, analysis: str, cope_tag: str
) -> str:
    if submission_report is None or not submission_report.is_file():
        return "unknown"
    for line in submission_report.read_text(encoding="utf-8").splitlines():
        if (
            line.startswith("## pipeline=")
            and f"analysis={analysis}" in line
            and f"cope={cope_tag}" in line
        ):
            m = re.search(r"submission_status=(\S+)", line)
            if m:
                return m.group(1)
    return "unknown"


def write_report(
    report_path: Path,
    subject_df: pd.DataFrame,
    filter_col: str,
    variants: list[str],
    copes: list[int],
    inclusion_sections: dict[tuple[str, str], InclusionSection],
    bids_root: Path,
    feat_root: Path,
    fir_root: Path,
    inclusion_report: Path,
    submission_report: Path | None,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# FIR 3rd-level inclusion report",
        f"# date={stamp}",
        f"# filter={filter_col}",
        f"# source_inclusion_report={inclusion_report}",
        f"# source_submission_report={submission_report or ''}",
        "",
    ]

    candidates = subjects_for_filter(subject_df, filter_col)

    for variant in variants:
        for cope in copes:
            cope_tag = f"cope{cope}"
            section = inclusion_sections.get((variant, cope_tag))
            included_subs: list[str] = []
            excluded_map: dict[str, tuple[str, str]] = {}

            if section:
                included_subs = [s.replace("sub-", "", 1) for s, _ in section.included]
                for subj, reason, path in section.excluded:
                    code = subj.replace("sub-", "", 1)
                    excluded_map[code] = (reason, path)
                for subj, _ in section.included:
                    code = subj.replace("sub-", "", 1)
                    excluded_map.pop(code, None)
            else:
                lines.append(
                    f"## analysis={variant}\tcope={cope_tag}\t"
                    "WARNING=no section in 07 inclusion report"
                )

            for code in candidates:
                if code not in included_subs and code not in excluded_map:
                    excluded_map[code] = (
                        "not listed in 07 inclusion report (filtered out or not processed)",
                        "",
                    )

            sub_status = submission_status_for(submission_report, variant, cope_tag)

            lines.extend(
                [
                    "",
                    "=" * 72,
                    f"## analysis={variant}\tcope={cope_tag}",
                    f"filter={filter_col}",
                    f"3rd_level_submission_status={sub_status}",
                    f"n_csv_candidates={len(candidates)}",
                    f"n_included={len(included_subs)}",
                    f"n_excluded={len(excluded_map)}",
                    "",
                    "# Included subjects (in 3rd-level FSF)",
                ]
            )
            if included_subs:
                for code in sorted(included_subs):
                    subj = f"sub-{code}"
                    path = ""
                    if section:
                        path = next(
                            (p for s, p in section.included if s == subj),
                            "",
                        )
                    lines.append(f"{subj}\t{path}")
            else:
                lines.append("(none)")

            lines.extend(["", "# Excluded subjects (trace-back)"])
            if not excluded_map:
                lines.append("(none)")
            else:
                for code in sorted(excluded_map):
                    reason, path = excluded_map[code]
                    lines.extend(
                        trace_excluded_subject(
                            feat_root,
                            bids_root,
                            fir_root,
                            code,
                            variant,
                            cope,
                            reason,
                            path,
                        )
                    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="FIR 3rd-level inclusion report")
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p.add_argument("--fir-root", default=os.environ.get("FIR_CODE_ROOT", ""))
    p.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p.add_argument("--filter-col", default=os.environ.get("FIR_SUBJECT_FILTER_COL", ""))
    p.add_argument("--variants-csv", default=os.environ.get("FIR_VARIANTS_CSV", ""))
    p.add_argument("--copes-csv", default=os.environ.get("FIR_3RD_COPES_CSV", ""))
    p.add_argument("--inclusion-report", default=os.environ.get("INCLUSION_REPORT", ""))
    p.add_argument("--submission-report", default=os.environ.get("SUBMISSION_REPORT", ""))
    p.add_argument("--report-dir", default=os.environ.get("REPORT_DIR", ""))
    args = p.parse_args()

    for name, val in (
        ("bids-root", args.bids_root),
        ("fir-root", args.fir_root),
        ("subject-csv", args.subject_csv),
        ("filter-col", args.filter_col),
        ("inclusion-report", args.inclusion_report),
        ("report-dir", args.report_dir),
    ):
        if not val:
            print(f"ERROR: --{name} is required", file=sys.stderr)
            return 1

    variants = [v.strip() for v in args.variants_csv.split(",") if v.strip()]
    if not variants:
        variants = ["FIR"]
    copes = [int(c) for c in args.copes_csv.split(",") if c.strip()]
    if not copes:
        copes = list(range(1, 43))

    bids_root = Path(args.bids_root)
    fir_root = Path(args.fir_root)
    feat_root = bids_root / "derivatives/fslFeat"
    report_dir = Path(args.report_dir)
    report_path = report_dir / "fir_3rd_level_inclusion_report.txt"

    subject_df = load_subject_csv(Path(args.subject_csv))
    inclusion_sections = parse_inclusion_report(Path(args.inclusion_report))
    submission_report = (
        Path(args.submission_report) if args.submission_report else None
    )

    write_report(
        report_path,
        subject_df,
        args.filter_col,
        variants,
        copes,
        inclusion_sections,
        bids_root,
        feat_root,
        fir_root,
        Path(args.inclusion_report),
        submission_report,
    )
    print(f">> wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
