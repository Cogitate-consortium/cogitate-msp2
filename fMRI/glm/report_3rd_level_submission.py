#!/usr/bin/env python3
"""
Record which subjects are included in each submitted (or skipped) 3rd-level FEAT job.

Reads subject lists from ``included_subjects_report.txt`` (written by
``*_make_fsf_files_3rdlevel.sh``) when possible, otherwise parses ``feat_files``
paths from the chosen FSF.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def subjects_from_fsf(fsf_path: Path) -> list[tuple[str, str]]:
    """Return [(sub_id, cope_feat_path), ...] in feat_files order."""
    out: list[tuple[str, str]] = []
    for line in fsf_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("set feat_files("):
            continue
        brace_m = re.match(r"^set feat_files\(\d+\) \{(.+)\}\s*$", line)
        if brace_m:
            path = brace_m.group(1)
        elif '"' in line:
            path = line.split('"', 2)[1]
        else:
            continue
        m = _SUB_RE.search(path)
        if m:
            out.append((m.group(1), path))
    return out


def subjects_from_inclusion_report(
    report_path: Path,
    *,
    analysis: str,
    cope: str | None = None,
) -> list[tuple[str, str]]:
    """Parse one ## analysis=... section from 05/08 included_subjects_report.txt."""
    if not report_path.is_file():
        return []

    lines = report_path.read_text(encoding="utf-8").splitlines()
    in_section = False
    in_included = False
    out: list[tuple[str, str]] = []

    for line in lines:
        if line.startswith("## analysis="):
            in_section = analysis in line and (
                cope is None or f"cope={cope}" in line
            )
            in_included = False
            continue
        if not in_section:
            continue
        if line.startswith("# Included subjects"):
            in_included = True
            continue
        if line.startswith("# Excluded subjects") or line.startswith("## "):
            in_included = False
            if line.startswith("## "):
                in_section = False
            continue
        if in_included and line.strip():
            parts = line.split("\t", 1)
            subj = parts[0].strip()
            path = parts[1].strip() if len(parts) > 1 else ""
            if subj.startswith("sub-"):
                out.append((subj, path))

    return out


def resolve_subjects(
    *,
    fsf_file: Path | None,
    inclusion_report: Path | None,
    analysis: str,
    cope: str | None,
) -> list[tuple[str, str]]:
    if inclusion_report is not None:
        from_report = subjects_from_inclusion_report(
            inclusion_report, analysis=analysis, cope=cope
        )
        if from_report:
            return from_report
    if fsf_file is not None and fsf_file.is_file():
        return subjects_from_fsf(fsf_file)
    return []


def init_report(report_path: Path, *, title: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    report_path.write_text(
        f"# {title}\n# date={stamp}\n\n",
        encoding="utf-8",
    )


def append_submission_section(
    report_path: Path,
    *,
    pipeline: str,
    analysis: str,
    cope: str | None,
    fsf_file: Path | None,
    inclusion_report: Path | None,
    submission_status: str,
) -> list[tuple[str, str]]:
    subjects = resolve_subjects(
        fsf_file=fsf_file,
        inclusion_report=inclusion_report,
        analysis=analysis,
        cope=cope,
    )
    sub_ids = [s for s, _ in subjects]
    inc_ids = ",".join(s.replace("sub-", "", 1) for s in sub_ids)

    header = (
        f"## pipeline={pipeline}\tanalysis={analysis}\t"
        f"submission_status={submission_status}"
    )
    if cope:
        header += f"\tcope={cope}"

    lines = [
        header,
        f"n_included={len(subjects)}",
        f"included_subject_ids={inc_ids}",
        f"fsf_file={fsf_file if fsf_file else ''}",
        f"inclusion_report={inclusion_report if inclusion_report else ''}",
        "",
        "# Included subjects",
    ]
    for subj, path in subjects:
        lines.append(f"{subj}\t{path}")
    if not subjects:
        lines.append("(none)")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return subjects


def cmd_init(args: argparse.Namespace) -> int:
    init_report(Path(args.report), title=args.title)
    print(f"init {args.report}")
    return 0


def _print_meta(subjects: list[tuple[str, str]]) -> None:
    sub_ids = [s.replace("sub-", "", 1) for s, _ in subjects]
    print(f"META n_included={len(subjects)}")
    print(f"META included_subject_ids={','.join(sub_ids)}")


def cmd_query(args: argparse.Namespace) -> int:
    fsf = Path(args.fsf) if args.fsf else None
    inc = Path(args.inclusion_report) if args.inclusion_report else None
    cope = args.cope or None
    subjects = resolve_subjects(
        fsf_file=fsf,
        inclusion_report=inc,
        analysis=args.analysis,
        cope=cope,
    )
    _print_meta(subjects)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    fsf = Path(args.fsf) if args.fsf else None
    inc = Path(args.inclusion_report) if args.inclusion_report else None
    cope = args.cope or None
    subjects = append_submission_section(
        Path(args.report),
        pipeline=args.pipeline,
        analysis=args.analysis,
        cope=cope,
        fsf_file=fsf,
        inclusion_report=inc,
        submission_status=args.status,
    )
    _print_meta(subjects)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="3rd-level FEAT submission inclusion reports.")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create/overwrite combined submission report header")
    p_init.add_argument("--report", required=True, type=Path)
    p_init.add_argument("--title", default="3rd-level FEAT submission inclusion report")
    p_init.set_defaults(func=cmd_init)

    p_q = sub.add_parser("query", help="Print META subject counts without writing the report")
    p_q.add_argument("--analysis", required=True)
    p_q.add_argument("--cope", default="")
    p_q.add_argument("--fsf", default="")
    p_q.add_argument("--inclusion-report", default="")
    p_q.set_defaults(func=cmd_query)

    p_rec = sub.add_parser("record", help="Append one analysis/cope submission section")
    p_rec.add_argument("--report", required=True, type=Path)
    p_rec.add_argument("--pipeline", required=True, help="e.g. activation, gPPI")
    p_rec.add_argument("--analysis", required=True)
    p_rec.add_argument("--cope", default="", help="e.g. cope1 (optional for activation)")
    p_rec.add_argument("--fsf", default="", help="Path to FSF submitted (if any)")
    p_rec.add_argument("--inclusion-report", default="", help="Path to 05/08 included_subjects_report.txt")
    p_rec.add_argument(
        "--status",
        required=True,
        help="submitted, no_fsf, skipped_marker, sbatch_failed, ...",
    )
    p_rec.set_defaults(func=cmd_record)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
