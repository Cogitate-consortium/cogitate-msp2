#!/usr/bin/env python3
"""
Walk FSL FEAT derivative folders and list analyses that did not complete successfully.

Primary checks (always on):
  - report.html exists and does not contain 'Error' or 'ERROR' (same heuristic as
    02_run_fsf_feat_analyses.py check_1st_level_feat_dirs)
  - report_log.html: if present, same Error / ERROR scan (detailed FEAT log);
    absence alone does not fail (older layouts may omit it)

Additional heuristics (unless --relaxed):
  - design.fsf should exist (written when FEAT sets up the analysis)
  - *.feat (non-gfeat): stats/ exists and contains at least one *.nii.gz
  - *.gfeat: either cope*.feat subdirectories exist, or stats/*.nii.gz at gfeat root
  - At least MIN_TOPLEVEL_CHILDREN entries directly under the feat dir
  - At least MIN_FILES_DEPTH2 files within depth 2 (catches nearly empty / aborted dirs)

Prints failed paths to stdout and writes the same list to a timestamped file
in the current working directory.

Progress: before and during the scan, status lines go to stderr (index, BIDS
subject id if present in the path, full .feat/.gfeat path) so stdout stays
suitable for redirecting the report only.

Tune thresholds via constants below or env FEAT_CHECK_* (see collect_heuristic_reasons).
Use --level 1st or --level 2nd to scan only run-level 1st-level or analysis-2ndGLM outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


DEFAULT_DERIVATIVES_PATH = (
    "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/"
    "derivatives/fslFeat"
)

# Default root for scanning (override: CLI arg, or env DERIVATIVES_PATH)
derivatives_path = DEFAULT_DERIVATIVES_PATH

# --- Heuristic thresholds (override with env FEAT_CHECK_* as integers) -------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


MIN_TOPLEVEL_CHILDREN = _env_int("FEAT_CHECK_MIN_TOPLEVEL", 4)
"""Typical completed FEAT has report, design.fsf, stats, reg, ..."""


MIN_STATS_NIFTI = _env_int("FEAT_CHECK_MIN_STATS_NIFTI", 1)
"""stats/ should contain cope/zstat images after a finished run."""


MIN_FILES_DEPTH2 = _env_int("FEAT_CHECK_MIN_FILES_DEPTH2", 12)
"""Minimum regular files found within depth 2 (aborted runs are often nearly empty)."""

_SUB_ID = re.compile(r"(sub-[A-Za-z0-9]+)")
_RUN_IN_NAME = re.compile(r"_run-\d+_")


def feat_matches_level(feat_dir: Path, level: str) -> bool:
    """
    Restrict which FEAT directories are scanned.

    - all: no filter
    - 1st: run-level first-level analyses (basename has _run-N_ and 1st-level tag)
    - 2nd: higher-level FEAT whose basename includes analysis-2ndGLM
    """
    lev = (level or "all").strip().lower()
    if lev in ("", "all", "both"):
        return True
    name = feat_dir.name
    namel = name.lower()
    if lev == "1st":
        if "analysis-2ndglm" in namel or "analysis-3rdglm" in namel:
            return False
        if not _RUN_IN_NAME.search(name):
            return False
        return (
            "analysis-1st" in namel
            or "1stglm" in namel
            or "1stroi" in namel
        )
    if lev == "2nd":
        return "analysis-2ndglm" in namel
    return True


def participant_from_path(p: Path) -> str:
    """BIDS subject id (sub-*) if it appears in the path, else empty string."""
    m = _SUB_ID.search(str(p))
    return m.group(1) if m else ""


def iter_feat_directories(root: Path) -> list[Path]:
    """All directories under root whose name ends with .feat or .gfeat."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if name.endswith(".feat") or name.endswith(".gfeat"):
                out.append(Path(dirpath) / name)
        dirnames[:] = [
            d
            for d in dirnames
            if d.endswith(".gfeat") or not d.endswith(".feat")
        ]
    out.sort(key=lambda p: str(p))
    return out


def count_files_max_depth(root: Path, max_depth: int, depth: int = 0) -> int:
    """Count regular files under root up to max_depth (0 = root files only)."""
    n = 0
    if depth > max_depth:
        return 0
    try:
        for child in root.iterdir():
            try:
                if child.is_file():
                    n += 1
                elif child.is_dir():
                    n += count_files_max_depth(child, max_depth, depth + 1)
            except OSError:
                continue
    except OSError:
        pass
    return n


def _html_error_scan(path: Path, label: str) -> list[str]:
    """Read an HTML report and flag Error / ERROR substrings (FEAT failure heuristic)."""
    out: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        out.append(f"cannot read {label} ({exc})")
        return out
    if "Error" in text or "ERROR" in text:
        out.append(f"error reported in {label}")
    return out


def collect_report_reasons(feat_dir: Path) -> list[str]:
    """Checks based on report.html and, when present, report_log.html."""
    reasons: list[str] = []
    report = feat_dir / "report.html"
    if not report.is_file():
        reasons.append("missing report.html")
    else:
        reasons.extend(_html_error_scan(report, "report.html"))

    log = feat_dir / "report_log.html"
    if log.is_file():
        reasons.extend(_html_error_scan(log, "report_log.html"))

    return reasons


def collect_heuristic_reasons(feat_dir: Path) -> list[str]:
    """
    Structural / completeness checks. Suppressed when reasons already include
    missing report (nothing else to inspect meaningfully).
    """
    reasons: list[str] = []
    name = feat_dir.name
    is_gfeat = name.endswith(".gfeat")

    if not (feat_dir / "design.fsf").is_file():
        reasons.append("missing design.fsf")

    try:
        n_child = sum(1 for _ in feat_dir.iterdir())
    except OSError as exc:
        reasons.append(f"cannot list directory ({exc})")
        return reasons

    if n_child < MIN_TOPLEVEL_CHILDREN:
        reasons.append(
            f"few top-level entries ({n_child}; suggest >= {MIN_TOPLEVEL_CHILDREN})"
        )

    file_count_d2 = count_files_max_depth(feat_dir, max_depth=2)
    if file_count_d2 < MIN_FILES_DEPTH2:
        reasons.append(
            f"few files within depth<=2 ({file_count_d2}; suggest >= {MIN_FILES_DEPTH2})"
        )

    if is_gfeat:
        cope_dirs = sorted(feat_dir.glob("cope*.feat"))
        stats_dir = feat_dir / "stats"
        has_top_stats = (
            stats_dir.is_dir() and any(stats_dir.glob("*.nii.gz"))
        )
        if not cope_dirs and not has_top_stats:
            reasons.append(
                "gfeat: no cope*.feat subdirs and no stats/*.nii.gz at gfeat root"
            )
    else:
        stats_dir = feat_dir / "stats"
        if not stats_dir.is_dir():
            reasons.append("missing stats/")
        else:
            n_stat_nii = len(list(stats_dir.glob("*.nii.gz")))
            if n_stat_nii < MIN_STATS_NIFTI:
                reasons.append(
                    f"stats/ has too few .nii.gz ({n_stat_nii}; need >= {MIN_STATS_NIFTI})"
                )

    return reasons


def feat_failure_reasons(feat_dir: Path, *, relaxed: bool) -> list[str]:
    """Return a list of problem strings; empty list means treat as OK."""
    report_reasons = collect_report_reasons(feat_dir)
    if relaxed:
        return report_reasons

    out = list(report_reasons)
    if any("missing report.html" in r for r in report_reasons):
        return out

    out.extend(collect_heuristic_reasons(feat_dir))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List FEAT (.feat/.gfeat) outputs under fslFeat derivatives that failed."
    )
    parser.add_argument(
        "derivatives_path",
        nargs="?",
        default=os.environ.get("DERIVATIVES_PATH", derivatives_path),
        help="Root to scan (default: env DERIVATIVES_PATH or module derivatives_path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: cwd feat_failed_outputs_<timestamp>.txt)",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Only use report.html checks (skip file-count / stats / design.fsf heuristics)",
    )
    parser.add_argument(
        "--level",
        choices=("all", "1st", "2nd"),
        default=os.environ.get("FEAT_CHECK_LEVEL", "all"),
        help=(
            "Which FEAT outputs to include: "
            "1st = run-level first-level (_run-N_ + 1stGLM/1stROI in name); "
            "2nd = analysis-2ndGLM in basename; "
            "all = no filter (default). "
            "Default may be set with env FEAT_CHECK_LEVEL."
        ),
    )
    args = parser.parse_args()

    root = Path(args.derivatives_path).resolve()

    if not root.is_dir():
        print(f"ERROR: derivatives path is not a directory: {root}", file=sys.stderr)
        return 1

    all_feat_dirs = iter_feat_directories(root)
    feat_dirs = [fd for fd in all_feat_dirs if feat_matches_level(fd, args.level)]
    n_filtered = len(all_feat_dirs) - len(feat_dirs)
    n_feat = len(feat_dirs)
    print(
        f"Found {len(all_feat_dirs)} .feat/.gfeat director(y/ies) under {root}; "
        f"after --level {args.level}: {n_feat} to scan"
        + (f" ({n_filtered} excluded by level filter)" if n_filtered else "")
        + ".",
        file=sys.stderr,
        flush=True,
    )
    failed: list[tuple[Path, str]] = []
    for i, fd in enumerate(feat_dirs, start=1):
        who = participant_from_path(fd) or "?"
        print(
            f"  [{i}/{n_feat}] participant {who}  {fd}",
            file=sys.stderr,
            flush=True,
        )
        rs = feat_failure_reasons(fd, relaxed=args.relaxed)
        if rs:
            failed.append((fd, "; ".join(rs)))

    out_path = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd()
        / f"feat_failed_outputs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )

    lines: list[str] = [
        f"# scanned: {root}",
        f"# level filter: {args.level}",
        f"# relaxed mode (report only): {args.relaxed}",
        f"# thresholds: toplevel>={MIN_TOPLEVEL_CHILDREN}, "
        f"files_depth2>={MIN_FILES_DEPTH2}, stats_nii>={MIN_STATS_NIFTI}",
        f"# total .feat/.gfeat directories scanned (after level filter): {len(feat_dirs)}",
        f"# failed / flagged: {len(failed)}",
        "# format: <path> :: <reason>",
        "",
    ]
    for path, reason in failed:
        lines.append(f"{path} :: {reason}")

    text_block = "\n".join(lines) + "\n"

    print(text_block, end="")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text_block, encoding="utf-8")
    print(f"# wrote: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
