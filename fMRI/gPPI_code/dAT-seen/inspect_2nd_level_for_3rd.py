#!/usr/bin/env python3
"""
Inspect gPPI 2nd-level FEAT outputs for problems that can yield all-zero 3rd-level maps.

Checks per subject / analysis (PPI_FFA -> cope3.feat, PPI_LOC -> cope4.feat):
  - parent .gfeat/.feat exists with clean report.html (same gates as 09_make)
  - target copeN.feat exists
  - inner stats/cope1.nii.gz and varcope1.nii.gz present
  - non-zero effect in inner cope (fslstats -R)
  - parent identity reg for 3rd-level gfeatprep (reg/example_func2standard.mat)
  - non-empty subject mask (reduces risk of collapsed group mask at 3rd level)

Writes:
  - inspect_2nd_level_summary.csv
  - inspect_2nd_level_report.txt
  - optional mask-intersection summary when FSL is available
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GLM_DIR = str(_REPO_ROOT / "glm")
if _GLM_DIR not in sys.path:
    sys.path.insert(0, _GLM_DIR)

from gppi_3rd_level import ANALYSES, ANALYSIS_COPE
from report_3rd_level_submission import subjects_from_fsf

SESSION = "ses-V2"
SPACE = "MNI152NLin2009cAsym"
DEFAULT_FILTER_COL = "SYNCHRONY_min_seen"
# |cope1| below this is treated as effectively zero for group FLAME.
DEFAULT_COPE_ABSMAX_THRESH = 1e-6
# Masks smaller than this may erase group intersection when many subjects are pooled.
DEFAULT_MIN_MASK_VOXELS = 1000


@dataclass
class CheckResult:
    sub_code: str
    analysis: str
    cope: int
    parent_path: str = ""
    parent_ext: str = ""
    cope_feat_path: str = ""
    ok_09_make: bool = False
    ok_3rd_strict: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cope1_min: str = ""
    cope1_max: str = ""
    cope1_absmax: str = ""
    mask_voxels: str = ""
    has_parent_reg: bool = False
    has_inner_cope1: bool = False
    has_inner_varcope1: bool = False


def load_subject_csv(path: Path, filter_col: str) -> list[str]:
    df = pd.read_csv(path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    if filter_col not in df.columns:
        raise ValueError(f"missing column {filter_col} in {path}")
    sub_df = df.loc[df[filter_col].astype(str).str.upper().eq("TRUE")]
    return [str(x) for x in sub_df["sub_code"].tolist()]


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def parent_paths(feat_root: Path, sub_code: str, analysis: str) -> list[tuple[Path, str]]:
    subj_id = f"sub-{sub_code}"
    stem = f"{subj_id}_{SESSION}_task-VG_analysis-2ndGLM_{analysis}-{SPACE}"
    base = feat_root / "gPPI" / subj_id
    return [(base / f"{stem}{ext}", ext) for ext in (".gfeat", ".feat")]


def fsl_available() -> bool:
    return shutil.which("fslstats") is not None


def fslstats_range(nii: Path) -> tuple[float | None, float | None, float | None]:
    if not fsl_available() or not nii.is_file():
        return None, None, None
    try:
        out = subprocess.run(
            ["fslstats", str(nii), "-R"],
            check=True,
            capture_output=True,
            text=True,
        )
        parts = out.stdout.strip().split()
        if len(parts) < 2:
            return None, None, None
        lo, hi = float(parts[0]), float(parts[1])
        return lo, hi, max(abs(lo), abs(hi))
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None, None, None


def fslstats_voxels(nii: Path) -> int | None:
    if not fsl_available() or not nii.is_file():
        return None
    try:
        out = subprocess.run(
            ["fslstats", str(nii), "-V"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(float(out.stdout.strip().split()[0]))
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


_BLOCKING_ISSUES = frozenset(
    {
        "MISSING_PARENT_DIR",
        "BAD_PARENT_REPORT",
        "MISSING_COPE_FEAT",
        "WRONG_COPE_FEAT",
        "MISSING_INNER_COPE1",
        "MISSING_INNER_VARCOPE1",
        "ZERO_OR_FLAT_INNER_COPE1",
        "MISSING_PARENT_IDENTITY_REG",
        "EMPTY_MASK",
    }
)


def _finalize_strict(res: CheckResult) -> CheckResult:
    res.ok_3rd_strict = res.ok_09_make and not (set(res.issues) & _BLOCKING_ISSUES)
    return res


def inspect_parent_cope(
    parent: Path,
    cope_feat: Path,
    sub_code: str,
    analysis: str,
    cope: int,
    *,
    parent_ext: str = "",
    cope_absmax_thresh: float,
    min_mask_voxels: int,
) -> CheckResult:
    res = CheckResult(sub_code=sub_code, analysis=analysis, cope=cope)
    res.parent_path = str(parent)
    res.parent_ext = parent_ext
    res.cope_feat_path = str(cope_feat)

    if not parent.is_dir():
        res.issues.append("MISSING_PARENT_DIR")
        return _finalize_strict(res)

    if not report_ok(parent):
        res.issues.append("BAD_PARENT_REPORT")
        return _finalize_strict(res)

    res.ok_09_make = True

    reg_mat = parent / "reg" / "example_func2standard.mat"
    res.has_parent_reg = reg_mat.is_file()
    if not res.has_parent_reg:
        res.issues.append("MISSING_PARENT_IDENTITY_REG")

    if cope_feat.name != f"cope{cope}.feat":
        res.issues.append("WRONG_COPE_FEAT")

    if not cope_feat.is_dir():
        res.issues.append("MISSING_COPE_FEAT")
        res.ok_09_make = False
        return _finalize_strict(res)

    inner_cope = cope_feat / "stats" / "cope1.nii.gz"
    inner_varcope = cope_feat / "stats" / "varcope1.nii.gz"
    res.has_inner_cope1 = inner_cope.is_file()
    res.has_inner_varcope1 = inner_varcope.is_file()

    if not res.has_inner_cope1:
        res.issues.append("MISSING_INNER_COPE1")
    if not res.has_inner_varcope1:
        res.issues.append("MISSING_INNER_VARCOPE1")

    if res.has_inner_cope1:
        lo, hi, absmax = fslstats_range(inner_cope)
        if lo is not None:
            res.cope1_min = f"{lo:g}"
            res.cope1_max = f"{hi:g}"
            res.cope1_absmax = f"{absmax:g}"
            if absmax is not None and absmax < cope_absmax_thresh:
                res.issues.append("ZERO_OR_FLAT_INNER_COPE1")
        else:
            res.warnings.append("COULD_NOT_READ_COPE1_STATS")

    mask_path = cope_feat / "mask"
    if not mask_path.is_file():
        mask_path = cope_feat / "mask.nii.gz"
    if mask_path.is_file():
        nvox = fslstats_voxels(mask_path)
        if nvox is not None:
            res.mask_voxels = str(nvox)
            if nvox == 0:
                res.issues.append("EMPTY_MASK")
            elif nvox < min_mask_voxels:
                res.warnings.append(f"TINY_MASK(<{min_mask_voxels})")
        else:
            res.warnings.append("COULD_NOT_READ_MASK")
    else:
        res.warnings.append("MISSING_MASK_FILE")

    if not (cope_feat / "example_func.nii.gz").is_file():
        example_func = cope_feat / "example_func"
        if not example_func.is_file():
            res.warnings.append("MISSING_EXAMPLE_FUNC")

    return _finalize_strict(res)


def inspect_cope_feat_dir(
    cope_feat: Path,
    analysis: str,
    cope: int,
    *,
    sub_code: str = "",
    cope_absmax_thresh: float = DEFAULT_COPE_ABSMAX_THRESH,
    min_mask_voxels: int = DEFAULT_MIN_MASK_VOXELS,
) -> CheckResult:
    if not sub_code:
        m = re.search(r"sub-([A-Za-z0-9]+)", str(cope_feat))
        sub_code = m.group(1) if m else ""
    parent = cope_feat.parent
    ext = parent.suffix if parent.suffix in (".gfeat", ".feat") else ""
    return inspect_parent_cope(
        parent,
        cope_feat,
        sub_code,
        analysis,
        cope,
        parent_ext=ext,
        cope_absmax_thresh=cope_absmax_thresh,
        min_mask_voxels=min_mask_voxels,
    )


def inspect_subject_cope(
    feat_root: Path,
    sub_code: str,
    analysis: str,
    cope: int,
    *,
    cope_absmax_thresh: float,
    min_mask_voxels: int,
) -> CheckResult:
    parent: Path | None = None
    parent_ext = ""

    for candidate, ext in parent_paths(feat_root, sub_code, analysis):
        if candidate.is_dir():
            parent = candidate
            parent_ext = ext
            break

    if parent is None:
        res = CheckResult(sub_code=sub_code, analysis=analysis, cope=cope)
        res.issues.append("MISSING_PARENT_DIR")
        return _finalize_strict(res)

    cope_feat = parent / f"cope{cope}.feat"
    return inspect_parent_cope(
        parent,
        cope_feat,
        sub_code,
        analysis,
        cope,
        parent_ext=parent_ext,
        cope_absmax_thresh=cope_absmax_thresh,
        min_mask_voxels=min_mask_voxels,
    )


def validate_fsf_strict(
    fsf_path: Path,
    analysis: str,
    *,
    cope_absmax_thresh: float,
    min_mask_voxels: int,
) -> tuple[bool, list[CheckResult]]:
    if analysis not in ANALYSIS_COPE:
        raise ValueError(f"unknown analysis: {analysis}")
    cope = ANALYSIS_COPE[analysis]
    subjects = subjects_from_fsf(fsf_path)
    if not subjects:
        raise ValueError(f"no feat_files in {fsf_path}")

    failures: list[CheckResult] = []
    for sub_id, path_str in subjects:
        cope_feat = Path(path_str)
        sub_code = sub_id.replace("sub-", "", 1) if sub_id.startswith("sub-") else sub_id
        res = inspect_cope_feat_dir(
            cope_feat,
            analysis,
            cope,
            sub_code=sub_code,
            cope_absmax_thresh=cope_absmax_thresh,
            min_mask_voxels=min_mask_voxels,
        )
        if not res.ok_3rd_strict:
            failures.append(res)
    return len(failures) == 0, failures


def write_csv(path: Path, rows: list[CheckResult]) -> None:
    fieldnames = [
        "sub_code",
        "analysis",
        "cope",
        "ok_09_make",
        "ok_3rd_strict",
        "issues",
        "warnings",
        "parent_path",
        "cope_feat_path",
        "has_parent_reg",
        "has_inner_cope1",
        "has_inner_varcope1",
        "cope1_min",
        "cope1_max",
        "cope1_absmax",
        "mask_voxels",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "sub_code": r.sub_code,
                    "analysis": r.analysis,
                    "cope": r.cope,
                    "ok_09_make": int(r.ok_09_make),
                    "ok_3rd_strict": int(r.ok_3rd_strict),
                    "issues": ";".join(r.issues),
                    "warnings": ";".join(r.warnings),
                    "parent_path": r.parent_path,
                    "cope_feat_path": r.cope_feat_path,
                    "has_parent_reg": int(r.has_parent_reg),
                    "has_inner_cope1": int(r.has_inner_cope1),
                    "has_inner_varcope1": int(r.has_inner_varcope1),
                    "cope1_min": r.cope1_min,
                    "cope1_max": r.cope1_max,
                    "cope1_absmax": r.cope1_absmax,
                    "mask_voxels": r.mask_voxels,
                }
            )


def write_report(
    path: Path,
    rows: list[CheckResult],
    *,
    subjects: list[str],
    filter_col: str,
    feat_root: Path,
    cope_absmax_thresh: float,
    min_mask_voxels: int,
    fsl_ok: bool,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# gPPI 2nd-level inspection for 3rd-level group FEAT",
        f"# date={stamp}",
        f"# filter={filter_col}",
        f"# n_subjects_csv={len(subjects)}",
        f"# feat_root={feat_root}",
        f"# cope_absmax_thresh={cope_absmax_thresh}",
        f"# min_mask_voxels_warn={min_mask_voxels}",
        f"# fslstats_available={fsl_ok}",
        "",
        "# Issue codes (blocking for strict 3rd-level):",
        "#   MISSING_PARENT_DIR, BAD_PARENT_REPORT, MISSING_COPE_FEAT",
        "#   MISSING_INNER_COPE1, MISSING_INNER_VARCOPE1, ZERO_OR_FLAT_INNER_COPE1",
        "#   MISSING_PARENT_IDENTITY_REG, EMPTY_MASK",
        "# Warnings: TINY_MASK, MISSING_MASK_FILE, MISSING_EXAMPLE_FUNC, COULD_NOT_READ_*",
        "",
    ]

    for analysis in ANALYSES:
        cope = ANALYSIS_COPE[analysis]
        subset = [r for r in rows if r.analysis == analysis]
        n_strict = sum(1 for r in subset if r.ok_3rd_strict)
        n_make = sum(1 for r in subset if r.ok_09_make)
        lines.extend(
            [
                f"## analysis={analysis}\tcope=cope{cope}",
                f"# ok_09_make={n_make}/{len(subjects)}",
                f"# ok_3rd_strict={n_strict}/{len(subjects)}",
                "",
            ]
        )

        def section(title: str, pred) -> None:
            lines.append(f"### {title}")
            matched = [r for r in subset if pred(r)]
            if not matched:
                lines.append("(none)")
            else:
                for r in sorted(matched, key=lambda x: x.sub_code):
                    extra = []
                    if r.cope1_absmax:
                        extra.append(f"absmax={r.cope1_absmax}")
                    if r.mask_voxels:
                        extra.append(f"mask_vox={r.mask_voxels}")
                    if r.issues:
                        extra.append("issues=" + ",".join(r.issues))
                    if r.warnings:
                        extra.append("warn=" + ",".join(r.warnings))
                    suffix = ("\t" + "\t".join(extra)) if extra else ""
                    lines.append(f"sub-{r.sub_code}{suffix}")
            lines.append("")

        section("Strict OK for 3rd-level (recommended inputs)", lambda r: r.ok_3rd_strict)
        section(
            "Passes 09_make gates but has 3rd-level risk flags",
            lambda r: r.ok_09_make and not r.ok_3rd_strict,
        )
        section(
            "Fails 09_make gates (do not include in 3rd-level FSF)",
            lambda r: not r.ok_09_make,
        )
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def simulate_mask_intersection(
    strict_rows: list[CheckResult],
    out_dir: Path,
    analysis: str,
) -> list[str]:
    """Optional: fslmaths -Tmin across subject masks (mimics 3rd-level gfeatprep)."""
    if not fsl_available():
        return ["# mask intersection: skipped (fslstats not in PATH)"]

    masks = []
    for r in strict_rows:
        if not r.ok_3rd_strict or not r.cope_feat_path:
            continue
        cope_feat = Path(r.cope_feat_path)
        mask = cope_feat / "mask"
        if not mask.is_file():
            mask = cope_feat / "mask.nii.gz"
        if mask.is_file():
            masks.append(mask)

    lines = [f"## mask_intersection_simulation analysis={analysis} n_masks={len(masks)}"]
    if len(masks) < 2:
        lines.append("# need >=2 strict-OK masks to simulate intersection")
        return lines

    work = out_dir / f"_mask_sim_{analysis}"
    work.mkdir(parents=True, exist_ok=True)
    out_mask = work / "group_mask_intersection.nii.gz"

    try:
        if len(masks) == 2:
            subprocess.run(
                ["fslmaths", str(masks[0]), "-mul", str(masks[1]), str(out_mask)],
                check=True,
                capture_output=True,
            )
        else:
            merged = work / "masks_merged.nii.gz"
            subprocess.run(
                ["fslmerge", "-t", str(merged)] + [str(m) for m in masks],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["fslmaths", str(merged), "-Tmin", str(out_mask)],
                check=True,
                capture_output=True,
            )
        nvox = fslstats_voxels(out_mask)
        lo, hi, _ = fslstats_range(out_mask)
        lines.append(f"# simulated_mask={out_mask}")
        lines.append(f"# intersection_voxels={nvox}")
        lines.append(f"# intersection_range={lo},{hi}")
        if nvox is not None and nvox == 0:
            lines.append("# WARNING: intersection empty — group 3rd-level likely all zeros")
        elif nvox is not None and nvox < 1000:
            lines.append("# WARNING: very small intersection — check registration / masks")
    except (subprocess.CalledProcessError, OSError) as exc:
        lines.append(f"# simulation failed: {exc}")

    return lines


def _add_common_threshold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cope-absmax-thresh", type=float, default=DEFAULT_COPE_ABSMAX_THRESH)
    p.add_argument("--min-mask-voxels", type=int, default=DEFAULT_MIN_MASK_VOXELS)


def cmd_scan(args: argparse.Namespace) -> int:
    if not args.bids_root or not args.subject_csv:
        print("ERROR: --bids-root and --subject-csv required", file=sys.stderr)
        return 1

    bids_root = Path(args.bids_root)
    subject_csv = Path(args.subject_csv)
    if args.feat_root:
        feat_root = Path(args.feat_root)
    else:
        feat_root = bids_root / "derivatives/fslFeat"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent / "3rd level report"

    out_dir.mkdir(parents=True, exist_ok=True)
    fsl_ok = fsl_available()

    subjects = load_subject_csv(subject_csv, args.filter_col)
    rows: list[CheckResult] = []

    for sub_code in subjects:
        for analysis in ANALYSES:
            cope = ANALYSIS_COPE[analysis]
            rows.append(
                inspect_subject_cope(
                    feat_root,
                    sub_code,
                    analysis,
                    cope,
                    cope_absmax_thresh=args.cope_absmax_thresh,
                    min_mask_voxels=args.min_mask_voxels,
                )
            )

    csv_path = out_dir / "inspect_2nd_level_summary.csv"
    report_path = out_dir / "inspect_2nd_level_report.txt"
    write_csv(csv_path, rows)
    write_report(
        report_path,
        rows,
        subjects=subjects,
        filter_col=args.filter_col,
        feat_root=feat_root,
        cope_absmax_thresh=args.cope_absmax_thresh,
        min_mask_voxels=args.min_mask_voxels,
        fsl_ok=fsl_ok,
    )

    if args.simulate_mask_intersection and fsl_ok:
        extra: list[str] = []
        for analysis in ANALYSES:
            subset = [r for r in rows if r.analysis == analysis]
            extra.extend(simulate_mask_intersection(subset, out_dir, analysis))
            extra.append("")
        with report_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(extra))

    for analysis in ANALYSES:
        subset = [r for r in rows if r.analysis == analysis]
        n_strict = sum(1 for r in subset if r.ok_3rd_strict)
        n_make = sum(1 for r in subset if r.ok_09_make)
        print(f">> {analysis}: ok_09_make={n_make}/{len(subjects)} ok_3rd_strict={n_strict}/{len(subjects)}")

    print(f">> wrote {csv_path}")
    print(f">> wrote {report_path}")
    if not fsl_ok:
        print("WARN: fslstats not in PATH — load FSL for cope/mask quantification", file=sys.stderr)

    return 0


def cmd_validate_fsf(args: argparse.Namespace) -> int:
    fsf_path = Path(args.fsf)
    if not fsf_path.is_file():
        print(f"ERROR: FSF not found: {fsf_path}", file=sys.stderr)
        return 1

    try:
        ok, failures = validate_fsf_strict(
            fsf_path,
            args.analysis,
            cope_absmax_thresh=args.cope_absmax_thresh,
            min_mask_voxels=args.min_mask_voxels,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_subjects = len(subjects_from_fsf(fsf_path))
    if ok:
        print(f">> strict OK: {args.analysis} n={n_subjects} fsf={fsf_path.name}")
        return 0

    print(
        f"ERROR: strict 2nd-level preflight failed for {args.analysis} "
        f"({len(failures)}/{n_subjects} subjects in FSF)",
        file=sys.stderr,
    )
    for res in sorted(failures, key=lambda r: r.sub_code):
        extra = []
        if res.cope1_absmax:
            extra.append(f"absmax={res.cope1_absmax}")
        if res.mask_voxels:
            extra.append(f"mask_vox={res.mask_voxels}")
        suffix = (" " + " ".join(extra)) if extra else ""
        print(
            f"  sub-{res.sub_code}: {','.join(res.issues)}{suffix}",
            file=sys.stderr,
        )
        if res.warnings:
            print(f"    warn: {','.join(res.warnings)}", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Inspect gPPI 2nd-level outputs for 3rd-level zero-map risk"
    )
    sub = p.add_subparsers(dest="command")

    p_scan = sub.add_parser(
        "scan",
        help="scan cohort and write inspect_2nd_level_summary.csv (default)",
    )
    p_scan.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p_scan.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p_scan.add_argument("--filter-col", default=os.environ.get("FILTER_COL", DEFAULT_FILTER_COL))
    p_scan.add_argument("--feat-root", default=os.environ.get("FSLFEAT_ROOT", ""))
    p_scan.add_argument("--out-dir", default=os.environ.get("GPPI_REPORT_DIR", ""))
    p_scan.add_argument(
        "--simulate-mask-intersection",
        action="store_true",
        help="run fslmaths -Tmin across strict-OK subject masks per analysis",
    )
    _add_common_threshold_args(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_val = sub.add_parser(
        "validate-fsf",
        help="strict-check every subject listed in a 3rd-level FSF (for 10_run)",
    )
    p_val.add_argument("--fsf", required=True)
    p_val.add_argument("--analysis", required=True, choices=ANALYSES)
    _add_common_threshold_args(p_val)
    p_val.set_defaults(func=cmd_validate_fsf)

    # Default command: scan (backward compatible with inspect_2nd_level_for_3rd.sh)
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["scan", *argv]
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
