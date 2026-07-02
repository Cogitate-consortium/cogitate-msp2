#!/usr/bin/env python3
"""
Check 2nd-level FEAT (.gfeat) contrast consistency across subjects before 3rd-level FEAT.

FSL fails with "not all input FEAT directories have the same set of contrasts" when
pooled subjects differ in design.con (contrast names/count/weights) or cope structure.
This script finds outliers relative to the majority fingerprint per analysis.

Usage:
  python check_2nd_level_consistency.py
  python check_2nd_level_consistency.py --analysis seen_face_vs_unseen_face
  python check_2nd_level_consistency.py --inputs activation/fsf_files/group/inputs_seen_face_vs_unseen_face.txt
    (one cope .nii.gz path per line from 06_make_fsf_files_3rdlevel.sh)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BIDS_ROOT = Path("/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids")
DEFAULT_CODE_ROOT = DEFAULT_BIDS_ROOT / "derivatives/fMRI_exp2"
DEFAULT_SUBJECT_CSV = DEFAULT_CODE_ROOT / "ses-v2-analysis-subs-fmri.csv"
DEFAULT_FSLFEAT_ROOT = DEFAULT_BIDS_ROOT / "derivatives/fslFeat"

# label, 2nd-level gfeat suffix, CSV filter column, 3rd-level cope index
ACTIVATION_ANALYSES: tuple[dict[str, str | int], ...] = (
    {
        "label": "seen_face_vs_unseen_face",
        "gfeat_suffix": "analysis-2ndGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym",
        "filter_col": "ACTIVATION_min_sf_uf",
        "cope": 1,
    },
    {
        "label": "seen_object_vs_unseen_object",
        "gfeat_suffix": "analysis-2ndGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym",
        "filter_col": "ACTIVATION_min_so_uo",
        "cope": 2,
    },
)

_CONTRAST_NAME_RE = re.compile(r"^/ContrastName(\d+)\s+(.+?)\s*$")


@dataclass(frozen=True)
class ContrastFingerprint:
    """Hashable summary of a FEAT design.con contrast block."""

    contrast_names: tuple[str, ...]
    matrix_rows: tuple[str, ...]

    @property
    def n_contrasts(self) -> int:
        return len(self.contrast_names)

    def describe(self) -> str:
        parts = [f"n={self.n_contrasts}"]
        for i, name in enumerate(self.contrast_names, start=1):
            row = self.matrix_rows[i - 1] if i - 1 < len(self.matrix_rows) else "?"
            parts.append(f"{i}:{name}({row})")
        return "; ".join(parts)


@dataclass
class SubjectGfeatStatus:
    sub_id: str
    sub_code: str
    gfeat_path: Path | None
    issues: list[str] = field(default_factory=list)
    fingerprint: ContrastFingerprint | None = None
    cope_fingerprint: ContrastFingerprint | None = None
    n_cope_feat_dirs: int = 0
    report_ok: bool = False

    @property
    def ok_for_pooling(self) -> bool:
        return self.gfeat_path is not None and not self.issues and self.fingerprint is not None


def parse_design_con(con_path: Path) -> ContrastFingerprint | None:
    if not con_path.is_file():
        return None
    text = con_path.read_text(encoding="utf-8", errors="replace")
    names_by_idx: dict[int, str] = {}
    for line in text.splitlines():
        m = _CONTRAST_NAME_RE.match(line.strip())
        if m:
            names_by_idx[int(m.group(1))] = m.group(2).strip()
    matrix_rows: list[str] = []
    if "/Matrix" in text:
        after = text.split("/Matrix", 1)[1]
        for line in after.splitlines():
            s = line.strip()
            if not s or s.startswith("/"):
                continue
            matrix_rows.append(s)
    if not names_by_idx:
        return None
    indices = sorted(names_by_idx)
    contrast_names = tuple(names_by_idx[i] for i in indices)
    # Align matrix rows to contrast index when counts match; otherwise keep raw rows.
    if len(matrix_rows) == len(contrast_names):
        matrix_tuple = tuple(matrix_rows)
    elif matrix_rows:
        matrix_tuple = tuple(matrix_rows)
    else:
        matrix_tuple = tuple("?" for _ in contrast_names)
    return ContrastFingerprint(contrast_names=contrast_names, matrix_rows=matrix_tuple)


def report_ok(gfeat_dir: Path) -> bool:
    report = gfeat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def resolve_gfeat(feat_root: Path, sub_code: str, gfeat_suffix: str) -> Path | None:
    subj_id = f"sub-{sub_code}"
    stem = f"{subj_id}_ses-V2_task-VG_{gfeat_suffix}"
    for ext in (".gfeat", ".feat"):
        candidate = feat_root / "activation" / subj_id / f"{stem}{ext}"
        if candidate.is_dir():
            return candidate
    return None


def _read_subject_table(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    raw = csv_path.read_text(encoding="utf-8", errors="replace")
    delimiter = ";" if raw.count(";") > raw.count(",") else ","
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames:
            rows = list(reader)
    if not rows:
        raise SystemExit(f"ERROR: empty or unreadable CSV: {csv_path}")
    return list(rows[0].keys()), rows


def load_subjects(csv_path: Path, filter_col: str) -> list[str]:
    fieldnames, rows = _read_subject_table(csv_path)
    if "sub_code" not in fieldnames:
        raise SystemExit(f"ERROR: missing sub_code column in {csv_path}")
    if filter_col not in fieldnames:
        raise SystemExit(f"ERROR: missing column {filter_col} in {csv_path}")
    out: list[str] = []
    for row in rows:
        if str(row.get(filter_col, "")).strip().upper() == "TRUE":
            code = str(row.get("sub_code", "")).strip()
            if code:
                out.append(code)
    return out


def load_paths_from_inputs(inputs_file: Path) -> list[Path]:
    paths: list[Path] = []
    for line in inputs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def resolve_gfeat_from_input_path(path: Path) -> Path | None:
    """Map a 06 inputs line (copeN.feat, cope .nii.gz, or legacy .gfeat) to the parent .gfeat dir."""
    if path.is_dir() and path.name.startswith("cope") and path.name.endswith(".feat"):
        parent = path.parent
        if parent.is_dir() and parent.name.endswith(".gfeat"):
            return parent
    if path.suffix == ".gz" and path.name.startswith("cope") and path.parent.name == "stats":
        cope_feat = path.parent.parent
        parent = cope_feat.parent
        if parent.is_dir() and parent.name.endswith(".gfeat"):
            return parent
        if parent.is_dir() and parent.name.endswith(".feat"):
            return parent
    if path.is_dir() and path.name.endswith(".gfeat"):
        return path
    return None


def cope_varcope_paths(gfeat_path: Path, cope_idx: int) -> tuple[Path, Path]:
    """Paths under .gfeat/copeN.feat/stats (sub-FEAT uses cope1, not copeN)."""
    stats = gfeat_path / f"cope{cope_idx}.feat/stats"
    cope_nii = stats / "cope1.nii.gz"
    varcope_nii = stats / "varcope1.nii.gz"
    if not cope_nii.is_file():
        cope_nii = stats / f"cope{cope_idx}.nii.gz"
    if not varcope_nii.is_file():
        varcope_nii = stats / f"varcope{cope_idx}.nii.gz"
    return cope_nii, varcope_nii


def inspect_gfeat(
    gfeat_path: Path,
    cope_idx: int,
    *,
    require_report: bool,
) -> SubjectGfeatStatus:
    sub_id = gfeat_path.parent.name
    sub_code = sub_id[4:] if sub_id.startswith("sub-") else sub_id
    status = SubjectGfeatStatus(sub_id=sub_id, sub_code=sub_code, gfeat_path=gfeat_path)

    if require_report:
        status.report_ok = report_ok(gfeat_path)
        if not status.report_ok:
            status.issues.append("bad or missing report.html on .gfeat")

    design_con = gfeat_path / "design.con"
    fp = parse_design_con(design_con)
    if fp is None:
        status.issues.append(f"missing or unreadable {design_con}")
    else:
        status.fingerprint = fp

    cope_dirs = sorted(gfeat_path.glob("cope*.feat"))
    status.n_cope_feat_dirs = len(cope_dirs)
    cope_feat = gfeat_path / f"cope{cope_idx}.feat"
    if not cope_feat.is_dir():
        status.issues.append(f"missing cope{cope_idx}.feat")
    else:
        cope_fp = parse_design_con(cope_feat / "design.con")
        status.cope_fingerprint = cope_fp
        if fp is not None and cope_fp is not None and cope_fp != fp:
            status.issues.append(
                f"cope{cope_idx}.feat design.con differs from parent .gfeat design.con"
            )

    cope_nii, varcope_nii = cope_varcope_paths(gfeat_path, cope_idx)
    if not cope_nii.is_file():
        status.issues.append(f"missing {cope_nii.name}")
    if not varcope_nii.is_file():
        status.issues.append(f"missing {varcope_nii.name}")

    return status


def inspect_input_path(
    input_path: Path,
    cope_idx: int,
    *,
    require_report: bool,
) -> SubjectGfeatStatus:
    gfeat = resolve_gfeat_from_input_path(input_path)
    if gfeat is None:
        sub_id = "unknown"
        m = re.search(r"(sub-[A-Za-z0-9]+)", str(input_path))
        if m:
            sub_id = m.group(1)
        sub_code = sub_id[4:] if sub_id.startswith("sub-") else sub_id
        return SubjectGfeatStatus(
            sub_id=sub_id,
            sub_code=sub_code,
            gfeat_path=None,
            issues=[f"cannot resolve 2nd-level .gfeat from input path: {input_path}"],
        )
    status = inspect_gfeat(gfeat, cope_idx, require_report=require_report)
    if input_path.suffix == ".gz" and not input_path.is_file():
        status.issues.append(f"missing cope image: {input_path}")
    elif input_path.name.endswith(".feat") and not input_path.is_dir():
        status.issues.append(f"missing cope feat dir: {input_path}")
    return status


def inspect_subject(
    feat_root: Path,
    sub_code: str,
    gfeat_suffix: str,
    cope_idx: int,
    *,
    require_report: bool,
) -> SubjectGfeatStatus:
    sub_id = f"sub-{sub_code}"
    gfeat = resolve_gfeat(feat_root, sub_code, gfeat_suffix)
    if gfeat is None:
        return SubjectGfeatStatus(
            sub_id=sub_id,
            sub_code=sub_code,
            gfeat_path=None,
            issues=["missing 2nd-level .gfeat/.feat directory"],
        )
    return inspect_gfeat(gfeat, cope_idx, require_report=require_report)


def majority_fingerprint(
    statuses: list[SubjectGfeatStatus],
) -> tuple[ContrastFingerprint | None, Counter]:
    fps = [s.fingerprint for s in statuses if s.fingerprint is not None]
    if not fps:
        return None, Counter()
    counts = Counter(fps)
    return counts.most_common(1)[0][0], counts


def run_analysis(
    spec: dict[str, str | int],
    feat_root: Path,
    subjects: list[str],
    *,
    inputs_file: Path | None,
    require_report: bool,
    verbose: bool,
) -> int:
    label = str(spec["label"])
    gfeat_suffix = str(spec["gfeat_suffix"])
    cope_idx = int(spec["cope"])

    print(f"\n{'=' * 72}")
    print(f"Analysis: {label}")
    print(f"2nd-level suffix: {gfeat_suffix}")
    print(f"3rd-level cope index: {cope_idx}")
    print(f"{'=' * 72}")

    if inputs_file is not None:
        paths = load_paths_from_inputs(inputs_file)
        statuses = [
            inspect_input_path(p, cope_idx, require_report=require_report) for p in paths
        ]
        print(f"Subjects: {len(statuses)} (from {inputs_file})")
    else:
        statuses = [
            inspect_subject(
                feat_root, sub_code, gfeat_suffix, cope_idx, require_report=require_report
            )
            for sub_code in subjects
        ]
        print(f"Subjects: {len(statuses)} (from CSV filter {spec['filter_col']})")

    missing = [s for s in statuses if s.gfeat_path is None]
    broken = [s for s in statuses if s.gfeat_path is not None and s.issues]
    poolable = [s for s in statuses if s.ok_for_pooling]

    if missing:
        print(f"\nMissing 2nd-level dir ({len(missing)}):")
        for s in missing:
            print(f"  {s.sub_id}")

    if broken:
        print(f"\nPresent but failing checks ({len(broken)}):")
        for s in broken:
            print(f"  {s.sub_id}: {'; '.join(s.issues)}")

    ref_fp, fp_counts = majority_fingerprint(poolable)
    if ref_fp is None:
        print("\nNo subjects with a parseable design.con — cannot assess contrast alignment.")
        return 1

    print(f"\nMajority contrast fingerprint ({fp_counts[ref_fp]} / {len(poolable)} poolable):")
    print(f"  {ref_fp.describe()}")

    outliers = [s for s in poolable if s.fingerprint != ref_fp]
    aligned = [s for s in poolable if s.fingerprint == ref_fp]

    print(f"\nAligned with majority: {len(aligned)}")
    if verbose and aligned:
        for s in aligned[:5]:
            print(f"  {s.sub_id}")
        if len(aligned) > 5:
            print(f"  ... and {len(aligned) - 5} more")

    if outliers:
        print(f"\nMISALIGNED (would break 3rd-level pooling): {len(outliers)}")
        for s in outliers:
            assert s.fingerprint is not None
            print(f"  {s.sub_id}  [{s.gfeat_path}]")
            print(f"    got:      {s.fingerprint.describe()}")
            print(f"    expected: {ref_fp.describe()}")
    else:
        print("\nNo contrast fingerprint outliers among poolable subjects.")

    # Secondary: multiple fingerprint groups among poolable (even if one is majority)
    if len(fp_counts) > 1:
        print(f"\nFingerprint groups among poolable subjects ({len(fp_counts)} distinct):")
        for fp, count in fp_counts.most_common():
            tag = "majority" if fp == ref_fp else "outlier"
            print(f"  [{tag}] n={count}: {fp.describe()}")

    n_problem = len(missing) + len(broken) + len(outliers)
    print(
        f"\nSummary: aligned={len(aligned)} misaligned={len(outliers)} "
        f"missing={len(missing)} other_issues={len(broken)}"
    )
    return 1 if n_problem else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find subjects whose 2nd-level .gfeat contrasts do not match the cohort majority.",
    )
    p.add_argument(
        "--bids-root",
        type=Path,
        default=DEFAULT_BIDS_ROOT,
        help="BIDS root (default: cluster path)",
    )
    p.add_argument(
        "--subject-csv",
        type=Path,
        default=None,
        help="Subject CSV (default: <code-root>/ses-v2-analysis-subs-fmri.csv)",
    )
    p.add_argument(
        "--feat-root",
        type=Path,
        default=None,
        help="fslFeat root (default: <bids-root>/derivatives/fslFeat)",
    )
    p.add_argument(
        "--code-root",
        type=Path,
        default=DEFAULT_CODE_ROOT,
        help="fMRI_exp2 repository root on cluster",
    )
    p.add_argument(
        "--analysis",
        choices=[str(a["label"]) for a in ACTIVATION_ANALYSES] + ["all"],
        default="all",
        help="Which activation 2nd-level GLM to check (default: all)",
    )
    p.add_argument(
        "--inputs",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional inputs_<label>.txt from 06_make (one copeN.feat path per line); overrides CSV subject list",
    )
    p.add_argument(
        "--no-require-report",
        action="store_true",
        help="Do not flag .gfeat dirs whose report.html contains Error/ERROR",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List first few aligned subjects",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write per-subject results to CSV",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    code_root = args.code_root
    subject_csv = args.subject_csv or (code_root / "ses-v2-analysis-subs-fmri.csv")
    feat_root = args.feat_root or (args.bids_root / "derivatives/fslFeat")
    require_report = not args.no_require_report

    if not subject_csv.is_file() and args.inputs is None:
        print(f"ERROR: subject CSV not found: {subject_csv}", file=sys.stderr)
        return 1

    specs = list(ACTIVATION_ANALYSES)
    if args.analysis != "all":
        specs = [s for s in specs if s["label"] == args.analysis]

    csv_rows: list[dict[str, str]] = []
    exit_code = 0

    for spec in specs:
        label = str(spec["label"])
        inputs_file = args.inputs
        if inputs_file is None and args.analysis == "all":
            default_inputs = (
                code_root / "activation" / "fsf_files" / "group" / f"inputs_{label}.txt"
            )
            if default_inputs.is_file():
                inputs_file = default_inputs

        if inputs_file is not None and args.analysis == "all" and args.inputs is None:
            print(f">> using inclusion list: {inputs_file}")

        subjects: list[str] = []
        if inputs_file is None:
            subjects = load_subjects(subject_csv, str(spec["filter_col"]))

        rc = run_analysis(
            spec,
            feat_root,
            subjects,
            inputs_file=inputs_file,
            require_report=require_report,
            verbose=args.verbose,
        )
        exit_code = max(exit_code, rc)

        # Collect rows for optional CSV (re-run inspection for export)
        if args.csv_out is not None:
            if inputs_file is not None:
                paths = load_paths_from_inputs(inputs_file)
                statuses = [
                    inspect_input_path(p, int(spec["cope"]), require_report=require_report)
                    for p in paths
                ]
            else:
                statuses = [
                    inspect_subject(
                        feat_root,
                        sc,
                        str(spec["gfeat_suffix"]),
                        int(spec["cope"]),
                        require_report=require_report,
                    )
                    for sc in subjects
                ]
            ref_fp, _ = majority_fingerprint([s for s in statuses if s.ok_for_pooling])
            for s in statuses:
                aligned = (
                    s.ok_for_pooling
                    and ref_fp is not None
                    and s.fingerprint == ref_fp
                )
                csv_rows.append(
                    {
                        "analysis": label,
                        "sub_id": s.sub_id,
                        "gfeat_path": str(s.gfeat_path or ""),
                        "aligned": "yes" if aligned else "no",
                        "issues": "; ".join(s.issues),
                        "fingerprint": s.fingerprint.describe() if s.fingerprint else "",
                        "n_cope_feat": str(s.n_cope_feat_dirs),
                    }
                )

    if args.csv_out is not None and csv_rows:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {args.csv_out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
