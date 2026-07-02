#!/usr/bin/env python3
"""
Build activation 3rd-level inclusion report with trace-back for excluded subjects.

Reads 05 included_subjects_report.txt, optional 04 included_runs_report.txt, and
checks derivative paths required for 1st / 2nd / 3rd-level FEAT.
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

SESSION = "ses-V2"
ANAT_SESSIONS = ("ses-V1", "ses-V2")
TASK_RUN_PREFIX = "ses-V2_task-VG_run-"
SPACE = "MNI152NLin2009cAsym"
FEAT_1ST_SUFFIX = "analysis-1stGLM_space-MNI152NLin2009cAsym"
MAX_RUNS = 8

EVENT_SUFFIXES = (
    "probedSeenFaceLeft_EV.txt",
    "probedSeenFaceRight_EV.txt",
    "probedSeenObjectLeft_EV.txt",
    "probedSeenObjectRight_EV.txt",
    "probedCorrectrejectNoneLeft_EV.txt",
    "probedCorrectrejectNoneRight_EV.txt",
    "probedUnseenFaceLeft_EV.txt",
    "probedUnseenFaceRight_EV.txt",
    "probedUnseenObjectLeft_EV.txt",
    "probedUnseenObjectRight_EV.txt",
    "probedFalsealarmNoneLeft_EV.txt",
    "probedFalsealarmNoneRight_EV.txt",
    "unprobedFaceLeft_EV.txt",
    "unprobedFaceRight_EV.txt",
    "unprobedObjectLeft_EV.txt",
    "unprobedObjectRight_EV.txt",
    "unprobedNoneLeft_EV.txt",
    "unprobedNoneRight_EV.txt",
)

# label:2nd_gfeat_suffix:filter:cope_idx:group_output_label
DEFAULT_ANALYSES = (
    "seen_face_vs_unseen_face:analysis-2ndGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym:ACTIVATION_min_sf_uf:1:seen_vs_unseen_F",
    "seen_object_vs_unseen_object:analysis-2ndGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym:ACTIVATION_min_so_uo:2:seen_vs_unseen_O",
)

# Defaults match activation/make_3rd_level_inclusion_report.sh (override via env or CLI).
DEFAULT_BIDS_ROOT = "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
DEFAULT_CODE_ROOT = f"{DEFAULT_BIDS_ROOT}/code"
DEFAULT_ACTIVATION_ROOT = f"{DEFAULT_CODE_ROOT}/activation"


@dataclass
class AnalysisSpec:
    label: str
    gfeat_suffix: str
    filter_mode: str
    cope: int
    group_label: str


@dataclass
class InclusionSection:
    analysis: str
    cope_tag: str
    filter_mode: str
    group_label: str
    included: list[tuple[str, str]] = field(default_factory=list)
    excluded: list[tuple[str, str, str]] = field(default_factory=list)


def parse_analyses(specs: list[str]) -> list[AnalysisSpec]:
    out = []
    for spec in specs:
        label, gfeat_suffix, filter_mode, cope_s, group_label = spec.split(":", 4)
        out.append(
            AnalysisSpec(
                label=label,
                gfeat_suffix=gfeat_suffix,
                filter_mode=filter_mode,
                cope=int(cope_s),
                group_label=group_label,
            )
        )
    return out


def load_subject_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    required = ["ACTIVATION_min_sf_uf", "ACTIVATION_min_so_uo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    return df


def subjects_for_filter(df: pd.DataFrame, filter_mode: str) -> list[str]:
    flag_sf = df["ACTIVATION_min_sf_uf"].astype(str).str.upper().eq("TRUE")
    flag_so = df["ACTIVATION_min_so_uo"].astype(str).str.upper().eq("TRUE")
    if filter_mode == "union":
        sub_df = df.loc[flag_sf | flag_so]
    elif filter_mode == "both":
        sub_df = df.loc[flag_sf & flag_so]
    elif filter_mode in df.columns:
        sub_df = df.loc[df[filter_mode].astype(str).str.upper().eq("TRUE")]
    else:
        raise ValueError(f"unknown filter_mode={filter_mode}")
    return [str(x) for x in sub_df["sub_code"].tolist()]


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def parse_inclusion_report(path: Path) -> dict[str, InclusionSection]:
    if not path.is_file():
        return {}
    sections: dict[str, InclusionSection] = {}
    current: InclusionSection | None = None
    mode: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## analysis="):
            parts = dict(p.split("=", 1) for p in line[3:].split("\t") if "=" in p)
            analysis = parts.get("analysis", "")
            cope_tag = parts.get("cope", "")
            filter_mode = parts.get("filter", "")
            current = InclusionSection(
                analysis=analysis,
                cope_tag=cope_tag,
                filter_mode=filter_mode,
                group_label="",
            )
            sections[analysis] = current
            mode = None
            continue
        if current is None:
            continue
        if line.startswith("group_output_label="):
            current.group_label = line.split("=", 1)[1].strip()
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


def parse_runs_report(path: Path, analysis: str, subj_id: str) -> dict[int, tuple[str, str]]:
    """Return {run: (status, path_or_detail)} for a subject/analysis from 04 report."""
    if not path.is_file():
        return {}
    runs: dict[int, tuple[str, str]] = {}
    in_section = False
    mode: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## analysis=") and f"subject={subj_id}" in line:
            in_section = analysis in line
            mode = None
            continue
        if not in_section:
            continue
        if line.startswith("# Included runs"):
            mode = "included"
            continue
        if line.startswith("# Excluded runs"):
            mode = "excluded"
            continue
        if line.startswith("## "):
            in_section = False
            continue
        if not line.strip() or line.startswith("#"):
            continue
        m = re.match(r"run-(\d+)\t(.+)", line)
        if not m:
            continue
        run = int(m.group(1))
        rest = m.group(2)
        if mode == "included":
            runs[run] = ("included", rest.split("\t", 1)[0])
        elif mode == "excluded":
            parts = rest.split("\t", 1)
            reason = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            runs[run] = ("excluded", f"{reason}\t{detail}".strip())
    return runs


def run_label(run: int) -> str:
    return f"{TASK_RUN_PREFIX}{run}"


def fmriprep_bold(bids_root: Path, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / "derivatives/fmriprep"
        / sub_id
        / SESSION
        / "func"
        / f"{sub_id}_{run_label(run)}_{SPACE}_desc-preproc_bold.nii.gz"
    )


def fmriprep_confounds_tsv(bids_root: Path, sub_code: str, run: int) -> list[Path]:
    sub_id = f"sub-{sub_code}"
    base = bids_root / "derivatives/fmriprep" / sub_id / SESSION / "func"
    return [
        base / f"{sub_id}_{run_label(run)}_desc-confounds_timeseries.tsv",
        base / f"{sub_id}_{run_label(run)}_desc-confounds_regressors.tsv",
    ]


def fmriprep_structural_paths(bids_root: Path, sub_code: str) -> dict[str, list[Path]]:
    """Candidate T1w / structural images from fMRIPrep (per session)."""
    sub_id = f"sub-{sub_code}"
    out: dict[str, list[Path]] = {}
    for ses in ANAT_SESSIONS:
        anat_dir = bids_root / "derivatives/fmriprep" / sub_id / ses / "anat"
        explicit = [
            anat_dir / f"{sub_id}_{ses}_T1w.nii.gz",
            anat_dir / f"{sub_id}_{ses}_run-1_T1w.nii.gz",
            anat_dir / f"{sub_id}_{ses}_desc-preproc_T1w.nii.gz",
            anat_dir / f"{sub_id}_{ses}_run-1_desc-preproc_T1w.nii.gz",
            anat_dir / f"{sub_id}_{ses}_{SPACE}_desc-preproc_T1w.nii.gz",
        ]
        if anat_dir.is_dir():
            explicit.extend(sorted(anat_dir.glob(f"{sub_id}*T1w*.nii.gz")))
        out[ses] = explicit
    return out


def append_structural_trace(lines: list[str], bids_root: Path, sub_code: str) -> None:
    """Report fMRIPrep T1w status under ses-V1 and ses-V2."""
    for ses, paths in fmriprep_structural_paths(bids_root, sub_code).items():
        found = [p for p in paths if p.is_file()]
        if found:
            lines.append(f"  [structural {ses}] OK: {found[0]}")
        else:
            lines.append(f"  [structural {ses}] missing T1w (checked):")
            for p in paths[:5]:
                lines.append(f"    - {p}")
            if len(paths) > 5:
                lines.append(f"    ... and {len(paths) - 5} more candidate(s)")


def confound_ev_file(bids_root: Path, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / "derivatives/regressoreventfiles"
        / sub_id
        / SESSION
        / "confound_event_files"
        / f"{sub_id}_{run_label(run)}_confounds.txt"
    )


def event_ev_file(bids_root: Path, sub_code: str, run: int, suffix: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / "derivatives/regressoreventfiles"
        / sub_id
        / SESSION
        / f"{sub_id}_{run_label(run)}_{suffix}"
    )


def first_level_feat(feat_root: Path, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        feat_root
        / "activation"
        / sub_id
        / f"{sub_id}_{SESSION}_task-VG_run-{run}_{FEAT_1ST_SUFFIX}.feat"
    )


def second_level_parent(feat_root: Path, sub_code: str, gfeat_suffix: str) -> tuple[Path | None, str]:
    sub_id = f"sub-{sub_code}"
    stem = f"{sub_id}_{SESSION}_task-VG_{gfeat_suffix}"
    for ext in (".gfeat", ".feat"):
        parent = feat_root / "activation" / sub_id / f"{stem}{ext}"
        if parent.is_dir():
            return parent, ext
    return None, ""


def trace_first_level_inputs(
    bids_root: Path, sub_code: str, run: int
) -> list[str]:
    lines: list[str] = []
    bold = fmriprep_bold(bids_root, sub_code, run)
    if not bold.is_file():
        lines.append(f"  [fmriprep] missing preprocessed BOLD: {bold}")

    conf_tsvs = fmriprep_confounds_tsv(bids_root, sub_code, run)
    if not any(p.is_file() for p in conf_tsvs):
        lines.append(
            "  [fmriprep] missing confounds TSV (checked):\n"
            + "".join(f"    - {p}\n" for p in conf_tsvs)
        )

    conf_txt = confound_ev_file(bids_root, sub_code, run)
    if not conf_txt.is_file():
        lines.append(f"  [confound EV] missing: {conf_txt}")

    missing_events = [
        event_ev_file(bids_root, sub_code, run, suf)
        for suf in EVENT_SUFFIXES
        if not event_ev_file(bids_root, sub_code, run, suf).is_file()
    ]
    if missing_events:
        lines.append(f"  [event EV] missing {len(missing_events)} file(s), e.g.:")
        for p in missing_events[:5]:
            lines.append(f"    - {p}")
        if len(missing_events) > 5:
            lines.append(f"    ... and {len(missing_events) - 5} more")

    return lines


def trace_first_level_run(
    feat_root: Path,
    bids_root: Path,
    sub_code: str,
    run: int,
    runs_report: Path | None,
    analysis_label: str,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []
    feat_path = first_level_feat(feat_root, sub_code, run)

    if runs_report is not None:
        run_info = parse_runs_report(runs_report, analysis_label, sub_id)
        if run in run_info:
            status, detail = run_info[run]
            if status == "excluded":
                lines.append(
                    f"  run-{run}: excluded in 04_run_2nd_level ({detail})"
                )
                input_lines = trace_first_level_inputs(bids_root, sub_code, run)
                if input_lines:
                    lines.append("  upstream inputs for this run:")
                    lines.extend(input_lines)
                return lines
            if status == "included":
                lines.append(f"  run-{run}: included in 04 patched 2nd-level FSF ({detail})")

    if not feat_path.is_dir():
        lines.append(f"  run-{run}: missing 1st-level .feat: {feat_path}")
        lines.append("  upstream inputs for this run:")
        lines.extend(trace_first_level_inputs(bids_root, sub_code, run))
        return lines

    if not (feat_path / "report.html").is_file():
        lines.append(f"  run-{run}: missing report.html in {feat_path}")
    elif not report_ok(feat_path):
        lines.append(f"  run-{run}: error in report.html ({feat_path})")
    elif not (feat_path / "reg" / "example_func2standard.mat").is_file():
        lines.append(
            f"  run-{run}: missing identity reg example_func2standard.mat "
            f"(run 02_create_identity_reg_for_2nd_lvl.sh?)"
        )
    else:
        lines.append(f"  run-{run}: 1st-level OK ({feat_path})")
    return lines


def trace_second_level(
    feat_root: Path,
    bids_root: Path,
    activation_root: Path,
    sub_code: str,
    spec: AnalysisSpec,
    runs_report: Path | None,
    analysis_label: str,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []
    parent, ext = second_level_parent(feat_root, sub_code, spec.gfeat_suffix)
    cope_path = (parent / f"cope{spec.cope}.feat") if parent else None

    job_dir = activation_root / "job_files/run_activation_2nd_level" / sub_code
    run_marker = (
        job_dir
        / f"{sub_id}_{SESSION}_task-VG_{spec.gfeat_suffix}.run"
    )

    if parent is None:
        lines.append(
            f"[2nd level] missing directory: "
            f"{feat_root / 'activation' / sub_id}/{sub_id}_{SESSION}_task-VG_{spec.gfeat_suffix}.gfeat"
        )
        if run_marker.is_file():
            lines.append(f"  2nd-level job marker exists: {run_marker}")
        else:
            lines.append(
                f"  2nd-level job marker missing (04_run_2nd_level.sh may not have submitted)"
            )
        lines.append("  tracing 1st-level runs required for 2nd level:")
        for run in range(1, MAX_RUNS + 1):
            lines.extend(
                trace_first_level_run(
                    feat_root,
                    bids_root,
                    sub_code,
                    run,
                    runs_report,
                    analysis_label,
                )
            )
        return lines

    if not report_ok(parent):
        lines.append(f"[2nd level] bad or missing report.html ({ext}): {parent}")
    if cope_path is None or not cope_path.is_dir():
        lines.append(
            f"[2nd level] missing cope{spec.cope}.feat under {parent} "
            f"(2nd-level may have failed or copeinput.{spec.cope} not selected)"
        )
        for run in range(1, MAX_RUNS + 1):
            c1 = first_level_feat(feat_root, sub_code, run) / "stats" / f"cope{spec.cope}.nii.gz"
            if c1.is_file():
                lines.append(f"  run-{run}: 1st-level cope{spec.cope} exists ({c1})")
        lines.append("  1st-level run status:")
        for run in range(1, MAX_RUNS + 1):
            lines.extend(
                trace_first_level_run(
                    feat_root,
                    bids_root,
                    sub_code,
                    run,
                    runs_report,
                    analysis_label,
                )
            )
    else:
        lines.append(f"[2nd level] directory OK: {cope_path}")
    return lines


def trace_excluded_subject(
    feat_root: Path,
    bids_root: Path,
    activation_root: Path,
    sub_code: str,
    spec: AnalysisSpec,
    reason_05: str,
    path_05: str,
    runs_report: Path | None,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    analysis_2nd_label = spec.label
    lines = [
        f"### {sub_id}",
        f"05_exclusion_reason: {reason_05}",
        f"05_path: {path_05}",
        "",
        "Subject-level derivatives:",
    ]
    append_structural_trace(lines, bids_root, sub_code)
    lines.append("")
    lines.extend(
        trace_second_level(
            feat_root,
            bids_root,
            activation_root,
            sub_code,
            spec,
            runs_report,
            analysis_2nd_label,
        )
    )
    lines.append("")
    return lines


def submission_status_for(
    submission_report: Path | None, analysis: str, cope_tag: str
) -> str:
    if submission_report is None or not submission_report.is_file():
        return "unknown"
    for line in submission_report.read_text(encoding="utf-8").splitlines():
        if line.startswith("## pipeline=") and f"analysis={analysis}" in line:
            if cope_tag and f"cope={cope_tag}" not in line:
                continue
            m = re.search(r"submission_status=(\S+)", line)
            if m:
                return m.group(1)
    return "unknown"


def write_report(
    report_path: Path,
    analyses: list[AnalysisSpec],
    subject_df: pd.DataFrame,
    inclusion_sections: dict[str, InclusionSection],
    bids_root: Path,
    feat_root: Path,
    activation_root: Path,
    inclusion_report: Path,
    runs_report: Path | None,
    submission_report: Path | None,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# Activation 3rd-level inclusion report",
        f"# date={stamp}",
        f"# source_inclusion_report={inclusion_report}",
        f"# source_runs_report={runs_report or ''}",
        f"# source_submission_report={submission_report or ''}",
        "",
    ]

    for spec in analyses:
        cope_tag = f"cope{spec.cope}"
        section = inclusion_sections.get(spec.label)
        candidates = subjects_for_filter(subject_df, spec.filter_mode)
        included_subs: list[str] = []
        excluded_map: dict[str, tuple[str, str]] = {}

        if section:
            included_subs = [s for s, _ in section.included]
            included_codes = {s.replace("sub-", "", 1) for s in included_subs}
            for subj, reason, path in section.excluded:
                code = subj.replace("sub-", "", 1)
                excluded_map[code] = (reason, path)
            for code in included_codes:
                excluded_map.pop(code, None)
        else:
            included_subs = []
            included_codes = set()
            lines.append(
                f"## analysis={spec.label}\tWARNING=no section in 05 inclusion report"
            )

        # Subjects in CSV filter but not in 05 included list
        for code in candidates:
            if code not in included_codes and code not in excluded_map:
                excluded_map[code] = (
                    "not listed in 05 inclusion report (filtered out or not processed)",
                    "",
                )

        sub_status = submission_status_for(submission_report, spec.label, cope_tag)

        lines.extend(
            [
                "",
                "=" * 72,
                f"## analysis={spec.label}",
                f"filter={spec.filter_mode}",
                f"2nd_level_cope={cope_tag}",
                f"group_output_label={spec.group_label}",
                f"3rd_level_submission_status={sub_status}",
                f"n_csv_candidates={len(candidates)}",
                f"n_included={len(included_codes)}",
                f"n_excluded={len(excluded_map)}",
                "",
                "# Included subjects (in 3rd-level FSF)",
            ]
        )
        if included_subs:
            for subj in included_subs:
                path = next((p for s, p in (section.included if section else []) if s == subj), "")
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
                        activation_root,
                        code,
                        spec,
                        reason,
                        path,
                        runs_report,
                    )
                )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Activation 3rd-level inclusion report")
    p.add_argument(
        "--bids-root",
        default=os.environ.get("BIDS_ROOT", DEFAULT_BIDS_ROOT),
    )
    p.add_argument(
        "--activation-root",
        default=os.environ.get("ACTIVATION_CODE_ROOT", DEFAULT_ACTIVATION_ROOT),
    )
    p.add_argument(
        "--subject-csv",
        default=os.environ.get("SUBJECT_CSV", f"{DEFAULT_CODE_ROOT}/ses-v2-analysis-subs-fmri.csv"),
    )
    p.add_argument(
        "--inclusion-report",
        default=os.environ.get(
            "INCLUSION_REPORT",
            f"{DEFAULT_ACTIVATION_ROOT}/fsf_files/group/included_subjects_report.txt",
        ),
    )
    p.add_argument(
        "--runs-report",
        default=os.environ.get(
            "RUNS_REPORT",
            f"{DEFAULT_ACTIVATION_ROOT}/job_files/run_activation_2nd_level/included_runs_report.txt",
        ),
    )
    p.add_argument(
        "--submission-report",
        default=os.environ.get(
            "SUBMISSION_REPORT",
            f"{DEFAULT_ACTIVATION_ROOT}/job_files/run_activation_3rd_level/submission_inclusion_report.txt",
        ),
    )
    p.add_argument(
        "--report-dir",
        default=os.environ.get("REPORT_DIR", f"{DEFAULT_ACTIVATION_ROOT}/3rd level report"),
    )
    p.add_argument(
        "--analyses-csv",
        default=os.environ.get("ANALYSES_CSV", "\n".join(DEFAULT_ANALYSES)),
        help="Newline-separated analysis specs (same format as 05)",
    )
    args = p.parse_args()

    for name, val in (
        ("bids-root", args.bids_root),
        ("activation-root", args.activation_root),
        ("subject-csv", args.subject_csv),
        ("inclusion-report", args.inclusion_report),
        ("report-dir", args.report_dir),
    ):
        if not val:
            print(f"ERROR: --{name} is required", file=sys.stderr)
            return 1

    bids_root = Path(args.bids_root)
    activation_root = Path(args.activation_root)
    feat_root = bids_root / "derivatives/fslFeat"
    report_dir = Path(args.report_dir)
    report_path = report_dir / "3rd_level_inclusion_report.txt"

    analyses = parse_analyses(
        [ln for ln in args.analyses_csv.splitlines() if ln.strip()]
    )
    subject_df = load_subject_csv(Path(args.subject_csv))
    inclusion_sections = parse_inclusion_report(Path(args.inclusion_report))
    runs_report = Path(args.runs_report) if args.runs_report else None
    submission_report = Path(args.submission_report) if args.submission_report else None

    write_report(
        report_path,
        analyses,
        subject_df,
        inclusion_sections,
        bids_root,
        feat_root,
        activation_root,
        Path(args.inclusion_report),
        runs_report,
        submission_report,
    )
    print(f">> wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
