#!/usr/bin/env python3
"""
Build gPPI 3rd-level inclusion report with trace-back for excluded subjects.

Reads 09_make included_subjects_report.txt, optional 07 included_runs_report.txt, and
checks derivatives (seeds, masks, structural images, events, fmriprep, FEAT).
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

from gppi_3rd_level import ANALYSES, ANALYSIS_COPE

SESSION = "ses-V2"
ANAT_SESSIONS = ("ses-V1", "ses-V2")
TASK_RUN_PREFIX = "ses-V2_task-VG_run-"
SPACE = "MNI152NLin2009cAsym"
MAX_RUNS = 8
FILTER_COL = "SYNCHRONY_min_seen"

PREFER_EV = {
    "PPI_FFA": "prefer_Face.txt",
    "PPI_LOC": "prefer_Object.txt",
}

GPPI_EVENT_SUFFIXES = (
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

PREFER_UPSTREAM_SUFFIXES = (
    "probedSeenFaceLeft_EV.txt",
    "probedSeenFaceRight_EV.txt",
    "probedSeenObjectLeft_EV.txt",
    "probedSeenObjectRight_EV.txt",
)

# Defaults match gPPI_code/make_3rd_level_inclusion_report.sh (override via env or CLI).
DEFAULT_BIDS_ROOT = "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
DEFAULT_CODE_ROOT = f"{DEFAULT_BIDS_ROOT}/code"
DEFAULT_GPPI_ROOT = f"{DEFAULT_CODE_ROOT}/gPPI_code"
DEFAULT_TIMECOURSE_ROOT = f"{DEFAULT_BIDS_ROOT}/derivatives/gppi_timecourse"


@dataclass
class InclusionSection:
    analysis: str
    cope_tag: str
    included: list[tuple[str, str]] = field(default_factory=list)
    excluded: list[tuple[str, str, str]] = field(default_factory=list)


def load_subject_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    if FILTER_COL not in df.columns:
        raise ValueError(f"missing column {FILTER_COL} in {path}")
    return df


def subjects_for_filter(df: pd.DataFrame) -> list[str]:
    sub_df = df.loc[df[FILTER_COL].astype(str).str.upper().eq("TRUE")]
    return [str(x) for x in sub_df["sub_code"].tolist()]


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


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


def parse_runs_report(path: Path, analysis: str, subj_id: str) -> dict[int, tuple[str, str]]:
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


def fmriprep_brain_mask(bids_root: Path, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / "derivatives/fmriprep"
        / sub_id
        / SESSION
        / "func"
        / f"{sub_id}_{run_label(run)}_{SPACE}_desc-brain_mask.nii.gz"
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
    by_session = fmriprep_structural_paths(bids_root, sub_code)
    for ses, paths in by_session.items():
        found = [p for p in paths if p.is_file()]
        if found:
            lines.append(f"  [structural {ses}] OK: {found[0]}")
        else:
            lines.append(f"  [structural {ses}] missing T1w (checked):")
            for p in paths[:5]:
                lines.append(f"    - {p}")
            if len(paths) > 5:
                lines.append(f"    ... and {len(paths) - 5} more candidate(s)")


def seed_mask_paths(seeds_root: Path, sub_code: str) -> dict[str, Path]:
    sub_id = f"sub-{sub_code}"
    seed_dir = seeds_root / sub_id
    return {
        "FFA": seed_dir
        / f"{sub_id}_rel_irrel_bh_face_FFA_n_voxels_300_{SPACE}.nii.gz",
        "LOC": seed_dir
        / f"{sub_id}_rel_irrel_bh_object_LOC_n_voxels_300_{SPACE}.nii.gz",
    }


def timecourse_paths(timecourse_root: Path, sub_code: str, run: int) -> dict[str, Path]:
    sub_id = f"sub-{sub_code}"
    run_base = f"{sub_id}_{run_label(run)}"
    tc_dir = timecourse_root / sub_id
    return {
        "FFA": tc_dir / f"{run_base}_timecourse_FFA.txt",
        "LOC": tc_dir / f"{run_base}_timecourse_LOC.txt",
    }


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


def first_level_feat(
    feat_root: Path, sub_code: str, run: int, analysis: str
) -> Path:
    sub_id = f"sub-{sub_code}"
    suffix = f"analysis-1stGLM_{analysis}-{SPACE}"
    return (
        feat_root
        / "gPPI"
        / sub_id
        / f"{sub_id}_{SESSION}_task-VG_run-{run}_{suffix}.feat"
    )


def second_level_parent(
    feat_root: Path, sub_code: str, analysis: str
) -> tuple[Path | None, str]:
    sub_id = f"sub-{sub_code}"
    stem = f"{sub_id}_{SESSION}_task-VG_analysis-2ndGLM_{analysis}-{SPACE}"
    for ext in (".gfeat", ".feat"):
        parent = feat_root / "gPPI" / sub_id / f"{stem}{ext}"
        if parent.is_dir():
            return parent, ext
    return None, ""


def trace_subject_level_derivatives(
    bids_root: Path, seeds_root: Path, sub_code: str
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []

    append_structural_trace(lines, bids_root, sub_code)

    masks = seed_mask_paths(seeds_root, sub_code)
    for label, path in masks.items():
        if not path.is_file():
            lines.append(f"  [seed mask {label}] missing anatomical ROI mask: {path}")
        else:
            lines.append(f"  [seed mask {label}] OK: {path}")

    seed_dir = seeds_root / sub_id
    if not seed_dir.is_dir():
        lines.append(f"  [seeds] missing subject seed directory: {seed_dir}")

    return lines


def trace_first_level_inputs(
    bids_root: Path,
    timecourse_root: Path,
    sub_code: str,
    run: int,
    analysis: str,
) -> list[str]:
    lines: list[str] = []
    bold = fmriprep_bold(bids_root, sub_code, run)
    if not bold.is_file():
        lines.append(f"  [fmriprep] missing preprocessed BOLD: {bold}")

    bmask = fmriprep_brain_mask(bids_root, sub_code, run)
    if not bmask.is_file():
        lines.append(f"  [fmriprep] missing brain mask: {bmask}")

    conf_tsvs = fmriprep_confounds_tsv(bids_root, sub_code, run)
    if not any(p.is_file() for p in conf_tsvs):
        lines.append("  [fmriprep] missing confounds TSV (checked):")
        for p in conf_tsvs:
            lines.append(f"    - {p}")

    conf_txt = confound_ev_file(bids_root, sub_code, run)
    if not conf_txt.is_file():
        lines.append(f"  [confound EV] missing: {conf_txt}")

    prefer = PREFER_EV.get(analysis, "")
    if prefer:
        prefer_path = event_ev_file(bids_root, sub_code, run, prefer)
        if not prefer_path.is_file():
            lines.append(f"  [event EV] missing {prefer}: {prefer_path}")
            missing_up = [
                event_ev_file(bids_root, sub_code, run, suf)
                for suf in PREFER_UPSTREAM_SUFFIXES
                if not event_ev_file(bids_root, sub_code, run, suf).is_file()
            ]
            if missing_up:
                lines.append("  [event EV] upstream for prefer_* (examples):")
                for p in missing_up[:4]:
                    lines.append(f"    - {p}")

    missing_events = [
        event_ev_file(bids_root, sub_code, run, suf)
        for suf in GPPI_EVENT_SUFFIXES
        if not event_ev_file(bids_root, sub_code, run, suf).is_file()
    ]
    if missing_events:
        lines.append(f"  [event EV] missing {len(missing_events)} gPPI task file(s), e.g.:")
        for p in missing_events[:4]:
            lines.append(f"    - {p}")

    tcs = timecourse_paths(timecourse_root, sub_code, run)
    for label, path in tcs.items():
        if not path.is_file():
            lines.append(f"  [timecourse {label}] missing (run 02_make_timecourse_files.sh?): {path}")

    return lines


def trace_first_level_run(
    feat_root: Path,
    bids_root: Path,
    seeds_root: Path,
    timecourse_root: Path,
    gppi_root: Path,
    sub_code: str,
    run: int,
    analysis: str,
    runs_report: Path | None,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []
    feat_path = first_level_feat(feat_root, sub_code, run, analysis)
    feat_suffix = f"analysis-1stGLM_{analysis}-{SPACE}"

    if runs_report is not None:
        run_info = parse_runs_report(runs_report, analysis, sub_id)
        if run in run_info:
            status, detail = run_info[run]
            if status == "excluded":
                lines.append(f"  run-{run}: excluded in 07_run_PPI_2nd_level ({detail})")
                lines.append("  upstream inputs for this run:")
                lines.extend(
                    trace_first_level_inputs(
                        bids_root, timecourse_root, sub_code, run, analysis
                    )
                )
                return lines

    tc_marker = (
        gppi_root
        / "job_files/make_timecourse_files"
        / sub_code
        / f"{sub_id}_{SESSION}_task-VG_run-{run}_timecourse.run"
    )
    if not tc_marker.is_file():
        lines.append(
            f"  run-{run}: missing timecourse job marker (02_make_timecourse_files.sh?): {tc_marker}"
        )

    fsf_marker = (
        gppi_root
        / "job_files/run_PPI_1st_level"
        / sub_code
        / f"{sub_id}_{SESSION}_task-VG_run-{run}_{feat_suffix}.run"
    )
    if not fsf_marker.is_file():
        lines.append(
            f"  run-{run}: missing 1st-level job marker (04_run_PPI_1st_level.sh?): {fsf_marker}"
        )

    if not feat_path.is_dir():
        lines.append(f"  run-{run}: missing 1st-level .feat: {feat_path}")
        lines.append("  upstream inputs for this run:")
        lines.extend(
            trace_first_level_inputs(bids_root, timecourse_root, sub_code, run, analysis)
        )
        return lines

    if not (feat_path / "report.html").is_file():
        lines.append(f"  run-{run}: missing report.html in {feat_path}")
    elif not report_ok(feat_path):
        lines.append(f"  run-{run}: error in report.html ({feat_path})")
    elif not (feat_path / "reg" / "example_func2standard.mat").is_file():
        lines.append(
            f"  run-{run}: missing identity reg (05_create_identity_reg_for_2nd_lvl.sh?)"
        )
    else:
        lines.append(f"  run-{run}: 1st-level OK ({feat_path})")
    return lines


def trace_second_level(
    feat_root: Path,
    bids_root: Path,
    seeds_root: Path,
    timecourse_root: Path,
    gppi_root: Path,
    sub_code: str,
    analysis: str,
    cope: int,
    runs_report: Path | None,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []
    parent, ext = second_level_parent(feat_root, sub_code, analysis)
    cope_path = (parent / f"cope{cope}.feat") if parent else None

    gfeat_suffix = f"analysis-2ndGLM_{analysis}-{SPACE}"
    run_marker = (
        gppi_root
        / "job_files/run_PPI_2nd_level"
        / sub_code
        / f"{sub_id}_{SESSION}_task-VG_{gfeat_suffix}.run"
    )

    if parent is None:
        lines.append(
            f"[2nd level] missing directory: "
            f"{feat_root / 'gPPI' / sub_id}/{sub_id}_{SESSION}_task-VG_{gfeat_suffix}.gfeat"
        )
        if run_marker.is_file():
            lines.append(f"  2nd-level job marker exists: {run_marker}")
        else:
            lines.append(
                "  2nd-level job marker missing (07_run_PPI_2nd_level_VG.sh may not have submitted)"
            )
        lines.append("  subject-level derivatives:")
        lines.extend(trace_subject_level_derivatives(bids_root, seeds_root, sub_code))
        lines.append("  1st-level run status:")
        for run in range(1, MAX_RUNS + 1):
            lines.extend(
                trace_first_level_run(
                    feat_root,
                    bids_root,
                    seeds_root,
                    timecourse_root,
                    gppi_root,
                    sub_code,
                    run,
                    analysis,
                    runs_report,
                )
            )
        return lines

    if not report_ok(parent):
        lines.append(f"[2nd level] bad or missing report.html ({ext}): {parent}")
    if cope_path is None or not cope_path.is_dir():
        lines.append(
            f"[2nd level] missing cope{cope}.feat under {parent}"
        )
        lines.append("  1st-level run status:")
        for run in range(1, MAX_RUNS + 1):
            lines.extend(
                trace_first_level_run(
                    feat_root,
                    bids_root,
                    seeds_root,
                    timecourse_root,
                    gppi_root,
                    sub_code,
                    run,
                    analysis,
                    runs_report,
                )
            )
    else:
        lines.append(f"[2nd level] directory OK: {cope_path}")
        reg_mat = parent / "reg" / "example_func2standard.mat"
        if not reg_mat.is_file():
            lines.append(
                "[2nd level] missing identity reg at .gfeat root "
                "(08_create_identity_reg_for_3rd_lvl.sh?)"
            )
    return lines


def trace_excluded_subject(
    feat_root: Path,
    bids_root: Path,
    seeds_root: Path,
    timecourse_root: Path,
    gppi_root: Path,
    sub_code: str,
    analysis: str,
    cope: int,
    reason_make: str,
    path_make: str,
    runs_report: Path | None,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines = [
        f"### {sub_id}",
        f"09_make_exclusion_reason: {reason_make}",
        f"09_make_path: {path_make}",
        "",
    ]
    lines.extend(
        trace_second_level(
            feat_root,
            bids_root,
            seeds_root,
            timecourse_root,
            gppi_root,
            sub_code,
            analysis,
            cope,
            runs_report,
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
    inclusion_sections: dict[tuple[str, str], InclusionSection],
    bids_root: Path,
    feat_root: Path,
    seeds_root: Path,
    timecourse_root: Path,
    gppi_root: Path,
    inclusion_report: Path,
    runs_report: Path | None,
    submission_report: Path | None,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# gPPI 3rd-level inclusion report",
        f"# date={stamp}",
        f"# filter={FILTER_COL}",
        f"# source_inclusion_report={inclusion_report}",
        f"# source_runs_report={runs_report or ''}",
        f"# source_submission_report={submission_report or ''}",
        "",
    ]

    candidates = subjects_for_filter(subject_df)

    for analysis in ANALYSES:
        cope = ANALYSIS_COPE[analysis]
        cope_tag = f"cope{cope}"
        section = inclusion_sections.get((analysis, cope_tag))
        included_subs: list[str] = []
        excluded_map: dict[str, tuple[str, str]] = {}

        if section:
            included_subs = [s for s, _ in section.included]
            for subj, reason, path in section.excluded:
                code = subj.replace("sub-", "", 1)
                excluded_map[code] = (reason, path)
            for subj, _ in section.included:
                code = subj.replace("sub-", "", 1)
                excluded_map.pop(code, None)
        else:
            lines.append(
                f"## analysis={analysis}\tcope={cope_tag}\t"
                "WARNING=no section in 08 inclusion report"
            )

        for code in candidates:
            if code not in included_subs and code not in excluded_map:
                excluded_map[code] = (
                    "not listed in 08 inclusion report (filtered out or not processed)",
                    "",
                )

        sub_status = submission_status_for(submission_report, analysis, cope_tag)

        lines.extend(
            [
                "",
                "=" * 72,
                f"## analysis={analysis}\tcope={cope_tag}",
                f"filter={FILTER_COL}",
                f"3rd_level_submission_status={sub_status}",
                f"n_csv_candidates={len(candidates)}",
                f"n_included={len(included_subs)}",
                f"n_excluded={len(excluded_map)}",
                "",
                "# Included subjects (in 3rd-level FSF)",
            ]
        )
        if included_subs:
            for subj in included_subs:
                path = next(
                    (p for s, p in (section.included if section else []) if s == subj),
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
                        seeds_root,
                        timecourse_root,
                        gppi_root,
                        code,
                        analysis,
                        cope,
                        reason,
                        path,
                        runs_report,
                    )
                )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="gPPI 3rd-level inclusion report")
    p.add_argument(
        "--bids-root",
        default=os.environ.get("BIDS_ROOT", DEFAULT_BIDS_ROOT),
    )
    p.add_argument(
        "--gppi-root",
        default=os.environ.get("GPPI_CODE_ROOT", DEFAULT_GPPI_ROOT),
    )
    p.add_argument(
        "--timecourse-root",
        default=os.environ.get("TIMECOURSE_ROOT", DEFAULT_TIMECOURSE_ROOT),
    )
    p.add_argument(
        "--subject-csv",
        default=os.environ.get("SUBJECT_CSV", f"{DEFAULT_CODE_ROOT}/ses-v2-analysis-subs-fmri.csv"),
    )
    p.add_argument(
        "--inclusion-report",
        default=os.environ.get(
            "INCLUSION_REPORT",
            f"{DEFAULT_GPPI_ROOT}/fsf_files/group/included_subjects_report.txt",
        ),
    )
    p.add_argument(
        "--runs-report",
        default=os.environ.get(
            "RUNS_REPORT",
            f"{DEFAULT_GPPI_ROOT}/job_files/run_PPI_2nd_level/included_runs_report.txt",
        ),
    )
    p.add_argument(
        "--submission-report",
        default=os.environ.get(
            "SUBMISSION_REPORT",
            f"{DEFAULT_GPPI_ROOT}/job_files/run_PPI_3rd_level/submission_inclusion_report.txt",
        ),
    )
    p.add_argument(
        "--report-dir",
        default=os.environ.get("REPORT_DIR", f"{DEFAULT_GPPI_ROOT}/3rd level report"),
    )
    args = p.parse_args()

    for name, val in (
        ("bids-root", args.bids_root),
        ("gppi-root", args.gppi_root),
        ("subject-csv", args.subject_csv),
        ("inclusion-report", args.inclusion_report),
        ("report-dir", args.report_dir),
    ):
        if not val:
            print(f"ERROR: --{name} is required", file=sys.stderr)
            return 1

    bids_root = Path(args.bids_root)
    gppi_root = Path(args.gppi_root)
    timecourse_root = Path(args.timecourse_root)
    feat_root = bids_root / "derivatives/fslFeat"
    seeds_root = bids_root / "derivatives/gppi_seeds"
    report_dir = Path(args.report_dir)
    report_path = report_dir / "3rd_level_inclusion_report.txt"

    subject_df = load_subject_csv(Path(args.subject_csv))
    inclusion_sections = parse_inclusion_report(Path(args.inclusion_report))
    runs_report = Path(args.runs_report) if args.runs_report else None
    submission_report = (
        Path(args.submission_report) if args.submission_report else None
    )

    write_report(
        report_path,
        subject_df,
        inclusion_sections,
        bids_root,
        feat_root,
        seeds_root,
        timecourse_root,
        gppi_root,
        Path(args.inclusion_report),
        runs_report,
        submission_report,
    )
    print(f">> wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
