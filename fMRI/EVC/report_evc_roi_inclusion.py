#!/usr/bin/env python3
"""
Inclusion report for EVC functional ROI pipeline (06_make_evc_roi.sh).

Lists subjects with successful MNI EVC ROI outputs and traces missing derivatives
for failures across the pipeline (events, regressors, FEAT, anat masks, warps).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EVC_SESSION = "ses-V2"
EVC_TASK = "task-EVCLoc_run-1"
FEAT_SUFFIX = "analysis-1stROI_space-T1w"
FEAT_BASENAME = f"{EVC_SESSION}_{EVC_TASK}_{FEAT_SUFFIX}"
ANAT_SESSIONS = ("ses-V2", "ses-V1")
EVENT_TYPES = ("TLBR", "TRBL", "buttonPress")


def load_subjects(csv_path: Path, columns_csv: str, single_column: str) -> tuple[list[str], str]:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(csv_path, sep=";")

    if single_column:
        cols = [single_column]
        desc = single_column
    else:
        cols = [c.strip() for c in columns_csv.split(",") if c.strip()]
        desc = " OR ".join(cols)

    if not cols:
        raise ValueError("set SUBJECT_CSV_COLUMNS or SUBJECT_CSV_COLUMN")

    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"missing column {col} in {csv_path}")
        mask = mask | df[col].astype(str).str.upper().eq("TRUE")

    return [str(x) for x in df.loc[mask, "sub_code"].tolist()], desc


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def roi_success(evc_rois_root: Path, sub_code: str) -> tuple[bool, Path]:
    sub_id = f"sub-{sub_code}"
    roi_mni = evc_rois_root / sub_id / f"{sub_id}_evc_300_V1V2.nii.gz"
    return roi_mni.is_file() and roi_mni.stat().st_size > 0, roi_mni


def bids_t1w(bids_root: Path, sub_code: str, ses: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return bids_root / sub_id / ses / "anat" / f"{sub_id}_{ses}_acq-anat_run-1_T1w.nii.gz"


def fmriprep_evcloc_bold(fmriprep_root: Path, sub_code: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        fmriprep_root
        / sub_id
        / EVC_SESSION
        / "func"
        / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_space-T1w_desc-preproc_bold.nii.gz"
    )


def fmriprep_mni_boldref_candidates(fmriprep_root: Path, sub_code: str) -> list[Path]:
    sub_id = f"sub-{sub_code}"
    func_dir = fmriprep_root / sub_id / EVC_SESSION / "func"
    prefix = f"{sub_id}_{EVC_SESSION}_{EVC_TASK}"
    return [
        func_dir / f"{prefix}_space-MNI152NLin2009cAsym_boldref.nii.gz",
        func_dir / f"{sub_id}_{EVC_SESSION}_task-VG_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz",
        func_dir / f"{sub_id}_ses-V1_task-Dur_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz",
    ]


def fmriprep_mni_xfm_candidates(fmriprep_root: Path, sub_code: str) -> list[tuple[str, Path]]:
    sub_id = f"sub-{sub_code}"
    out: list[tuple[str, Path]] = []
    v2 = (
        fmriprep_root
        / sub_id
        / EVC_SESSION
        / "anat"
        / f"{sub_id}_{EVC_SESSION}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )
    v1 = (
        fmriprep_root
        / sub_id
        / "ses-V1"
        / "anat"
        / f"{sub_id}_ses-V1_acq-anat_run-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )
    return [("ses-V2", v2), ("ses-V1", v1)]


def events_qc_path(bids_root: Path, sub_code: str) -> Path:
    return (
        bids_root
        / "derivatives/logfilechecks"
        / f"sub-{sub_code}_ses-V2-EVCLoc_errorFlags.csv"
    )


def events_tsv_path(bids_root: Path, sub_code: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / sub_id
        / EVC_SESSION
        / "func"
        / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_events.tsv"
    )


def regressor_ev(regressor_root: Path, sub_code: str, event_type: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        regressor_root
        / sub_id
        / EVC_SESSION
        / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_{event_type}_EV.txt"
    )


def confound_ev(regressor_root: Path, sub_code: str) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        regressor_root
        / sub_id
        / EVC_SESSION
        / "confound_event_files"
        / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_confounds.txt"
    )


def regressor_status_path(bids_root: Path, sub_code: str) -> Path:
    return (
        bids_root
        / "derivatives/logfilechecks"
        / f"sub-{sub_code}_ses-V2-EVCLoc_regressor_status.csv"
    )


def beh_log_glob(raw_dir: Path, sub_code: str) -> str:
    return str(
        raw_dir
        / sub_code
        / f"{sub_code}_MR_V2"
        / "RESOURCES"
        / "BEHEVCLoc"
        / f"{sub_code}E*_Beh_EVCLoc.csv"
    )


def trace_subject(
    *,
    sub_code: str,
    bids_root: Path,
    fmriprep_root: Path,
    fslfeat_root: Path,
    evc_rois_root: Path,
    freesurfer_root: Path,
    regressor_root: Path,
    raw_dir: Path,
    evc_code_root: Path,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    lines: list[str] = []

    ok, roi_mni = roi_success(evc_rois_root, sub_code)
    if ok:
        lines.append(f"[EVC ROI] OK: {roi_mni}")
        marker = evc_code_root / "job_files/run_make_evc_roi" / sub_code / f"{sub_id}_make_evc_roi.run"
        if marker.is_file() and "status=OK" in marker.read_text(encoding="utf-8", errors="replace"):
            lines.append(f"  job marker: status=OK ({marker})")
        return lines

    lines.append(f"[EVC ROI] missing or empty: {roi_mni}")

    marker = evc_code_root / "job_files/run_make_evc_roi" / sub_code / f"{sub_id}_make_evc_roi.run"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8", errors="replace")
        if "status=FAILED" in text or "JOB FAILED" in text:
            lines.append(f"  [06 job] FAILED (see {marker})")
        elif "status=SUBMITTED" in text:
            lines.append(f"  [06 job] submitted but ROI not written ({marker})")
    else:
        lines.append("  [06 job] no make_evc_roi job marker (06_make_evc_roi.sh not submitted?)")

    anat_v1v2 = evc_rois_root / sub_id / "anat" / f"{sub_id}_V1V2.nii.gz"
    anat_done = evc_rois_root / sub_id / "anat" / f".{sub_id}_V1V2_T1w.done"
    if anat_v1v2.is_file():
        lines.append(f"  [anat mask] OK: {anat_v1v2}")
    else:
        lines.append(f"  [anat mask] missing: {anat_v1v2}")
        fs_dir = freesurfer_root / sub_id
        if not (fs_dir / "scripts/recon-all.done").is_file():
            lines.append(f"  [FreeSurfer] missing recon-all.done: {fs_dir}")
        for ses in ANAT_SESSIONS:
            t1w = bids_t1w(bids_root, sub_code, ses)
            if t1w.is_file():
                lines.append(f"  [BIDS T1w {ses}] OK: {t1w}")
            else:
                lines.append(f"  [BIDS T1w {ses}] missing: {t1w}")
        anat_marker = evc_code_root / "job_files/run_make_anat_masks" / sub_code / f"{sub_id}_anat_masks.run"
        if anat_marker.is_file():
            lines.append(f"  [05 anat job] marker: {anat_marker}")
        if anat_done.is_file():
            lines.append(f"  [05 anat] done marker exists but V1V2.nii.gz missing")

    feat_path = fslfeat_root / "EVC" / sub_id / f"{sub_id}_{FEAT_BASENAME}.feat"
    if not feat_path.is_dir():
        lines.append(f"  [FEAT] missing: {feat_path}")
    elif not report_ok(feat_path):
        lines.append(f"  [FEAT] bad/missing report.html: {feat_path}")
    else:
        lines.append(f"  [FEAT] directory OK: {feat_path}")
        for z in ("zstat4.nii.gz", "zstat5.nii.gz"):
            zp = feat_path / "stats" / z
            if not zp.is_file():
                lines.append(f"  [FEAT] missing stats/{z}")
        mean_func = feat_path / "mean_func.nii.gz"
        bold = fmriprep_evcloc_bold(fmriprep_root, sub_code)
        if not mean_func.is_file() and not bold.is_file():
            lines.append(f"  [func ref] missing mean_func and: {bold}")
        feat_marker = evc_code_root / "job_files/run_EVC_1st_level" / sub_code / f"{sub_id}_EVCLoc_{FEAT_SUFFIX}.run"
        if not feat_marker.is_file():
            lines.append(f"  [04 FEAT job] no marker: {feat_marker}")

    mni_refs = fmriprep_mni_boldref_candidates(fmriprep_root, sub_code)
    if not any(p.is_file() for p in mni_refs):
        lines.append("  [MNI boldref] missing (checked):")
        for p in mni_refs:
            lines.append(f"    - {p}")

    if not any(p.is_file() for _, p in fmriprep_mni_xfm_candidates(fmriprep_root, sub_code)):
        lines.append("  [MNI warp] missing transform (checked):")
        for label, p in fmriprep_mni_xfm_candidates(fmriprep_root, sub_code):
            lines.append(f"    - {label}: {p}")

    qc = events_qc_path(bids_root, sub_code)
    if not qc.is_file():
        lines.append(f"  [events 01] missing QC: {qc}")
    else:
        qc_df = pd.read_csv(qc)
        if qc_df.empty:
            lines.append(f"  [events 01] empty QC: {qc}")
        else:
            status = str(qc_df.iloc[0].get("status", "")).upper()
            note = str(qc_df.iloc[0].get("note", ""))
            lines.append(f"  [events 01] QC status={status}: {note} ({qc})")

    etsv = events_tsv_path(bids_root, sub_code)
    if not etsv.is_file():
        lines.append(f"  [events] missing: {etsv}")

    reg_stat = regressor_status_path(bids_root, sub_code)
    if reg_stat.is_file():
        rs = pd.read_csv(reg_stat)
        if not rs.empty:
            st = str(rs.iloc[0].get("status", ""))
            note = str(rs.iloc[0].get("note", ""))
            if st.upper() not in ("PASS", "WARN"):
                lines.append(f"  [regressors 02] status={st}: {note}")

    for ev in EVENT_TYPES:
        ev_path = regressor_ev(regressor_root, sub_code, ev)
        if not ev_path.is_file():
            lines.append(f"  [regressor EV] missing {ev}: {ev_path}")

    conf = confound_ev(regressor_root, sub_code)
    if not conf.is_file():
        lines.append(f"  [confound EV] missing: {conf}")

    func_dir = fmriprep_root / sub_id / EVC_SESSION / "func"
    conf_tsvs = [
        func_dir / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_desc-confounds_timeseries.tsv",
        func_dir / f"{sub_id}_{EVC_SESSION}_{EVC_TASK}_desc-confounds_regressors.tsv",
    ]
    if not any(p.is_file() for p in conf_tsvs):
        lines.append("  [fmriprep] missing confounds TSV:")
        for p in conf_tsvs:
            lines.append(f"    - {p}")

    if raw_dir.is_dir():
        import glob

        logs = glob.glob(beh_log_glob(raw_dir, sub_code))
        if not logs:
            lines.append(f"  [behaviour log] none matched: {beh_log_glob(raw_dir, sub_code)}")

    return lines


def write_report(
    report_path: Path,
    candidates: list[str],
    included: list[str],
    excluded_traces: dict[str, list[str]],
    filter_desc: str,
    evc_rois_root: Path,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# EVC ROI inclusion report",
        f"# date={stamp}",
        f"# filter={filter_desc}",
        f"# success_criterion={evc_rois_root}/sub-XX/sub-XX_evc_300_V1V2.nii.gz",
        f"# evc_rois_root={evc_rois_root}",
        f"# n_csv_candidates={len(candidates)}",
        f"# n_included={len(included)}",
        f"# n_excluded={len(candidates) - len(included)}",
        "",
        "# Included subjects (MNI EVC ROI created)",
    ]
    if included:
        for code in sorted(included):
            _, roi = roi_success(evc_rois_root, code)
            lines.append(f"sub-{code}\t{roi}")
    else:
        lines.append("(none)")

    lines.extend(["", "# Excluded / incomplete subjects (trace-back)"])
    excluded = [c for c in candidates if c not in included]
    if not excluded:
        lines.append("(none)")
    else:
        for code in sorted(excluded):
            lines.extend(["", f"### sub-{code}"])
            lines.extend(excluded_traces.get(code, ["  (no trace generated)"]))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="EVC ROI inclusion report")
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p.add_argument("--code-path", default=os.environ.get("CODE_PATH", ""))
    p.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p.add_argument("--report-dir", default=os.environ.get("REPORT_DIR", ""))
    p.add_argument("--evc-rois-root", default=os.environ.get("EVC_ROIS_ROOT", ""))
    p.add_argument("--raw-dir", default=os.environ.get("RAW_DIR", ""))
    p.add_argument("--subject-csv-columns", default=os.environ.get("SUBJECT_CSV_COLUMNS", ""))
    p.add_argument("--subject-csv-column", default=os.environ.get("SUBJECT_CSV_COLUMN", ""))
    args = p.parse_args()

    for name, val in (
        ("bids-root", args.bids_root),
        ("code-path", args.code_path),
        ("subject-csv", args.subject_csv),
        ("report-dir", args.report_dir),
    ):
        if not val:
            print(f"ERROR: --{name} is required", file=sys.stderr)
            return 1

    bids_root = Path(args.bids_root)
    code_path = Path(args.code_path)
    evc_code_root = code_path / "EVC"
    fmriprep_root = bids_root / "derivatives/fmriprep"
    fslfeat_root = bids_root / "derivatives/fslFeat"
    evc_rois_root = Path(args.evc_rois_root) if args.evc_rois_root else bids_root / "derivatives/evc_rois"
    regressor_root = bids_root / "derivatives/regressoreventfiles"
    freesurfer_root = bids_root / "derivatives/freesurfer"
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path("/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII")
    report_dir = Path(args.report_dir)
    report_path = report_dir / "evc_roi_inclusion_report.txt"

    columns_csv = args.subject_csv_columns or "SYNCHRONY_min_seen"
    candidates, filter_desc = load_subjects(
        Path(args.subject_csv),
        columns_csv,
        args.subject_csv_column.strip(),
    )

    included: list[str] = []
    excluded_traces: dict[str, list[str]] = {}

    for code in candidates:
        trace = trace_subject(
            sub_code=code,
            bids_root=bids_root,
            fmriprep_root=fmriprep_root,
            fslfeat_root=fslfeat_root,
            evc_rois_root=evc_rois_root,
            freesurfer_root=freesurfer_root,
            regressor_root=regressor_root,
            raw_dir=raw_dir,
            evc_code_root=evc_code_root,
        )
        if trace and trace[0].startswith("[EVC ROI] OK"):
            included.append(code)
        else:
            excluded_traces[code] = trace

    write_report(
        report_path,
        candidates,
        included,
        excluded_traces,
        filter_desc,
        evc_rois_root,
    )
    print(f">> wrote {report_path}")
    print(f">> included={len(included)} excluded={len(candidates) - len(included)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
