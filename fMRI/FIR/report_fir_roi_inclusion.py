#!/usr/bin/env python3
"""
Inclusion report for FIR/10_average_FIR_in_FFA_and_LOC.sh.

Lists subjects included in ROI-averaged CSV outputs and traces missing derivatives
for combined FIR pipeline (FFA + LOC)
(BASELINE_min_seen_unseen cohort for all variants).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SESSION = "ses-V2"
ANAT_SESSIONS = ("ses-V1", "ses-V2")
TASK_RUN_PREFIX = "ses-V2_task-VG_run-"
SPACE = "MNI152NLin2009cAsym"
MAX_RUNS = 8
SEEN_COPES = range(1, 15)
UNSEEN_COPES = range(15, 29)
ALL_COPES = range(1, 29)

FIR_EV_UPSTREAM = (
    "probedSeenFaceLeft_EV.txt",
    "probedSeenFaceRight_EV.txt",
    "probedSeenObjectLeft_EV.txt",
    "probedSeenObjectRight_EV.txt",
    "probedUnseenFaceLeft_EV.txt",
    "probedUnseenFaceRight_EV.txt",
    "probedUnseenObjectLeft_EV.txt",
    "probedUnseenObjectRight_EV.txt",
)


@dataclass(frozen=True)
class FirVariantCheck:
    label: str
    filter_col: str
    fslfeat_subdir: str
    feat_1st_suffix: str
    feat_2nd_stem: str
    csv_suffix: str
    csv_names: tuple[str, ...]
    ev_suffixes: tuple[str, ...]
    check_ffa_mask: bool
    check_loc_mask: bool


ROI_VARIANT_CHECKS: tuple[FirVariantCheck, ...] = (
    FirVariantCheck(
        label="combined (FIR → FFA + LOC)",
        filter_col="BASELINE_min_seen_unseen",
        fslfeat_subdir="FIR",
        feat_1st_suffix="analysis-1stGLM_FIR_space-MNI152NLin2009cAsym",
        feat_2nd_stem="analysis-2ndGLM_FIR-MNI152NLin2009cAsym",
        csv_suffix="BASELINE_min_seen_unseen",
        csv_names=(
            "fir_FFA_seen_bins_BASELINE_min_seen_unseen.csv",
            "fir_FFA_unseen_bins_BASELINE_min_seen_unseen.csv",
            "fir_LOC_seen_bins_BASELINE_min_seen_unseen.csv",
            "fir_LOC_unseen_bins_BASELINE_min_seen_unseen.csv",
        ),
        ev_suffixes=("probedSeen_Shifted.txt", "probedUnseen_Shifted.txt"),
        check_ffa_mask=True,
        check_loc_mask=True,
    ),
)


def load_subject_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df


def subjects_for_filter(df: pd.DataFrame, filter_col: str) -> list[str]:
    if filter_col not in df.columns:
        raise ValueError(f"missing column {filter_col}")
    sub_df = df.loc[df[filter_col].astype(str).str.upper().eq("TRUE")]
    return [str(x) for x in sub_df["sub_code"].tolist()]


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def run_label(run: int) -> str:
    return f"{TASK_RUN_PREFIX}{run}"


def seed_masks(seeds_root: Path, sub_code: str) -> dict[str, Path]:
    sub_id = f"sub-{sub_code}"
    seed_dir = seeds_root / sub_id
    return {
        "FFA": seed_dir
        / f"{sub_id}_rel_irrel_bh_face_FFA_n_voxels_300_{SPACE}.nii.gz",
        "LOC": seed_dir
        / f"{sub_id}_rel_irrel_bh_object_LOC_n_voxels_300_{SPACE}.nii.gz",
    }


def fir_cope_map(feat_root: Path, cfg: FirVariantCheck, sub_code: str, cope: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        feat_root
        / cfg.fslfeat_subdir
        / sub_id
        / f"{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}.gfeat"
        / f"cope{cope}.feat"
        / "stats"
        / "cope1.nii.gz"
    )


def second_level_parent(
    feat_root: Path, cfg: FirVariantCheck, sub_code: str
) -> tuple[Path | None, str]:
    sub_id = f"sub-{sub_code}"
    stem = f"{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}"
    for ext in (".gfeat", ".feat"):
        parent = feat_root / cfg.fslfeat_subdir / sub_id / f"{stem}{ext}"
        if parent.is_dir():
            return parent, ext
    return None, ""


def first_level_feat(feat_root: Path, cfg: FirVariantCheck, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        feat_root
        / cfg.fslfeat_subdir
        / sub_id
        / f"{sub_id}_{SESSION}_task-VG_run-{run}_{cfg.feat_1st_suffix}.feat"
    )


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


def fmriprep_confounds(bids_root: Path, sub_code: str, run: int) -> list[Path]:
    sub_id = f"sub-{sub_code}"
    base = bids_root / "derivatives/fmriprep" / sub_id / SESSION / "func"
    return [
        base / f"{sub_id}_{run_label(run)}_desc-confounds_timeseries.tsv",
        base / f"{sub_id}_{run_label(run)}_desc-confounds_regressors.tsv",
    ]


def fmriprep_structural(bids_root: Path, sub_code: str) -> dict[str, list[Path]]:
    sub_id = f"sub-{sub_code}"
    out: dict[str, list[Path]] = {}
    for ses in ANAT_SESSIONS:
        anat = bids_root / "derivatives/fmriprep" / sub_id / ses / "anat"
        paths = [
            anat / f"{sub_id}_{ses}_T1w.nii.gz",
            anat / f"{sub_id}_{ses}_run-1_T1w.nii.gz",
            anat / f"{sub_id}_{ses}_desc-preproc_T1w.nii.gz",
            anat / f"{sub_id}_{ses}_run-1_desc-preproc_T1w.nii.gz",
            anat / f"{sub_id}_{ses}_{SPACE}_desc-preproc_T1w.nii.gz",
        ]
        if anat.is_dir():
            paths.extend(sorted(anat.glob(f"{sub_id}*T1w*.nii.gz")))
        out[ses] = paths
    return out


def append_structural_trace(lines: list[str], bids_root: Path, sub_code: str) -> None:
    for ses, paths in fmriprep_structural(bids_root, sub_code).items():
        found = [p for p in paths if p.is_file()]
        if found:
            lines.append(f"[structural {ses}] OK: {found[0]}")
        else:
            lines.append(f"[structural {ses}] missing T1w (checked):")
            for p in paths[:5]:
                lines.append(f"  - {p}")


def confound_ev(bids_root: Path, sub_code: str, run: int) -> Path:
    sub_id = f"sub-{sub_code}"
    return (
        bids_root
        / "derivatives/regressoreventfiles"
        / sub_id
        / SESSION
        / "confound_event_files"
        / f"{sub_id}_{run_label(run)}_confounds.txt"
    )


def fir_ev_file(bids_root: Path, sub_code: str, run: int, suffix: str) -> Path:
    sub_id = f"sub-{sub_code}"
    run_base = f"{sub_id}-{SESSION}_task-VG_run-{run}"
    return (
        bids_root
        / "derivatives/regressoreventfiles"
        / sub_id
        / SESSION
        / "FIR"
        / f"{run_base}_{suffix}"
    )


def parse_runs_report(
    path: Path, subj_id: str, variant_name: str
) -> dict[int, tuple[str, str]]:
    if not path.is_file():
        return {}
    runs: dict[int, tuple[str, str]] = {}
    in_section = False
    mode: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## analysis=") and f"subject={subj_id}" in line:
            in_section = f"analysis={variant_name}" in line
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
        m = re.match(r"run-(\d+)\t(.+)", line.strip())
        if not m:
            continue
        run = int(m.group(1))
        rest = m.group(2)
        if mode == "included":
            runs[run] = ("included", rest.split("\t", 1)[0])
        elif mode == "excluded":
            parts = rest.split("\t", 1)
            runs[run] = ("excluded", parts[0] if parts else rest)
    return runs


def parse_error_log_for_subject(
    error_log: Path | None, sub_code: str, pass_hint: str = ""
) -> list[str]:
    if error_log is None or not error_log.is_file():
        return []
    lines = []
    for line in error_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Skip ") or sub_code not in line:
            continue
        if pass_hint and pass_hint not in line and "pass combined" not in line:
            continue
        lines.append(f"  [10 runtime] {line}")
    return lines


def read_included_from_csv(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, sep=";")
    if "sub_code" not in df.columns:
        return set()
    return {str(x) for x in df["sub_code"].dropna()}


def read_included_from_csvs(out_dir: Path, csv_names: tuple[str, ...]) -> set[str]:
    included: set[str] = set()
    for name in csv_names:
        included.update(read_included_from_csv(out_dir / name))
    return included


def read_included_per_csv(
    out_dir: Path, csv_names: tuple[str, ...]
) -> dict[str, set[str]]:
    return {name: read_included_from_csv(out_dir / name) for name in csv_names}


def summarize_exclusion_reason(trace: list[str], runtime: list[str]) -> str:
    """Short label for why a subject is missing from ROI summary tables."""
    blob = "\n".join(trace + runtime)
    reasons: list[str] = []
    if "[seed mask FFA] missing" in blob:
        reasons.append("FFA seed mask")
    if "[seed mask LOC] missing" in blob:
        reasons.append("LOC seed mask")
    if "[structural " in blob and "missing T1w" in blob:
        reasons.append("anatomical (fmriprep T1w)")
    if "[2nd level" in blob and "missing:" in blob:
        reasons.append("2nd-level FIR .gfeat/.feat")
    if "bad/missing report.html" in blob and "[2nd level" in blob:
        reasons.append("2nd-level FEAT failed (report.html)")
    if "missing cope maps" in blob:
        reasons.append("2nd-level FIR cope maps (stats/cope1.nii.gz)")
    if "missing 1st-level .feat" in blob:
        reasons.append("1st-level FIR .feat")
    if "excluded in 06" in blob:
        reasons.append("1st-level run excluded (2nd-level job report)")
    if "missing FIR map cope" in blob or "fslstats failed" in blob:
        reasons.append("FIR cope map or ROI extraction (script 10)")
    if "missing ROI mask" in blob:
        reasons.append("ROI mask (script 10)")
    if "[fmriprep] missing BOLD" in blob:
        reasons.append("fmriprep BOLD")
    if "[FIR event] missing" in blob or "[event EV]" in blob:
        reasons.append("FIR event files")
    if not reasons and runtime:
        return "ROI averaging failed at runtime (see [10 runtime] below)"
    if not reasons:
        return "see trace-back below"
    return "; ".join(dict.fromkeys(reasons))


def check_cope_maps(
    feat_root: Path, cfg: FirVariantCheck, sub_code: str
) -> tuple[list[int], list[Path]]:
    missing_copes: list[int] = []
    missing_paths: list[Path] = []
    for cope in ALL_COPES:
        path = fir_cope_map(feat_root, cfg, sub_code, cope)
        if not path.is_file():
            missing_copes.append(cope)
            missing_paths.append(path)
    return missing_copes, missing_paths


def trace_first_level(
    feat_root: Path,
    bids_root: Path,
    fir_root: Path,
    cfg: FirVariantCheck,
    sub_code: str,
    runs_report: Path | None,
) -> list[str]:
    sub_id = f"sub-{sub_code}"
    variant = cfg.fslfeat_subdir
    marker_glm = "analysis-1stGLM_FIR-MNI152NLin2009cAsym"

    lines: list[str] = []
    run_info = parse_runs_report(runs_report, sub_id, variant) if runs_report else {}

    for run in range(1, MAX_RUNS + 1):
        feat_path = first_level_feat(feat_root, cfg, sub_code, run)
        if run in run_info and run_info[run][0] == "excluded":
            lines.append(f"  run-{run}: excluded in 06 ({run_info[run][1]})")
            lines.extend(trace_run_inputs(bids_root, sub_code, run, cfg.ev_suffixes))
            continue
        if not feat_path.is_dir():
            lines.append(f"  run-{run}: missing 1st-level .feat: {feat_path}")
            lines.extend(trace_run_inputs(bids_root, sub_code, run, cfg.ev_suffixes))
        elif not report_ok(feat_path):
            lines.append(f"  run-{run}: error in report.html: {feat_path}")
        elif not (feat_path / "reg" / "example_func2standard.mat").is_file():
            lines.append(
                f"  run-{run}: missing identity reg (04_create_identity_reg_for_2nd_lvl.sh?)"
            )
        else:
            lines.append(f"  run-{run}: 1st-level OK")

        marker = (
            fir_root
            / "job_files/run_FIR_1st_level"
            / variant
            / sub_code
            / f"{sub_id}_{SESSION}_task-VG_run-{run}_{marker_glm}.run"
        )
        if not marker.is_file():
            lines.append(f"  run-{run}: missing 1st-level job marker: {marker}")
    return lines


def trace_run_inputs(
    bids_root: Path, sub_code: str, run: int, ev_suffixes: tuple[str, ...]
) -> list[str]:
    lines: list[str] = []
    bold = fmriprep_bold(bids_root, sub_code, run)
    if not bold.is_file():
        lines.append(f"    [fmriprep] missing BOLD: {bold}")
    confs = fmriprep_confounds(bids_root, sub_code, run)
    if not any(p.is_file() for p in confs):
        lines.append("    [fmriprep] missing confounds TSV")
    conf_txt = confound_ev(bids_root, sub_code, run)
    if not conf_txt.is_file():
        lines.append(f"    [confound EV] missing: {conf_txt}")
    for suf in ev_suffixes:
        p = fir_ev_file(bids_root, sub_code, run, suf)
        if not p.is_file():
            lines.append(f"    [FIR event] missing: {p}")
    missing_up = []
    for suf in FIR_EV_UPSTREAM:
        up = (
            bids_root
            / "derivatives/regressoreventfiles"
            / f"sub-{sub_code}"
            / SESSION
            / f"sub-{sub_code}_{run_label(run)}_{suf}"
        )
        if not up.is_file():
            missing_up.append(up)
    if missing_up:
        lines.append("    [event EV] upstream for FIR (examples):")
        for p in missing_up[:4]:
            lines.append(f"      - {p}")
    return lines


def trace_subject(
    feat_root: Path,
    bids_root: Path,
    seeds_root: Path,
    fir_root: Path,
    cfg: FirVariantCheck,
    sub_code: str,
    runs_report: Path | None,
    error_log: Path | None,
) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    variant = cfg.fslfeat_subdir
    sub_id = f"sub-{sub_code}"

    masks = seed_masks(seeds_root, sub_code)
    if cfg.check_ffa_mask:
        path = masks["FFA"]
        if not path.is_file():
            lines.append(f"[seed mask FFA] missing: {path}")
            ok = False
        else:
            lines.append(f"[seed mask FFA] OK: {path}")
    if cfg.check_loc_mask:
        path = masks["LOC"]
        if not path.is_file():
            lines.append(f"[seed mask LOC] missing: {path}")
            ok = False
        else:
            lines.append(f"[seed mask LOC] OK: {path}")

    append_structural_trace(lines, bids_root, sub_code)

    parent, ext = second_level_parent(feat_root, cfg, sub_code)
    if parent is None:
        lines.append(
            f"[2nd level {variant}] missing: "
            f"{feat_root / variant / sub_id}/{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}.gfeat"
        )
        ok = False
        marker = (
            fir_root
            / "job_files/run_FIR_2nd_level"
            / variant
            / sub_code
            / f"{sub_id}_{SESSION}_task-VG_{cfg.feat_2nd_stem}.run"
        )
        if not marker.is_file():
            lines.append(f"  2nd-level job marker missing (06_run_FIR_2nd_level_VG.sh?): {marker}")
        lines.append("  1st-level runs:")
        lines.extend(trace_first_level(feat_root, bids_root, fir_root, cfg, sub_code, runs_report))
        lines.extend(parse_error_log_for_subject(error_log, sub_code))
        return False, lines

    if not report_ok(parent):
        lines.append(f"[2nd level {variant}] bad/missing report.html ({ext}): {parent}")
        ok = False

    missing_copes, missing_paths = check_cope_maps(feat_root, cfg, sub_code)
    if missing_copes:
        lines.append(
            f"[2nd level {variant}] missing cope maps (stats/cope1.nii.gz): "
            f"{len(missing_copes)} of 28"
        )
        if len(missing_copes) <= 8:
            for c, p in zip(missing_copes, missing_paths):
                label = "seen" if c in SEEN_COPES else "unseen"
                lines.append(f"  cope{c} ({label}): {p}")
        else:
            for c, p in zip(missing_copes[:5], missing_paths[:5]):
                lines.append(f"  cope{c}: {p}")
            lines.append(f"  ... and {len(missing_copes) - 5} more")
        ok = False
    else:
        lines.append(f"[2nd level {variant}] all 28 cope maps present under {parent}")

    lines.extend(parse_error_log_for_subject(error_log, sub_code))
    return ok, lines


def write_report(
    report_path: Path,
    sections: list[
        tuple[
            FirVariantCheck,
            list[str],
            dict[str, set[str]],
            dict[str, list[str]],
            Path | None,
        ]
    ],
    out_dir: Path,
    error_log: Path | None,
) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# FIR ROI averaging inclusion report",
        f"# date={stamp}",
        f"# script=10_average_FIR_in_FFA_and_LOC.sh",
        f"# outputs_dir={out_dir}",
        f"# error_log={error_log or ''}",
        "",
    ]

    for cfg, candidates, per_csv, excluded_traces, _err in sections:
        csv_sets = [per_csv[name] for name in cfg.csv_names]
        in_any = set().union(*csv_sets) if csv_sets else set()
        in_all = set.intersection(*csv_sets) if csv_sets else set()
        candidate_set = set(candidates)

        lines.extend(
            [
                f"## {cfg.label}",
                f"# filter={cfg.filter_col}",
                f"# fslFeat_subdir={cfg.fslfeat_subdir}",
                f"# n_candidates={len(candidates)}",
                f"# n_in_any_summary_table={len(in_any & candidate_set)}",
                f"# n_in_all_four_tables={len(in_all & candidate_set)}",
                "",
                "# Inclusion per summary table (from CSV row counts)",
            ]
        )
        for name in cfg.csv_names:
            subs = sorted(per_csv[name] & candidate_set)
            lines.append(f"## table={name}\tn_included={len(subs)}")
            if subs:
                for code in subs:
                    lines.append(f"sub-{code}")
            else:
                lines.append("(none)")
            lines.append("")

        partial = sorted((in_any - in_all) & candidate_set)
        lines.append("# Subjects in at least one table but not all four")
        if partial:
            for code in partial:
                present = [n for n in cfg.csv_names if code in per_csv[n]]
                missing_tables = [n for n in cfg.csv_names if code not in per_csv[n]]
                lines.append(
                    f"sub-{code}\tin={','.join(present)}\tnot_in={','.join(missing_tables)}"
                )
        else:
            lines.append("(none)")
        lines.append("")

        lines.append("# Subjects in all four summary tables")
        if in_all & candidate_set:
            for code in sorted(in_all & candidate_set):
                lines.append(f"sub-{code}")
        else:
            lines.append("(none)")
        lines.append("")

        lines.append("# Excluded subjects (not in any summary table)")
        excluded = sorted(candidate_set - in_any)
        if not excluded:
            lines.append("(none)")
        else:
            for code in excluded:
                trace = excluded_traces.get(code, ["  (no trace generated)"])
                runtime = parse_error_log_for_subject(error_log, code)
                summary = summarize_exclusion_reason(trace, runtime)
                lines.extend(
                    [
                        "",
                        f"### sub-{code}",
                        f"# summary={summary}",
                    ]
                )
                lines.extend(trace)
                lines.extend(runtime)
        lines.append("")

        lines.append("# Excluded from some tables only (partial; trace-back)")
        if partial:
            for code in partial:
                trace = excluded_traces.get(code, [])
                runtime = parse_error_log_for_subject(error_log, code)
                summary = summarize_exclusion_reason(trace, runtime)
                lines.extend(
                    [
                        "",
                        f"### sub-{code}",
                        f"# summary={summary}",
                    ]
                )
                lines.extend(trace)
                lines.extend(runtime)
        else:
            lines.append("(none)")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="FIR ROI averaging inclusion report")
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", ""))
    p.add_argument("--fir-root", default=os.environ.get("FIR_CODE_ROOT", ""))
    p.add_argument("--subject-csv", default=os.environ.get("SUBJECT_CSV", ""))
    p.add_argument("--report-dir", default=os.environ.get("REPORT_DIR", ""))
    p.add_argument("--out-dir", default=os.environ.get("FIR_DERIV_ROOT", ""))
    p.add_argument("--error-log", default=os.environ.get("ERROR_LOG", ""))
    args = p.parse_args()

    for name, val in (
        ("bids-root", args.bids_root),
        ("fir-root", args.fir_root),
        ("subject-csv", args.subject_csv),
    ):
        if not val:
            print(f"ERROR: --{name} is required", file=sys.stderr)
            return 1

    bids_root = Path(args.bids_root)
    fir_root = Path(args.fir_root)
    feat_root = bids_root / "derivatives/fslFeat"
    seeds_root = bids_root / "derivatives/gppi_seeds"
    out_dir = (
        Path(args.out_dir) if args.out_dir else bids_root / "derivatives/fir"
    )
    report_dir = Path(args.report_dir) if args.report_dir else out_dir
    report_path = out_dir / "fir_roi_averaging_inclusion_report.txt"
    error_log = Path(args.error_log) if args.error_log else out_dir / "fir_averaging_error_log.txt"

    df = load_subject_csv(Path(args.subject_csv))
    err_path = error_log if error_log.is_file() else None
    sections: list[
        tuple[
            FirVariantCheck,
            list[str],
            dict[str, set[str]],
            dict[str, list[str]],
            Path | None,
        ]
    ] = []

    for cfg in ROI_VARIANT_CHECKS:
        runs_report = (
            fir_root
            / "job_files/run_FIR_2nd_level"
            / f"included_runs_report_{cfg.fslfeat_subdir}.txt"
        )
        candidates = subjects_for_filter(df, cfg.filter_col)
        per_csv = read_included_per_csv(out_dir, cfg.csv_names)
        excluded_traces: dict[str, list[str]] = {}

        for code in candidates:
            _would_include, trace = trace_subject(
                feat_root,
                bids_root,
                seeds_root,
                fir_root,
                cfg,
                code,
                runs_report,
                err_path,
            )
            excluded_traces[code] = trace

        in_any = set().union(*(per_csv[n] for n in cfg.csv_names))
        n_in_any = len(in_any & set(candidates))
        sections.append((cfg, candidates, per_csv, excluded_traces, err_path))
        print(
            f">> {cfg.label}: in_any_table={n_in_any} "
            f"excluded_entirely={len(candidates) - n_in_any} (filter={cfg.filter_col})"
        )

    write_report(report_path, sections, out_dir, err_path)
    print(f">> wrote {report_path}")

    copy_path = report_dir / "fir_roi_averaging_inclusion_report.txt"
    if copy_path.resolve() != report_path.resolve():
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f">> copied to {copy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
