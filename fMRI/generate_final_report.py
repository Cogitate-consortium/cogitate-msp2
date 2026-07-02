#!/usr/bin/env python3
"""
Final 3rd-level failure reports for activation, FIR, and gPPI pipelines.

Reads existing inclusion reports (and optionally refreshes them), traces excluded
subjects via existing pipeline modules, and writes actionable failure summaries
under final_report/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BIDS_ROOT = "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
DEFAULT_CODE_ROOT = f"{DEFAULT_BIDS_ROOT}/code"
DEFAULT_ACTIVATION_ROOT = f"{DEFAULT_CODE_ROOT}/activation"
DEFAULT_FIR_ROOT = f"{DEFAULT_CODE_ROOT}/FIR"
DEFAULT_GPPI_ROOT = f"{DEFAULT_CODE_ROOT}/gPPI_code"
DEFAULT_TIMECOURSE_ROOT = f"{DEFAULT_BIDS_ROOT}/derivatives/gppi_timecourse"

FIR_VARIANTS = ("FIR",)
FIR_FILTER_COL = "BASELINE_min_seen_unseen"
FIR_COPES = list(range(1, 43))

ACTIVATION_ANALYSES_CSV = (
    "seen_face_vs_unseen_face:analysis-2ndGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym:ACTIVATION_min_sf_uf:1:seen_vs_unseen_F\n"
    "seen_object_vs_unseen_object:analysis-2ndGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym:ACTIVATION_min_so_uo:2:seen_vs_unseen_O"
)

# All valid --analysis names (comma-separated for multiple).
ALL_KNOWN_ANALYSES: frozenset[str] = frozenset(
    (
        "seen_face_vs_unseen_face",
        "seen_object_vs_unseen_object",
        "FIR",
        "PPI_LOC",
        "PPI_FFA",
    )
)
ANALYSIS_TO_PIPELINE: dict[str, str] = {
    "seen_face_vs_unseen_face": "activation",
    "seen_object_vs_unseen_object": "activation",
    "FIR": "fir",
    "PPI_LOC": "gppi",
    "PPI_FFA": "gppi",
}


@dataclass
class SubjectFailure:
    subj_id: str
    exclusion_reason: str
    exclusion_path: str
    trace_lines: list[str] = field(default_factory=list)


@dataclass
class AnalysisFailures:
    pipeline: str
    analysis: str
    cope_tag: str | None
    failures: list[SubjectFailure] = field(default_factory=list)


@dataclass
class IndexRow:
    pipeline: str
    analysis: str
    cope_tag: str
    n_failed: int
    report_path: str
    status: str


def parse_analysis_filter(arg: str | None) -> set[str] | None:
    if not arg or not arg.strip():
        return None
    selected = {a.strip() for a in arg.split(",") if a.strip()}
    unknown = selected - ALL_KNOWN_ANALYSES
    if unknown:
        known = ", ".join(sorted(ALL_KNOWN_ANALYSES))
        raise ValueError(
            f"unknown analysis name(s): {', '.join(sorted(unknown))}; "
            f"valid values: {known}"
        )
    return selected


def pipelines_for_analyses(analysis_filter: set[str] | None) -> set[str] | None:
    if analysis_filter is None:
        return None
    return {ANALYSIS_TO_PIPELINE[a] for a in analysis_filter}


def analysis_selected(name: str, analysis_filter: set[str] | None) -> bool:
    return analysis_filter is None or name in analysis_filter


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pipeline_modules(code_root: Path) -> dict[str, Any]:
    activation_dir = code_root / "activation"
    fir_dir = code_root / "FIR"
    gppi_dir = code_root / "gPPI_code"

    fir_roi = load_module("report_fir_roi_inclusion", fir_dir / "report_fir_roi_inclusion.py")
    fir_3rd = load_module("report_fir_3rd_level_inclusion", fir_dir / "report_fir_3rd_level_inclusion.py")
    return {
        "activation": load_module(
            "report_activation_3rd_level_inclusion",
            activation_dir / "report_3rd_level_inclusion.py",
        ),
        "fir_roi": fir_roi,
        "fir": fir_3rd,
        "gppi": load_module(
            "report_gppi_3rd_level_inclusion",
            gppi_dir / "report_3rd_level_inclusion.py",
        ),
        "gppi_3rd": load_module("gppi_3rd_level", gppi_dir / "gppi_3rd_level.py"),
    }


def run_refresh(code_root: Path, pipelines: set[str]) -> None:
    scripts = {
        "activation": code_root / "activation" / "make_3rd_level_inclusion_report.sh",
        "fir": code_root / "FIR" / "make_fir_3rd_level_inclusion_report.sh",
        "gppi": code_root / "gPPI_code" / "make_3rd_level_inclusion_report.sh",
    }
    for key, script in scripts.items():
        if key not in pipelines:
            continue
        if not script.is_file():
            print(f"WARN: refresh script not found: {script}", file=sys.stderr)
            continue
        print(f">> refreshing {key} inclusion report via {script.name}")
        try:
            subprocess.run(
                ["bash", str(script)],
                cwd=str(script.parent),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            print(f"WARN: could not run {script}: {exc}", file=sys.stderr)


def extract_analysis_section(
    full_text: str, analysis: str, cope_tag: str | None = None
) -> str:
    lines = full_text.splitlines()
    captured: list[str] = []
    in_section = False
    for i, line in enumerate(lines):
        if line.startswith("## analysis="):
            if in_section:
                break
            parts = dict(p.split("=", 1) for p in line[3:].split("\t") if "=" in p)
            match_analysis = parts.get("analysis") == analysis
            match_cope = cope_tag is None or parts.get("cope") == cope_tag
            if match_analysis and match_cope:
                in_section = True
            continue
        if in_section:
            if line.startswith("=" * 72) and captured:
                break
            captured.append(line)
    return "\n".join(captured)


def parse_trace_blocks(section_text: str) -> dict[str, list[str]]:
    traces: dict[str, list[str]] = {}
    current_subj: str | None = None
    current_lines: list[str] = []
    for line in section_text.splitlines():
        if line.startswith("### "):
            if current_subj is not None:
                traces[current_subj] = current_lines
            current_subj = line[4:].strip()
            current_lines = []
        elif current_subj is not None:
            current_lines.append(line)
    if current_subj is not None:
        traces[current_subj] = current_lines
    return traces


def infer_rerun_step(
    pipeline: str, exclusion_reason: str, trace_lines: list[str]
) -> tuple[str, str, str]:
    blob = "\n".join([exclusion_reason, *trace_lines])
    runs = sorted({m.group(1) for m in re.finditer(r"run-(\d+)", blob)})

    def scope_for(script: str) -> str:
        base = "subject-level"
        if runs:
            base = f"runs={','.join(runs)}"
        return base

    rules: list[tuple[str, str, str]] = [
        (
            r"\[structural [^\]]+\] missing|missing T1w",
            "fMRIPrep",
            "Re-run fMRIPrep for missing structural image(s).",
        ),
        (
            r"\[fmriprep\]|missing preprocessed BOLD|missing brain mask|missing confounds TSV",
            "fMRIPrep",
            "Re-run fMRIPrep for missing functional preprocessing outputs.",
        ),
    ]

    if pipeline == "fir":
        rules.append(
            (
                r"\[event EV\]|probedSeen_Shifted|probedUnseen_Shifted",
                "FIR/01_create_ev_files.sh",
                "Regenerate FIR event files, then rerun downstream FEAT steps.",
            )
        )
    elif pipeline == "gppi":
        rules.extend(
            [
                (
                    r"\[event EV\]|prefer_Face|prefer_Object",
                    "gPPI_code/01_create_ev_files.sh",
                    "Regenerate gPPI event files, then rerun downstream steps.",
                ),
                (
                    r"\[timecourse|02_make_timecourse_files",
                    "gPPI_code/02_make_timecourse_files.sh",
                    "Extract seed timecourses before 1st-level PPI.",
                ),
                (
                    r"\[seed mask|\[seeds\] missing",
                    "upstream seed extraction",
                    "Regenerate gPPI seed masks / seed directory for this subject.",
                ),
            ]
        )

    rules.extend(
        [
            (
                r"02_create_identity_reg",
                "activation/02_create_identity_reg_for_2nd_lvl.sh",
                "Create identity registration matrices for 2nd-level FEAT.",
            ),
            (
                r"04_create_identity_reg|missing identity reg \(05_create",
                "FIR/04_create_identity_reg_for_2nd_lvl.sh"
                if pipeline == "fir"
                else "gPPI_code/05_create_identity_reg_for_2nd_lvl.sh",
                "Create identity registration matrices for 2nd-level FEAT.",
            ),
            (
                r"05_create_identity_reg",
                "gPPI_code/05_create_identity_reg_for_2nd_lvl.sh",
                "Create identity registration matrices for 2nd-level PPI.",
            ),
            (
                r"08_create_identity_reg|07_create_identity_reg|missing identity reg at \.gfeat root",
                {
                    "gppi": "gPPI_code/08_create_identity_reg_for_3rd_lvl.sh",
                    "fir": "FIR/07_create_identity_reg_for_3rd_lvl.sh",
                }.get(pipeline, "create_identity_reg_for_3rd_lvl.sh"),
                "Create identity registration at 2nd-level .gfeat roots for 3rd-level FEAT.",
            ),
            (
                r"missing 1st-level \.feat|error in report\.html|missing report\.html in|"
                r"04_run_PPI_1st_level|03_run_FIR_1st_level|1st-level job marker missing",
                {
                    "activation": "glm/02_run_fsf_feat_analyses.py",
                    "fir": "FIR/03_run_FIR_1st_level.sh",
                    "gppi": "gPPI_code/04_run_PPI_1st_level.sh",
                }.get(pipeline, "1st-level FEAT"),
                "Fix or re-run 1st-level FEAT for affected run(s), then 2nd/3rd level.",
            ),
            (
                r"\[2nd level\]|missing cope\d+\.feat|bad/missing report\.html \(\.gfeat|"
                r"04_run_2nd_level|06_run_FIR_2nd_level|07_run_PPI_2nd_level|"
                r"2nd-level job marker missing|excluded in 0[47]_run",
                {
                    "activation": "activation/04_run_2nd_level.sh",
                    "fir": "FIR/06_run_FIR_2nd_level_VG.sh",
                    "gppi": "gPPI_code/07_run_PPI_2nd_level_VG.sh",
                }.get(pipeline, "2nd-level FEAT"),
                "Re-run 2nd-level FEAT after 1st-level outputs are OK.",
            ),
            (
                r"not listed in 0[5789] inclusion report|05_exclusion|07_exclusion|08_exclusion|"
                r"08_make_exclusion|09_make_exclusion",
                {
                    "activation": "activation/06_make_fsf_files_3rdlevel.sh then activation/07_run_3rd_level.sh",
                    "fir": (
                        "FIR/07_create_identity_reg_for_3rd_lvl.sh then "
                        "FIR/08_make_fsf_files_3rdlevel.sh then "
                        "FIR/09_run_FIR_3rd_level.sh"
                    ),
                    "gppi": (
                        "gPPI_code/08_create_identity_reg_for_3rd_lvl.sh then "
                        "gPPI_code/09_make_fsf_files_3rdlevel.sh then "
                        "gPPI_code/10_run_PPI_3rd_level.sh"
                    ),
                }.get(pipeline, "3rd-level FEAT"),
                "Refresh 3rd-level FSF / group FEAT after 2nd-level is complete.",
            ),
        ]
    )

    primary_issue = exclusion_reason.strip() or "see trace detail"
    for line in trace_lines:
        stripped = line.strip()
        if stripped.startswith("[") or stripped.startswith("run-"):
            primary_issue = stripped
            break

    recommended = "inspect trace and upstream logs"
    rerun_note = "Review inclusion trace for this subject."
    for pattern, script, note in rules:
        if re.search(pattern, blob, re.IGNORECASE):
            recommended = script
            rerun_note = note
            break

    rerun_scope = f"subject ({scope_for(recommended)}; {rerun_note})"
    return primary_issue, recommended, rerun_scope


def format_subject_block(
    pipeline: str,
    failure: SubjectFailure,
    reason_prefix: str,
) -> list[str]:
    primary, rerun, scope = infer_rerun_step(
        pipeline, failure.exclusion_reason, failure.trace_lines
    )
    lines = [
        f"### {failure.subj_id}",
        f"{reason_prefix}: {failure.exclusion_reason}",
    ]
    if failure.exclusion_path:
        path_key = reason_prefix.replace("_exclusion_reason", "_path")
        lines.append(f"{path_key}: {failure.exclusion_path}")
    lines.extend(
        [
            f"primary_issue: {primary}",
            "detail:",
        ]
    )
    if failure.trace_lines:
        for tl in failure.trace_lines:
            if tl.strip():
                lines.append(f"  {tl.rstrip()}")
    else:
        lines.append("  (no trace available; run with --force-trace or --refresh)")
    lines.extend(
        [
            f"recommended_rerun: {rerun}",
            f"rerun_scope: {scope}",
            "",
        ]
    )
    return lines


def build_excluded_map_activation(
    mod: Any,
    subject_df: Any,
    spec: Any,
    section: Any | None,
    inclusion_text: str,
    force_trace: bool,
    ctx: dict[str, Any],
) -> dict[str, SubjectFailure]:
    candidates = mod.subjects_for_filter(subject_df, spec.filter_mode)
    excluded_map: dict[str, SubjectFailure] = {}
    section_text = extract_analysis_section(inclusion_text, spec.label)
    trace_blocks = parse_trace_blocks(section_text)
    included_codes: set[str] = set()

    if section:
        included_codes = {s.replace("sub-", "", 1) for s, _ in section.included}
        for subj, reason, path in section.excluded:
            code = subj.replace("sub-", "", 1)
            excluded_map[code] = SubjectFailure(
                subj_id=f"sub-{code}",
                exclusion_reason=reason,
                exclusion_path=path,
                trace_lines=trace_blocks.get(f"sub-{code}", []),
            )
        for code in list(excluded_map):
            if code in included_codes:
                del excluded_map[code]
    for code in candidates:
        if code not in included_codes and code not in excluded_map:
            excluded_map[code] = SubjectFailure(
                subj_id=f"sub-{code}",
                exclusion_reason="not listed in 05 inclusion report (filtered out or not processed)",
                exclusion_path="",
                trace_lines=trace_blocks.get(f"sub-{code}", []),
            )

    if force_trace:
        for code, fail in excluded_map.items():
            fail.trace_lines = mod.trace_excluded_subject(
                ctx["feat_root"],
                ctx["bids_root"],
                ctx["activation_root"],
                code,
                spec,
                fail.exclusion_reason,
                fail.exclusion_path,
                ctx.get("runs_report"),
            )
    elif any(not f.trace_lines for f in excluded_map.values()):
        for code, fail in excluded_map.items():
            if not fail.trace_lines:
                fail.trace_lines = mod.trace_excluded_subject(
                    ctx["feat_root"],
                    ctx["bids_root"],
                    ctx["activation_root"],
                    code,
                    spec,
                    fail.exclusion_reason,
                    fail.exclusion_path,
                    ctx.get("runs_report"),
                )
    return excluded_map


def build_excluded_map_fir_gppi(
    pipeline: str,
    parse_sections: dict,
    subject_df: Any,
    candidates: list[str],
    analysis: str,
    cope_tag: str,
    cope: int,
    inclusion_text: str,
    force_trace: bool,
    trace_fn: Any,
    reason_prefix: str,
) -> dict[str, SubjectFailure]:
    section = parse_sections.get((analysis, cope_tag))
    section_text = extract_analysis_section(inclusion_text, analysis, cope_tag)
    trace_blocks = parse_trace_blocks(section_text)
    excluded_map: dict[str, SubjectFailure] = {}

    included_codes: set[str] = set()
    if section:
        included_codes = {s.replace("sub-", "", 1) for s, _ in section.included}
        for subj, reason, path in section.excluded:
            code = subj.replace("sub-", "", 1)
            excluded_map[code] = SubjectFailure(
                subj_id=f"sub-{code}",
                exclusion_reason=reason,
                exclusion_path=path,
                trace_lines=trace_blocks.get(f"sub-{code}", []),
            )
        for code in list(excluded_map):
            if code in included_codes:
                del excluded_map[code]

    for code in candidates:
        if code not in included_codes and code not in excluded_map:
            excluded_map[code] = SubjectFailure(
                subj_id=f"sub-{code}",
                exclusion_reason=f"not listed in inclusion report (filtered out or not processed)",
                exclusion_path="",
                trace_lines=trace_blocks.get(f"sub-{code}", []),
            )

    need_trace = force_trace or any(not f.trace_lines for f in excluded_map.values())
    if need_trace:
        for code, fail in excluded_map.items():
            if force_trace or not fail.trace_lines:
                fail.trace_lines = trace_fn(code, cope, fail.exclusion_reason, fail.exclusion_path)
    return excluded_map


def collect_activation(
    mods: dict[str, Any],
    paths: dict[str, Path],
    subject_df: Any,
    force_trace: bool,
    analysis_filter: set[str] | None = None,
) -> list[AnalysisFailures]:
    mod = mods["activation"]
    inclusion_path = paths["activation_inclusion_report"]
    inclusion_text = inclusion_path.read_text(encoding="utf-8") if inclusion_path.is_file() else ""
    sections = mod.parse_inclusion_report(inclusion_path)
    analyses = mod.parse_analyses(
        [ln for ln in ACTIVATION_ANALYSES_CSV.splitlines() if ln.strip()]
    )
    ctx = {
        "feat_root": paths["feat_root"],
        "bids_root": paths["bids_root"],
        "activation_root": paths["activation_root"],
        "runs_report": paths["activation_runs_report"]
        if paths["activation_runs_report"].is_file()
        else None,
    }
    out: list[AnalysisFailures] = []
    for spec in analyses:
        if not analysis_selected(spec.label, analysis_filter):
            continue
        section = sections.get(spec.label)
        excluded_map = build_excluded_map_activation(
            mod,
            subject_df,
            spec,
            section,
            inclusion_text,
            force_trace,
            ctx,
        )
        failures = [
            excluded_map[c] for c in sorted(excluded_map) if excluded_map[c].exclusion_reason
        ]
        out.append(
            AnalysisFailures(
                pipeline="activation",
                analysis=spec.label,
                cope_tag=f"cope{spec.cope}",
                failures=failures,
            )
        )
    return out


def collect_fir(
    mods: dict[str, Any],
    paths: dict[str, Path],
    subject_df: Any,
    force_trace: bool,
    analysis_filter: set[str] | None = None,
) -> list[tuple[str, list[SubjectFailure]]]:
    if not analysis_selected("FIR", analysis_filter):
        return []
    mod = mods["fir"]
    roi_mod = mods["fir_roi"]
    inclusion_path = paths["fir_inclusion_report"]
    inclusion_text = inclusion_path.read_text(encoding="utf-8") if inclusion_path.is_file() else ""
    sections = mod.parse_inclusion_report(inclusion_path)
    candidates = roi_mod.subjects_for_filter(subject_df, FIR_FILTER_COL)
    feat_root = paths["feat_root"]
    bids_root = paths["bids_root"]
    fir_root = paths["fir_root"]

    cope_sections: list[tuple[str, list[SubjectFailure]]] = []

    for variant in FIR_VARIANTS:
        for cope in FIR_COPES:
            cope_tag = f"cope{cope}"

            def trace_fn(
                code: str, c: int, reason: str, path: str, v: str = variant
            ) -> list[str]:
                return mod.trace_excluded_subject(
                    feat_root, bids_root, fir_root, code, v, c, reason, path
                )

            excluded_map = build_excluded_map_fir_gppi(
                "fir",
                sections,
                subject_df,
                candidates,
                variant,
                cope_tag,
                cope,
                inclusion_text,
                force_trace,
                lambda code, c, reason, path: trace_fn(code, c, reason, path),
                "08_make_exclusion_reason",
            )
            failures = [excluded_map[c] for c in sorted(excluded_map)]
            if failures:
                cope_sections.append((cope_tag, failures))

    return cope_sections


def collect_gppi(
    mods: dict[str, Any],
    paths: dict[str, Path],
    subject_df: Any,
    force_trace: bool,
    analysis_filter: set[str] | None = None,
) -> list[tuple[AnalysisFailures, list[tuple[str, list[SubjectFailure]]]]]:
    mod = mods["gppi"]
    inclusion_path = paths["gppi_inclusion_report"]
    inclusion_text = inclusion_path.read_text(encoding="utf-8") if inclusion_path.is_file() else ""
    sections = mod.parse_inclusion_report(inclusion_path)
    candidates = mod.subjects_for_filter(subject_df)
    feat_root = paths["feat_root"]
    bids_root = paths["bids_root"]
    seeds_root = paths["seeds_root"]
    timecourse_root = paths["timecourse_root"]
    gppi_root = paths["gppi_root"]
    runs_report = (
        paths["gppi_runs_report"] if paths["gppi_runs_report"].is_file() else None
    )

    gppi_analyses = mods["gppi_3rd"].ANALYSES
    gppi_analysis_cope = mods["gppi_3rd"].ANALYSIS_COPE

    results: list[tuple[AnalysisFailures, list[tuple[str, list[SubjectFailure]]]]] = []
    for analysis in gppi_analyses:
        if not analysis_selected(analysis, analysis_filter):
            continue
        cope = gppi_analysis_cope[analysis]
        cope_sections: list[tuple[str, list[SubjectFailure]]] = []
        cope_tag = f"cope{cope}"

        def trace_fn(
            code: str,
            c: int,
            reason: str,
            path: str,
            ana: str = analysis,
        ) -> list[str]:
            sub_id = f"sub-{code}"
            lines = [
                f"### {sub_id}",
                f"09_make_exclusion_reason: {reason}",
                f"09_make_path: {path}",
                "",
            ]
            lines.extend(
                mod.trace_second_level(
                    feat_root,
                    bids_root,
                    seeds_root,
                    timecourse_root,
                    gppi_root,
                    code,
                    ana,
                    c,
                    runs_report,
                )
            )
            return lines

        excluded_map = build_excluded_map_fir_gppi(
            "gppi",
            sections,
            subject_df,
            candidates,
            analysis,
            cope_tag,
            cope,
            inclusion_text,
            force_trace,
            lambda code, c, reason, path: trace_fn(code, c, reason, path),
            "09_make_exclusion_reason",
        )
        failures = [excluded_map[c] for c in sorted(excluded_map)]
        if failures:
            cope_sections.append((cope_tag, failures))
        results.append(
            (
                AnalysisFailures(pipeline="gppi", analysis=analysis, cope_tag=None, failures=[]),
                cope_sections,
            )
        )
    return results


def write_activation_report(
    report_dir: Path, af: AnalysisFailures, stamp: str
) -> Path | None:
    if not af.failures:
        return None
    path = report_dir / f"activation_{af.analysis}_failures.txt"
    lines = [
        f"# Activation 3rd-level failures: {af.analysis}",
        f"# cope={af.cope_tag}",
        f"# date={stamp}",
        f"# n_failed={len(af.failures)}",
        "",
    ]
    for fail in af.failures:
        lines.extend(format_subject_block("activation", fail, "05_exclusion_reason"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_fir_report(
    report_dir: Path,
    variant: str,
    cope_sections: list[tuple[str, list[SubjectFailure]]],
    stamp: str,
) -> Path | None:
    if not cope_sections:
        return None
    path = report_dir / f"fir_{variant}_failures.txt"
    n_total = sum(len(f) for _, f in cope_sections)
    lines = [
        f"# FIR 3rd-level failures: {variant}",
        f"# date={stamp}",
        f"# n_failed_total={n_total}",
        f"# cope_sections_with_failures={len(cope_sections)}",
        "",
    ]
    for cope_tag, failures in cope_sections:
        lines.extend(
            [
                f"## cope={cope_tag}",
                f"n_failed={len(failures)}",
                "",
            ]
        )
        for fail in failures:
            lines.extend(format_subject_block("fir", fail, "08_make_exclusion_reason"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_gppi_report(
    report_dir: Path,
    analysis: str,
    cope_sections: list[tuple[str, list[SubjectFailure]]],
    stamp: str,
) -> Path | None:
    if not cope_sections:
        return None
    path = report_dir / f"gppi_{analysis}_failures.txt"
    n_total = sum(len(f) for _, f in cope_sections)
    lines = [
        f"# gPPI 3rd-level failures: {analysis}",
        f"# date={stamp}",
        f"# n_failed_total={n_total}",
        f"# cope_sections_with_failures={len(cope_sections)}",
        "",
    ]
    for cope_tag, failures in cope_sections:
        lines.extend(
            [
                f"## cope={cope_tag}",
                f"n_failed={len(failures)}",
                "",
            ]
        )
        for fail in failures:
            lines.extend(format_subject_block("gppi", fail, "09_make_exclusion_reason"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_index(report_dir: Path, rows: list[IndexRow], stamp: str) -> Path:
    path = report_dir / "index.txt"
    lines = [
        "# Final 3rd-level failure report index",
        f"# date={stamp}",
        "# pipeline\tanalysis\tcope\tn_failed\tstatus\treport_path",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row.pipeline}\t{row.analysis}\t{row.cope_tag}\t{row.n_failed}\t"
            f"{row.status}\t{row.report_path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate final 3rd-level failure reports")
    p.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT", DEFAULT_BIDS_ROOT))
    p.add_argument("--code-root", default=os.environ.get("CODE_ROOT", DEFAULT_CODE_ROOT))
    p.add_argument(
        "--report-dir",
        default=os.environ.get("FINAL_REPORT_DIR", f"{DEFAULT_CODE_ROOT}/final_report"),
    )
    p.add_argument(
        "--subject-csv",
        default=os.environ.get(
            "SUBJECT_CSV", f"{DEFAULT_CODE_ROOT}/ses-v2-analysis-subs-fmri.csv"
        ),
    )
    p.add_argument(
        "--pipelines",
        default=os.environ.get("PIPELINES", "activation,fir,gppi"),
        help="Comma-separated: activation,fir,gppi",
    )
    p.add_argument(
        "--analysis",
        default=os.environ.get("ANALYSES", ""),
        metavar="NAME",
        help=(
            "Comma-separated analysis name(s) to check (limits work within --pipelines). "
            "Valid: seen_face_vs_unseen_face, seen_object_vs_unseen_object, "
            "FIR, PPI_LOC, PPI_FFA"
        ),
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Run make_*_inclusion_report.sh before parsing",
    )
    p.add_argument(
        "--force-trace",
        action="store_true",
        help="Always re-run live filesystem trace (ignore cached trace blocks)",
    )
    args = p.parse_args()

    try:
        analysis_filter = parse_analysis_filter(args.analysis)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    code_root = Path(args.code_root)
    bids_root = Path(args.bids_root)
    report_dir = Path(args.report_dir)
    subject_csv = Path(args.subject_csv)
    pipelines = {x.strip().lower() for x in args.pipelines.split(",") if x.strip()}

    implied = pipelines_for_analyses(analysis_filter)
    if implied is not None:
        pipelines &= implied
        if not pipelines:
            print(
                "ERROR: no pipeline left after applying --analysis with --pipelines",
                file=sys.stderr,
            )
            return 1

    paths = {
        "bids_root": bids_root,
        "feat_root": bids_root / "derivatives/fslFeat",
        "activation_root": code_root / "activation",
        "fir_root": code_root / "FIR",
        "gppi_root": code_root / "gPPI_code",
        "seeds_root": bids_root / "derivatives/gppi_seeds",
        "timecourse_root": Path(
            os.environ.get("TIMECOURSE_ROOT", DEFAULT_TIMECOURSE_ROOT)
        ),
        "activation_inclusion_report": code_root
        / "activation"
        / "fsf_files"
        / "group"
        / "included_subjects_report.txt",
        "activation_runs_report": code_root
        / "activation"
        / "job_files"
        / "run_activation_2nd_level"
        / "included_runs_report.txt",
        "fir_inclusion_report": code_root
        / "FIR"
        / "fsf_files"
        / "group"
        / "included_subjects_report.txt",
        "gppi_inclusion_report": code_root
        / "gPPI_code"
        / "fsf_files"
        / "group"
        / "included_subjects_report.txt",
        "gppi_runs_report": code_root
        / "gPPI_code"
        / "job_files"
        / "run_PPI_2nd_level"
        / "included_runs_report.txt",
    }

    if args.refresh:
        run_refresh(code_root, pipelines)

    if not subject_csv.is_file():
        print(f"ERROR: subject CSV not found: {subject_csv}", file=sys.stderr)
        return 1

    try:
        mods = load_pipeline_modules(code_root)
    except Exception as exc:
        print(f"ERROR: failed to load pipeline modules: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    index_rows: list[IndexRow] = []

    if "activation" in pipelines:
        try:
            subject_df = mods["activation"].load_subject_csv(subject_csv)
            activation_failures = collect_activation(
                mods, paths, subject_df, args.force_trace
            )
            for af in activation_failures:
                n_failed = len(af.failures)
                out_path = write_activation_report(report_dir, af, stamp)
                rel = str(out_path.relative_to(report_dir)) if out_path else ""
                index_rows.append(
                    IndexRow(
                        pipeline="activation",
                        analysis=af.analysis,
                        cope_tag=af.cope_tag or "",
                        n_failed=n_failed,
                        report_path=rel,
                        status="failures" if n_failed else "ok",
                    )
                )
                if out_path:
                    print(f">> wrote {out_path}")
        except Exception as exc:
            print(f"WARN: activation pipeline failed: {exc}", file=sys.stderr)

    if "fir" in pipelines:
        try:
            subject_df = mods["fir_roi"].load_subject_csv(subject_csv)
            fir_cope_sections = collect_fir(
                mods, paths, subject_df, args.force_trace, analysis_filter
            )
            n_total = sum(len(f) for _, f in fir_cope_sections)
            out_path = write_fir_report(report_dir, "FIR", fir_cope_sections, stamp)
            rel = str(out_path.relative_to(report_dir)) if out_path else ""
            index_rows.append(
                IndexRow(
                    pipeline="fir",
                    analysis="FIR",
                    cope_tag="all",
                    n_failed=n_total,
                    report_path=rel,
                    status="failures" if n_total else "ok",
                )
            )
            if out_path:
                print(f">> wrote {out_path}")
        except Exception as exc:
            print(f"WARN: FIR pipeline failed: {exc}", file=sys.stderr)

    if "gppi" in pipelines:
        try:
            subject_df = mods["gppi"].load_subject_csv(subject_csv)
            gppi_results = collect_gppi(
                mods, paths, subject_df, args.force_trace, analysis_filter
            )
            for _af, cope_sections in gppi_results:
                analysis = _af.analysis
                n_total = sum(len(f) for _, f in cope_sections)
                out_path = write_gppi_report(report_dir, analysis, cope_sections, stamp)
                rel = str(out_path.relative_to(report_dir)) if out_path else ""
                index_rows.append(
                    IndexRow(
                        pipeline="gppi",
                        analysis=analysis,
                        cope_tag="all",
                        n_failed=n_total,
                        report_path=rel,
                        status="failures" if n_total else "ok",
                    )
                )
                if out_path:
                    print(f">> wrote {out_path}")
        except Exception as exc:
            print(f"WARN: gPPI pipeline failed: {exc}", file=sys.stderr)

    index_path = write_index(report_dir, index_rows, stamp)
    print(f">> wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
