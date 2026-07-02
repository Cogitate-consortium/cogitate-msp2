#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Build 3rd-level gPPI FSFs (inputtype 1: per-subject copeN.feat dirs) from completed
# 2nd-level .gfeat dirs (SYNCHRONY_min_seen). PPI_FFA -> cope3.feat, PPI_LOC ->
# cope4.feat. Only subjects with successful 2nd-level FEAT and the target copeN.feat
# are included. Writes per-cope input lists and one combined inclusion report.
#
# Run after 07_run_PPI_2nd_level_VG.sh and 08_create_identity_reg_for_3rd_lvl.sh;
# before 10_run_PPI_3rd_level.sh.
#
# Usage: bash 09_make_fsf_files_3rdlevel.sh

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_TEMPLATE_DIR="${GPPI_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files/group"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
TEMPLATE_FSF="${FSF_TEMPLATE_DIR}/NXX_ses-V2_task-VG_analysis-3rdGLM_ANALYSISXX-MNI152NLin2009cAsym_desc-copeXX.fsf"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"

# shellcheck source=gppi_3rd_level.sh
source "${GPPI_CODE_ROOT}/gppi_3rd_level.sh"
ANALYSES=("${GPPI_3RD_ANALYSES[@]}")

for _analysis in PPI_FFA PPI_LOC; do
    _cope="$(gppi_3rd_cope_for "$_analysis")" || exit 1
    case "$_analysis" in
        PPI_FFA) [ "$_cope" -eq 3 ] || { echo "ERROR: PPI_FFA must use cope 3, got ${_cope}" >&2; exit 1; } ;;
        PPI_LOC) [ "$_cope" -eq 4 ] || { echo "ERROR: PPI_LOC must use cope 4, got ${_cope}" >&2; exit 1; } ;;
    esac
done
echo ">> 2nd-level cope mapping: PPI_FFA=cope3, PPI_LOC=cope4"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$TEMPLATE_FSF" ]; then
    echo "ERROR: missing template: $TEMPLATE_FSF" >&2
    exit 1
fi

mkdir -p "$FSF_FILES_ROOT"

n_removed_fsf=0
shopt -s nullglob
for old_fsf in "${FSF_FILES_ROOT}"/N*_ses-V2_task-VG_analysis-3rdGLM_*_desc-cope*.fsf; do
    rm -f "$old_fsf"
    n_removed_fsf=$((n_removed_fsf + 1))
done
shopt -u nullglob
if [ "$n_removed_fsf" -gt 0 ]; then
    echo ">> removed ${n_removed_fsf} previous 3rd-level FSF(s) from ${FSF_FILES_ROOT}"
fi

{
    echo "# gPPI 3rd-level — 2nd-level cope inclusion report (all analyses/copes)"
    echo "# date=$(date -Iseconds)"
    echo "# filter=SYNCHRONY_min_seen"
    echo ""
} > "$INCLUSION_REPORT"

ANALYSES_CSV="$(printf '%s\n' "${ANALYSES[@]}")"
SUBJECT_CSV="$SUBJECT_CSV" \
ANALYSES_CSV="$ANALYSES_CSV" \
FSLFEAT_ROOT="$FSLFEAT_ROOT" \
FSF_FILES_ROOT="$FSF_FILES_ROOT" \
TEMPLATE_FSF="$TEMPLATE_FSF" \
INCLUSION_REPORT="$INCLUSION_REPORT" \
GPPI_CODE_ROOT="$GPPI_CODE_ROOT" \
python3 <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd

subject_csv = os.environ["SUBJECT_CSV"]
feat_root = Path(os.environ["FSLFEAT_ROOT"])
fsf_out_root = Path(os.environ["FSF_FILES_ROOT"])
template_fsf = Path(os.environ["TEMPLATE_FSF"])
report_path = Path(os.environ["INCLUSION_REPORT"])

sys.path.insert(0, os.environ["GPPI_CODE_ROOT"])
from gppi_3rd_level import ANALYSIS_COPE

# PPI_FFA -> cope3 (ppi_FFA), PPI_LOC -> cope4 (ppi_LOC); must match gppi_3rd_level.sh
REQUIRED_COPE = {"PPI_FFA": 3, "PPI_LOC": 4}
if ANALYSIS_COPE != REQUIRED_COPE:
    print(
        f"ERROR: ANALYSIS_COPE must be {REQUIRED_COPE}, got {ANALYSIS_COPE}",
        file=sys.stderr,
    )
    sys.exit(1)

analyses = [line for line in os.environ["ANALYSES_CSV"].splitlines() if line.strip()]
for analysis in analyses:
    if analysis not in REQUIRED_COPE:
        print(f"ERROR: unknown analysis {analysis!r}", file=sys.stderr)
        sys.exit(1)

subj_df = pd.read_csv(subject_csv, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(subject_csv, sep=";")

col = "SYNCHRONY_min_seen"
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {subject_csv}", file=sys.stderr)
    sys.exit(1)

subjects = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE"), "sub_code"].tolist()
if not subjects:
    print(f"ERROR: no subjects after filtering {subject_csv}", file=sys.stderr)
    sys.exit(1)

template_text = template_fsf.read_text(encoding="utf-8")


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def find_second_level_cope_feat(sub_code: str, analysis: str, cope: int):
    subj_id = f"sub-{sub_code}"
    parent_stem = f"{subj_id}_ses-V2_task-VG_analysis-2ndGLM_{analysis}-MNI152NLin2009cAsym"
    excluded = []
    for ext in (".gfeat", ".feat"):
        parent = feat_root / "gPPI" / subj_id / f"{parent_stem}{ext}"
        cope_feat = parent / f"cope{cope}.feat"
        if not parent.is_dir():
            excluded.append((subj_id, f"missing 2nd-level dir ({ext})", str(parent)))
            continue
        if not report_ok(parent):
            excluded.append((subj_id, f"bad/missing report.html ({ext})", str(parent)))
            continue
        if not cope_feat.is_dir():
            excluded.append((subj_id, f"missing cope{cope}.feat ({ext})", str(cope_feat)))
            continue
        return str(cope_feat), str(parent), excluded
    return None, None, excluded


def format_feat_file_line(idx: int, path: str) -> str:
    # Tcl braces: paths like ..._ses-V2_... must not expand $ses in double quotes.
    return f"set feat_files({idx}) {{{path}}}\n\n"


def build_blocks(n_subjects: int):
    feat_lines = []
    evg_lines = []
    groupmem_lines = []
    for idx in range(1, n_subjects + 1):
        feat_lines.append(f"# 4D AVW data or FEAT directory ({idx})")
        evg_lines.append(f"# Higher-level EV value for EV 1 and input {idx}")
        evg_lines.append(f"set fmri(evg{idx}.1) 1")
        evg_lines.append("")
        groupmem_lines.append(f"# Group membership for input {idx}")
        groupmem_lines.append(f"set fmri(groupmem.{idx}) 1")
        groupmem_lines.append("")
    return "\n".join(feat_lines), "\n".join(evg_lines), "\n".join(groupmem_lines)


n_written = 0
n_skipped = 0

for analysis in analyses:
    cope = ANALYSIS_COPE[analysis]
    included_paths = []
    excluded_rows = []
    for sub_code in subjects:
        cope_path, gfeat_path, exc = find_second_level_cope_feat(sub_code, analysis, cope)
        if cope_path:
            included_paths.append((sub_code, cope_path, gfeat_path))
        else:
            excluded_rows.extend(exc)

    cope_tag = f"cope{cope}"
    section = [
        f"## analysis={analysis}\tcope={cope_tag}\tstatus=generated",
        f"n_included={len(included_paths)}",
        f"n_excluded={len(subjects) - len(included_paths)}",
        "",
        "# Included subjects",
    ]
    for sub_code, cope_path, gfeat_path in included_paths:
        section.append(f"sub-{sub_code}\t{cope_path}\t{gfeat_path}")
    section.extend(["", "# Excluded subjects"])
    for subj_id, reason, path in excluded_rows:
        section.append(f"{subj_id}\t{reason}\t{path}")
    section.append("")
    with report_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(section) + "\n")

    inputs_file = fsf_out_root / f"inputs_{analysis}_{cope_tag}.txt"
    if not included_paths:
        print(f">> {analysis} {cope_tag}: skip (no included subjects)", file=sys.stderr)
        inputs_file.write_text("", encoding="utf-8")
        n_skipped += 1
        continue

    feat_block_header, evg_block, groupmem_block = build_blocks(len(included_paths))
    feat_block = feat_block_header + "\n"
    for idx, (_sub_code, cope_path, _gfeat_path) in enumerate(included_paths, start=1):
        feat_block += format_feat_file_line(idx, cope_path)

    n_subjects = len(included_paths)
    feat_suffix = (
        f"ses-V2_task-VG_analysis-3rdGLM_{analysis}-MNI152NLin2009cAsym_desc-{cope_tag}"
    )
    fsf_out = fsf_out_root / f"N{n_subjects}_{feat_suffix}.fsf"

    text = template_text
    text = text.replace("ANALYSISXX", analysis)
    text = text.replace("copeXX", cope_tag)
    text = text.replace("XXX_NPTS", str(n_subjects))
    text = text.replace("NXX", f"N{n_subjects}")
    text = text.replace("### FEAT_FILES_BLOCK ###", feat_block.rstrip())
    text = text.replace("### EVG_BLOCK ###", evg_block.rstrip())
    text = text.replace("### GROUPMEM_BLOCK ###", groupmem_block.rstrip())
    fsf_out.write_text(text, encoding="utf-8")

    inputs_file.write_text(
        "\n".join(cope_path for _, cope_path, _ in included_paths) + "\n", encoding="utf-8"
    )
    print(f">> {analysis} {cope_tag}: fsf={fsf_out.name} included={n_subjects}")
    n_written += 1

print(f">> wrote {n_written} FSF(s); skipped {n_skipped} cope(s) with no inputs")
print(f">> inclusion report: {report_path}")
PY
