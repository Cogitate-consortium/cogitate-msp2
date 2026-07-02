#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Build 3rd-level FIR FSFs (inputtype 1: per-subject copeN.feat dirs) from completed
# 2nd-level .gfeat dirs (BASELINE_min_seen_unseen). Loops FIR × copes 1–42.
#
# Run after 06_run_FIR_2nd_level_VG.sh and 07_create_identity_reg_for_3rd_lvl.sh;
# before 09_run_FIR_3rd_level.sh.
#
# Usage: bash 08_make_fsf_files_3rdlevel.sh

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_TEMPLATE_DIR="${FIR_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files/group"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
TEMPLATE_FSF="${FSF_TEMPLATE_DIR}/NXX_ses-V2_task-VG_analysis-3rdGLM_ANALYSISXX-MNI152NLin2009cAsym_desc-copeXX.fsf"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

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
    echo "# FIR 3rd-level — 2nd-level cope inclusion report (all variants/copes)"
    echo "# date=$(date -Iseconds)"
    echo "# filter=${FIR_SUBJECT_FILTER_COL}"
    echo ""
} >"$INCLUSION_REPORT"

FIR_VARIANTS_CSV="$(IFS=,; echo "${FIR_ALL_VARIANTS[*]}")"
FIR_3RD_COPES_CSV="$(IFS=,; echo "${FIR_3RD_COPES[*]}")"

SUBJECT_CSV="$SUBJECT_CSV" \
FIR_SUBJECT_FILTER_COL="$FIR_SUBJECT_FILTER_COL" \
FIR_VARIANTS_CSV="$FIR_VARIANTS_CSV" \
FIR_3RD_COPES_CSV="$FIR_3RD_COPES_CSV" \
FSLFEAT_ROOT="$FSLFEAT_ROOT" \
FSF_FILES_ROOT="$FSF_FILES_ROOT" \
TEMPLATE_FSF="$TEMPLATE_FSF" \
INCLUSION_REPORT="$INCLUSION_REPORT" \
python3 <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd

subject_csv = os.environ["SUBJECT_CSV"]
filter_col = os.environ["FIR_SUBJECT_FILTER_COL"]
variants = [v.strip() for v in os.environ["FIR_VARIANTS_CSV"].split(",") if v.strip()]
copes = [int(c) for c in os.environ["FIR_3RD_COPES_CSV"].split(",") if c.strip()]
feat_root = Path(os.environ["FSLFEAT_ROOT"])
fsf_out_root = Path(os.environ["FSF_FILES_ROOT"])
template_fsf = Path(os.environ["TEMPLATE_FSF"])
report_path = Path(os.environ["INCLUSION_REPORT"])

subj_df = pd.read_csv(subject_csv, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(subject_csv, sep=";")

if filter_col not in subj_df.columns:
    print(f"ERROR: missing column {filter_col} in {subject_csv}", file=sys.stderr)
    sys.exit(1)

subjects = subj_df.loc[subj_df[filter_col].astype(str).str.upper().eq("TRUE"), "sub_code"].tolist()
if not subjects:
    print(f"ERROR: no subjects after filtering {subject_csv} ({filter_col})", file=sys.stderr)
    sys.exit(1)

template_text = template_fsf.read_text(encoding="utf-8")


def report_ok(feat_dir: Path) -> bool:
    report = feat_dir / "report.html"
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "Error" not in text and "ERROR" not in text


def find_second_level_cope_feat(sub_code: str, variant: str, cope: int):
    subj_id = f"sub-{sub_code}"
    parent_stem = f"{subj_id}_ses-V2_task-VG_analysis-2ndGLM_{variant}-MNI152NLin2009cAsym"
    excluded = []
    for ext in (".gfeat", ".feat"):
        parent = feat_root / variant / subj_id / f"{parent_stem}{ext}"
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

for variant in variants:
    for cope in copes:
        included_paths = []
        excluded_rows = []
        for sub_code in subjects:
            cope_path, gfeat_path, exc = find_second_level_cope_feat(sub_code, variant, cope)
            if cope_path:
                included_paths.append((sub_code, cope_path, gfeat_path))
            else:
                excluded_rows.extend(exc)

        cope_tag = f"cope{cope}"
        section = [
            f"## analysis={variant}\tcope={cope_tag}\tstatus=generated",
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

        inputs_file = fsf_out_root / f"inputs_{variant}_{cope_tag}.txt"
        third_suffix = f"analysis-3rdGLM_{variant}-MNI152NLin2009cAsym"

        if not included_paths:
            print(f">> {variant} {cope_tag}: skip (no included subjects)", file=sys.stderr)
            inputs_file.write_text("", encoding="utf-8")
            n_skipped += 1
            continue

        feat_block_header, evg_block, groupmem_block = build_blocks(len(included_paths))
        feat_block = feat_block_header + "\n"
        for idx, (_sub_code, cope_path, _gfeat_path) in enumerate(included_paths, start=1):
            feat_block += format_feat_file_line(idx, cope_path)

        n_subjects = len(included_paths)
        fsf_out = fsf_out_root / (
            f"N{n_subjects}_ses-V2_task-VG_{third_suffix}_desc-{cope_tag}.fsf"
        )

        text = template_text
        text = text.replace("ANALYSISXX", variant)
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
        print(f">> {variant} {cope_tag}: fsf={fsf_out.name} included={n_subjects}")
        n_written += 1

print(f">> wrote {n_written} FSF(s); skipped {n_skipped} cope(s) with no inputs")
print(f">> inclusion report: {report_path}")
PY
