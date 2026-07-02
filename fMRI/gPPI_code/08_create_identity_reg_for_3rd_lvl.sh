#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# After 2nd-level gPPI FEAT (.gfeat), create dummy reg/ at each subject's 2nd-level
# .gfeat root for 3rd-level FEAT (inputtype 1 + copeN.feat inputs). PPI_FFA uses
# cope 3; PPI_LOC uses cope 4.
#
# Run after 07_run_PPI_2nd_level_VG.sh; before 09_make_fsf_files_3rdlevel.sh.
#
# Usage: bash 08_create_identity_reg_for_3rd_lvl.sh
#
# Cluster rerun (gPPI 3rd level): 08 -> 09_make -> delete group .gfeat (or
# delete_fslfeat_derivatives.py --analysis gPPI --level 3rd --execute) ->
# clear job_files/run_PPI_3rd_level/*/*.run -> 10_run_PPI_3rd_level.sh
#
# Requires FSL in PATH. Shared helpers: glm/identity_reg_for_2nd_lvl.sh

module load FSL

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/create_identity_reg_for_3rd_lvl"
IDENTITY_REG_LIB="${CODE_PATH}/glm/identity_reg_for_2nd_lvl.sh"

# shellcheck source=identity_reg_for_2nd_lvl.sh
source "${IDENTITY_REG_LIB}"

# shellcheck source=gppi_3rd_level.sh
source "${GPPI_CODE_ROOT}/gppi_3rd_level.sh"

declare -a ANALYSES=("${GPPI_3RD_ANALYSES[@]}")

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "${FSLDIR}/etc/flirtsch/ident.mat" ]; then
    echo "ERROR: identity matrix not found (is FSL loaded?): ${FSLDIR}/etc/flirtsch/ident.mat" >&2
    exit 1
fi

report_ok() {
    local feat_dir="$1"
    local report="${feat_dir}/report.html"
    if [ ! -f "$report" ]; then
        return 1
    fi
    if grep -qE 'Error|ERROR' "$report" 2>/dev/null; then
        return 1
    fi
    return 0
}

_sublist_tmp="$(mktemp)"
ANALYSES_CSV="$(printf '%s\n' "${ANALYSES[@]}")"
SUBJECT_CSV="$SUBJECT_CSV" ANALYSES_CSV="$ANALYSES_CSV" GPPI_CODE_ROOT="$GPPI_CODE_ROOT" python3 <<'PY' >"$_sublist_tmp"
import os
import sys

sys.path.insert(0, os.environ["GPPI_CODE_ROOT"])
from gppi_3rd_level import ANALYSES, ANALYSIS_COPE

import pandas as pd

path = os.environ["SUBJECT_CSV"]
analyses = [line for line in os.environ["ANALYSES_CSV"].splitlines() if line.strip()]

subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

col = "SYNCHRONY_min_seen"
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
    sys.exit(1)

subs = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE"), "sub_code"]
for analysis in analyses:
    cope = ANALYSIS_COPE[analysis]
    suffix = f"analysis-2ndGLM_{analysis}-MNI152NLin2009cAsym"
    for sub in subs:
        print(f"{sub}\t{analysis}\t{suffix}\t{cope}")
PY
_py_st=$?
if [ "$_py_st" -ne 0 ]; then
    rm -f "$_sublist_tmp"
    exit "$_py_st"
fi

n_total_failed=0
n_total_created=0
n_total_skipped=0
n_total_missing=0

while IFS=$'\t' read -r sub_code analysis_label gfeat_suffix cope_s || [ -n "${sub_code}" ]; do
    [ -n "$sub_code" ] || continue
    cope_idx="$cope_s"
    subj_id="sub-${sub_code}"
    job_dir="${JOB_FILES_ROOT}/${analysis_label}"
    mkdir -p "$job_dir"

    parent_stem="${subj_id}_ses-V2_task-VG_${gfeat_suffix}"
    gfeat_path=""
    for ext in .gfeat .feat; do
        candidate="${FSLFEAT_ROOT}/gPPI/${subj_id}/${parent_stem}${ext}"
        if [ -d "$candidate" ]; then
            gfeat_path="$candidate"
            break
        fi
    done

    marker="${job_dir}/${parent_stem}_cope${cope_idx}.identity_reg.run"

    if [ -z "$gfeat_path" ]; then
        n_total_missing=$((n_total_missing + 1))
        continue
    fi

    if [ -f "$marker" ]; then
        if spot_check_identity_reg_gfeat "$gfeat_path" "$cope_idx" >/dev/null 2>&1; then
            n_total_skipped=$((n_total_skipped + 1))
            continue
        fi
        echo "WARN: marker present but reg/QC missing; re-creating identity reg for ${gfeat_path}" >&2
    fi

    if ! report_ok "$gfeat_path"; then
        echo "WARN: skip ${gfeat_path} (bad/missing report.html)" >&2
        n_total_missing=$((n_total_missing + 1))
        continue
    fi

    if [ ! -d "${gfeat_path}/cope${cope_idx}.feat" ]; then
        echo "WARN: skip ${gfeat_path} (missing cope${cope_idx}.feat)" >&2
        n_total_missing=$((n_total_missing + 1))
        continue
    fi

    if run_identity_reg_and_qc_gfeat "$gfeat_path" "$cope_idx" "$marker"; then
        n_total_created=$((n_total_created + 1))
        echo ">> ${analysis_label} ${subj_id}: identity reg OK (${gfeat_path})"
    else
        n_total_failed=$((n_total_failed + 1))
        echo "ERROR: identity reg failed for ${gfeat_path} cope${cope_idx}" >&2
    fi
done < "$_sublist_tmp"
rm -f "$_sublist_tmp"

echo ">> summary: created=${n_total_created} skipped_marker=${n_total_skipped} missing_or_skip=${n_total_missing} failed=${n_total_failed}"

if [ "$n_total_failed" -gt 0 ]; then
    echo "ERROR: identity reg creation failed for ${n_total_failed} 2nd-level .gfeat run(s)" >&2
    exit 1
fi
