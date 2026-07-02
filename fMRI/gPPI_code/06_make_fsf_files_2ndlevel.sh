#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
#"""

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_TEMPLATE_DIR="${GPPI_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
LEGACY_FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

_sublist_tmp="$(mktemp)"
SUBJECT_CSV="$SUBJECT_CSV" python3 <<'PY' >"$_sublist_tmp"
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

col = "SYNCHRONY_min_seen"
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
    sys.exit(1)
subj_df = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE")]

for sub in subj_df["sub_code"].values:
    print(sub)
PY
_py_st=$?
if [ "$_py_st" -ne 0 ]; then
    rm -f "$_sublist_tmp"
    exit "$_py_st"
fi
mapfile -t SUBJECTS < "$_sublist_tmp"
rm -f "$_sublist_tmp"

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: no subjects after loading/filtering $SUBJECT_CSV" >&2
    exit 1
fi

echo ">> Number of subjects to use: ${#SUBJECTS[@]} (SYNCHRONY_min_seen == TRUE; list: $SUBJECT_CSV)"

# To test/run just one participant, uncomment and set the line below (use the sub_code, e.g., SUBJECTS=(SC108))
#SUBJECTS=(SC108)

declare -a ANALYSES=(PPI_LOC PPI_FFA)

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fsf_dir="${FSF_FILES_ROOT}/${subj}"
    mkdir -p "$fsf_dir"
    n_written=0

    for analysis in "${ANALYSES[@]}"; do
        template_fsf="${FSF_TEMPLATE_DIR}/sub-PXX_ses-V2_task-VG_analysis-2ndGLM_${analysis}-MNI152NLin2009cAsym.fsf"
        fsf_out="${fsf_dir}/${subj}_ses-V2_task-VG_analysis-2ndGLM_${analysis}-MNI152NLin2009cAsym.fsf"

        if [ ! -f "$template_fsf" ]; then
            echo "ERROR: missing template: $template_fsf" >&2
            exit 1
        fi

        sed -e "s/sub-PXX/${subj_id}/g" \
            -e "s|${LEGACY_FSLFEAT_ROOT}|${FSLFEAT_ROOT}|g" \
            "$template_fsf" > "$fsf_out"
        n_written=$((n_written + 1))
    done

    echo ">> ${subj_id}: fsf_files=${n_written}"
done
