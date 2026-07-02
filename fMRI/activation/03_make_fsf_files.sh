#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Generate per-subject 2nd-level activation FSFs. Run after 02_create_identity_reg_for_2nd_lvl.sh.
#
# Usage: bash 03_make_fsf_files.sh

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
ACTIVATION_CODE_ROOT="${CODE_PATH}/activation"
FSF_TEMPLATE_DIR="${ACTIVATION_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${ACTIVATION_CODE_ROOT}/fsf_files"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

load_subjects() {
    local filter_mode="$1"
    local _tmp
    _tmp="$(mktemp)"
    SUBJECT_CSV="$SUBJECT_CSV" FILTER_MODE="$filter_mode" python3 <<'PY' >"$_tmp"
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
mode = os.environ["FILTER_MODE"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

required = ["ACTIVATION_min_sf_uf", "ACTIVATION_min_so_uo"]
missing = [c for c in required if c not in subj_df.columns]
if missing:
    print(f"ERROR: missing columns {missing} in {path}", file=sys.stderr)
    sys.exit(1)

flag_sf = subj_df["ACTIVATION_min_sf_uf"].astype(str).str.upper().eq("TRUE")
flag_so = subj_df["ACTIVATION_min_so_uo"].astype(str).str.upper().eq("TRUE")

if mode == "union":
    subj_df = subj_df.loc[flag_sf | flag_so]
elif mode == "both":
    subj_df = subj_df.loc[flag_sf & flag_so]
elif mode in subj_df.columns:
    subj_df = subj_df.loc[subj_df[mode].astype(str).str.upper().eq("TRUE")]
else:
    print(f"ERROR: unknown FILTER_MODE={mode}", file=sys.stderr)
    sys.exit(1)

for sub in subj_df["sub_code"].values:
    print(sub)
PY
    local _st=$?
    if [ "$_st" -ne 0 ]; then
        rm -f "$_tmp"
        return "$_st"
    fi
    SUBJECTS=()
    mapfile -t SUBJECTS < "$_tmp"
    rm -f "$_tmp"
    return 0
}

make_fsf_for_analysis() {
    local analysis_suffix="$1"
    local filter_mode="$2"
    local template_fsf="${FSF_TEMPLATE_DIR}/sub-PXX_ses-V2_task-VG_${analysis_suffix}.fsf"

    if [ ! -f "$template_fsf" ]; then
        echo "ERROR: missing template: $template_fsf" >&2
        return 1
    fi

    if ! load_subjects "$filter_mode"; then
        return 1
    fi

    if [ ${#SUBJECTS[@]} -eq 0 ]; then
        echo "ERROR: no subjects for ${analysis_suffix} (${filter_mode})" >&2
        return 1
    fi

    echo ">> ${analysis_suffix}: subjects=${#SUBJECTS[@]} (filter: ${filter_mode})"

    for subj in "${SUBJECTS[@]}"; do
        fsf_dir="${FSF_FILES_ROOT}/${subj}"
        mkdir -p "$fsf_dir"
        fsf_out="${fsf_dir}/sub-${subj}_ses-V2_task-VG_${analysis_suffix}.fsf"
        sed "s/sub-PXX/sub-${subj}/g" "$template_fsf" > "$fsf_out"
    done
}

make_fsf_for_analysis "analysis-2ndGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym" "ACTIVATION_min_sf_uf" || exit 1
make_fsf_for_analysis "analysis-2ndGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym" "ACTIVATION_min_so_uo" || exit 1
