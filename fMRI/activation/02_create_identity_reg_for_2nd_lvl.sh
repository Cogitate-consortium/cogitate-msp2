#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# After 1st-level GLM FEAT with registration off (fMRIPrep MNI BOLD), create dummy reg/
# in each run-level .feat so 2nd-level FEAT can merge inputs. Run before 03_make_fsf_files.sh.
#
# Usage: bash 02_create_identity_reg_for_2nd_lvl.sh
#
# Requires FSL in PATH. Shared fallbacks/QC: glm/identity_reg_for_2nd_lvl.sh

module load FSL

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
ACTIVATION_CODE_ROOT="${CODE_PATH}/activation"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
JOB_FILES_ROOT="${ACTIVATION_CODE_ROOT}/job_files/create_identity_reg_for_2nd_lvl"
IDENTITY_REG_LIB="${CODE_PATH}/glm/identity_reg_for_2nd_lvl.sh"
FEAT_SUFFIX="analysis-1stGLM_space-MNI152NLin2009cAsym"

# shellcheck source=identity_reg_for_2nd_lvl.sh
source "${IDENTITY_REG_LIB}"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "${FSLDIR}/etc/flirtsch/ident.mat" ]; then
    echo "ERROR: identity matrix not found (is FSL loaded?): ${FSLDIR}/etc/flirtsch/ident.mat" >&2
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

required = ["ACTIVATION_min_sf_uf", "ACTIVATION_min_so_uo"]
missing = [c for c in required if c not in subj_df.columns]
if missing:
    print(f"ERROR: missing columns {missing} in {path}", file=sys.stderr)
    sys.exit(1)

flag_sf = subj_df["ACTIVATION_min_sf_uf"].astype(str).str.upper().eq("TRUE")
flag_so = subj_df["ACTIVATION_min_so_uo"].astype(str).str.upper().eq("TRUE")
subj_df = subj_df.loc[flag_sf | flag_so]

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
    echo "ERROR: no subjects after loading/filtering $SUBJECT_CSV (ACTIVATION_min_sf_uf | ACTIVATION_min_so_uo)" >&2
    exit 1
fi

echo ">> Number of subjects to use: ${#SUBJECTS[@]} (ACTIVATION_min_sf_uf | ACTIVATION_min_so_uo == TRUE; list: $SUBJECT_CSV)"

declare -a RUNS=(1 2 3 4 5 6 7 8)
n_total_failed=0

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    n_created=0
    n_skipped=0
    n_missing_feat=0
    n_failed=0

    for i_run in "${RUNS[@]}"; do
        feat_path="${FSLFEAT_ROOT}/activation/${subj_id}/${subj_id}_ses-V2_task-VG_run-${i_run}_${FEAT_SUFFIX}.feat"
        marker="${job_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_${FEAT_SUFFIX}.identity_reg.run"

        if [ ! -d "$feat_path" ]; then
            n_missing_feat=$((n_missing_feat + 1))
            continue
        fi

        if [ -f "$marker" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi

        if run_identity_reg_and_qc "$feat_path" "$marker"; then
            n_created=$((n_created + 1))
        else
            n_failed=$((n_failed + 1))
        fi
    done

    n_total_failed=$((n_total_failed + n_failed))
    echo ">> ${subj_id}: created=${n_created} skipped=${n_skipped} missing_feat=${n_missing_feat} failed=${n_failed}"
done

if [ "$n_total_failed" -gt 0 ]; then
    echo "ERROR: identity reg creation failed for ${n_total_failed} run(s)" >&2
    exit 1
fi
