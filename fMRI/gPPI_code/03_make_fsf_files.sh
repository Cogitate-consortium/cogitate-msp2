#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
#"""

# Use the same subject list and filtering as in 02_make_timecourse_files.sh
BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_TEMPLATE_DIR="${GPPI_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files"
FMRIPREP_ROOT="${BIDS_ROOT}/derivatives/fmriprep"

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

declare -a RUNS=(1 2 3 4 5 6 7 8)
declare -a ANALYSES=(PPI_LOC PPI_FFA)

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fsf_dir="${FSF_FILES_ROOT}/${subj}"
    mkdir -p "$fsf_dir"

    for analysis in "${ANALYSES[@]}"; do
        echo "Creating first-level design file(s) for subject ${subj} analysis=${analysis}"
        template_fsf="${FSF_TEMPLATE_DIR}/sub-PXX_ses-V2_task-VG_run-X_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.fsf"

        for i_run in "${RUNS[@]}"; do
            run_base="${subj}_ses-V2_task-VG_run-${i_run}"
            fsf_name="${run_base}_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.fsf"
            fsf_tmp="${fsf_dir}/${fsf_name%.fsf}_.fsf"
            fsf_out="${fsf_dir}/${fsf_name}"
            nifti_file="${FMRIPREP_ROOT}/${subj_id}/ses-V2/func/${subj_id}_ses-V2_task-VG_run-${i_run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

            sed "s/sub-PXX/sub-${subj}/g" "$template_fsf" > "$fsf_tmp"
            sed "s/run-X/run-${i_run}/g" "$fsf_tmp" > "$fsf_out"
            rm -f "$fsf_tmp"

            NO_VOLUMES=$(fslhd "$nifti_file" | awk '/^dim4/ {print $2; exit}')
            sed -i "s!set fmri(npts) XXX_NTPS!set fmri(npts) ${NO_VOLUMES}!g" "$fsf_out"
        done
    done
done
