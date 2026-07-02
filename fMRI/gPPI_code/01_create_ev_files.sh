#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
#"""

#module load FSL

# Subjects from ses-v2-analysis-subs-fmri.csv with SYNCHRONY_min_seen == TRUE only
BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
SUBJECT_CSV="${BIDS_ROOT}/code/ses-v2-analysis-subs-fmri.csv"
EV_FILES_ROOT="${BIDS_ROOT}/derivatives/regressoreventfiles"

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
    print(f"sub-{sub}")
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

# Run just one subject for testing
#SUBJECTS=(SC108)

declare -a RUNS=(1 2 3 4 5 6 7 8)

for subj_id in "${SUBJECTS[@]}"; do
    echo ">> create ev-files for: ${subj_id}"
    ev_dir="${EV_FILES_ROOT}/${subj_id}/ses-V2"

    for i_run in "${RUNS[@]}"; do
        run_base="${subj_id}_ses-V2_task-VG_run-${i_run}"

        # dAT — Object
        cat "${ev_dir}/${run_base}_probedSeenObjectRight_EV.txt" > "${ev_dir}/${run_base}_probedSeenObject.txt"
        cat "${ev_dir}/${run_base}_probedSeenObjectLeft_EV.txt" >> "${ev_dir}/${run_base}_probedSeenObject.txt"

        # Face
        cat "${ev_dir}/${run_base}_probedSeenFaceRight_EV.txt" > "${ev_dir}/${run_base}_probedSeenFace.txt"
        cat "${ev_dir}/${run_base}_probedSeenFaceLeft_EV.txt" >> "${ev_dir}/${run_base}_probedSeenFace.txt"

        obj_evfile="${ev_dir}/${run_base}_probedSeenObject.txt"
        face_evfile="${ev_dir}/${run_base}_probedSeenFace.txt"

        # Preferred Object: object as-is + face with negated intensity
        outfile="${ev_dir}/${run_base}_prefer_Object.txt"
        cat "${obj_evfile}" > "${outfile}"
        awk '{if(NF>=3){$3=-1*$3; print $1, $2, $3}else{print}}' "${face_evfile}" >> "${outfile}"

        # Preferred Face: face as-is + object with negated intensity
        outfile="${ev_dir}/${run_base}_prefer_Face.txt"
        cat "${face_evfile}" > "${outfile}"
        awk '{if(NF>=3){$3=-1*$3; print $1, $2, $3}else{print}}' "${obj_evfile}" >> "${outfile}"
    done
done
