#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Local (no Slurm) per-run fslmeants jobs (see job_templates/make_timecourse_files.sh).
#"""

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"
# shellcheck source=local_parallel.sh
source "${_SCRIPT_DIR}/local_parallel.sh"
gppi_local_init || exit 1

GPPI_CODE_ROOT="${_SCRIPT_DIR}"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/make_timecourse_files"
JOB_TEMPLATE="${GPPI_CODE_ROOT}/job_templates/make_timecourse_files.sh"
SEEDS_ROOT="${BIDS_ROOT}/derivatives/gppi_seeds"
TIMECOURSE_ROOT="${BIDS_ROOT}/derivatives/gppi_timecourse"
FMRIPREP_ROOT="${BIDS_ROOT}/derivatives/fmriprep"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$JOB_TEMPLATE" ]; then
    echo "ERROR: job template not found: $JOB_TEMPLATE" >&2
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
# SUBJECTS=(SC108)

declare -a RUNS=(1 2 3 4)

mkdir -p "$JOB_FILES_ROOT"
job_list="${JOB_FILES_ROOT}/_queued_jobs.txt"
: > "$job_list"
n_queued=0
n_skipped=0

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    echo ">> queue timecourse jobs for: ${subj_id}"

    for i_run in "${RUNS[@]}"; do
        run_base="${subj_id}_ses-V2_task-Replay_run-${i_run}"
        seed_path="${SEEDS_ROOT}/${subj_id}"
        tc_path="${TIMECOURSE_ROOT}/${subj_id}"
        seed_LOC="${seed_path}/${subj_id}_rel_irrel_bh_object_LOC_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"
        seed_FFA="${seed_path}/${subj_id}_rel_irrel_bh_face_FFA_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"
        func_data="${FMRIPREP_ROOT}/${subj_id}/ses-V2/func/${run_base}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ts_LOC="${tc_path}/${run_base}_timecourse_LOC.txt"
        ts_FFA="${tc_path}/${run_base}_timecourse_FFA.txt"

        run_marker="${job_dir}/${subj_id}_ses-V2_task-Replay_run-${i_run}_timecourse.run"
        job_script="${job_dir}/local_${subj_id}_run-${i_run}_timecourse.sh"
        job_log="${job_dir}/timecourse_${subj_id}_run-${i_run}.log"

        if [ -f "$run_marker" ]; then
            echo " > Skipping, marker exists: $run_marker"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        cat > "$run_marker" <<EOF
# gPPI seed timecourses — marker / local job summary
status=QUEUED
date_queued=$(date -Iseconds)
submit_host=$(hostname)
user=${USER:-unknown}
subj_id=${subj_id}
subj_code=${subj}
run=${i_run}
func_data=${func_data}
seed_loc=${seed_LOC}
seed_ffa=${seed_FFA}
ts_loc=${ts_LOC}
ts_ffa=${ts_FFA}
job_log=${job_log}
job_script=${job_script}
---
EOF
        gppi_assemble_job "$job_script" "$JOB_TEMPLATE" "### GPPI_EXPORT_BLOCK_INSERT ###" \
            "$(printf 'export RUN_MARKER=%q' "$run_marker")" \
            "$(printf 'export JOB_LOG=%q' "$job_log")" \
            "$(printf 'export FUNC_DATA=%q' "$func_data")" \
            "$(printf 'export SEED_LOC=%q' "$seed_LOC")" \
            "$(printf 'export SEED_FFA=%q' "$seed_FFA")" \
            "$(printf 'export TS_LOC=%q' "$ts_LOC")" \
            "$(printf 'export TS_FFA=%q' "$ts_FFA")"
        printf '%s\n' "$job_script" >> "$job_list"
        n_queued=$((n_queued + 1))
    done
done

echo ">> queued=${n_queued} skipped=${n_skipped} GPPI_NJOBS=${GPPI_NJOBS}"
n_run_failed=0
if ! gppi_run_job_list "$job_list"; then
    n_run_failed=1
fi
echo ">> timecourse local run finished (run_failed=${n_run_failed})"
if [ "$n_run_failed" -ne 0 ]; then
    echo "ERROR: one or more timecourse jobs failed (see job_files/make_timecourse_files/*.run)" >&2
    exit 1
fi
