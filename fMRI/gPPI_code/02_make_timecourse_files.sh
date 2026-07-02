#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
#"""

# Submits per-run fslmeants jobs via Slurm (see slurm_templates/make_timecourse_files.sh).

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/make_timecourse_files"
SLURM_TEMPLATE="${GPPI_CODE_ROOT}/slurm_templates/make_timecourse_files.sh"
SEEDS_ROOT="${BIDS_ROOT}/derivatives/gppi_seeds"
TIMECOURSE_ROOT="${BIDS_ROOT}/derivatives/gppi_timecourse"
FMRIPREP_ROOT="${BIDS_ROOT}/derivatives/fmriprep"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
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

declare -a RUNS=(1 2 3 4 5 6 7 8)

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    echo ">> submit timecourse jobs for: ${subj_id}"

    for i_run in "${RUNS[@]}"; do
        run_base="${subj_id}_ses-V2_task-VG_run-${i_run}"
        seed_path="${SEEDS_ROOT}/${subj_id}"
        tc_path="${TIMECOURSE_ROOT}/${subj_id}"
        seed_LOC="${seed_path}/${subj_id}_rel_irrel_bh_object_LOC_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"
        seed_FFA="${seed_path}/${subj_id}_rel_irrel_bh_face_FFA_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"
        func_data="${FMRIPREP_ROOT}/${subj_id}/ses-V2/func/${run_base}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ts_LOC="${tc_path}/${run_base}_timecourse_LOC.txt"
        ts_FFA="${tc_path}/${run_base}_timecourse_FFA.txt"

        run_marker="${job_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_timecourse.run"
        slurm_script="${job_dir}/slurm_${subj_id}_run-${i_run}_timecourse.sh"
        job_log="${job_dir}/timecourse_${subj_id}_run-${i_run}.log"
        slurm_out="${job_dir}/slurm_${subj_id}_run-${i_run}_timecourse_%j.out"

        if [ ! -f "$run_marker" ]; then
            cat > "$run_marker" <<EOF
# gPPI seed timecourses — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
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
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF
            {
                awk '/^### GPPI_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
                printf '#SBATCH --output=%s\n' "$slurm_out"
                printf '#SBATCH --error=%s\n' "$slurm_out"
                printf 'export RUN_MARKER=%q\n' "$run_marker"
                printf 'export JOB_LOG=%q\n' "$job_log"
                printf 'export FUNC_DATA=%q\n' "$func_data"
                printf 'export SEED_LOC=%q\n' "$seed_LOC"
                printf 'export SEED_FFA=%q\n' "$seed_FFA"
                printf 'export TS_LOC=%q\n' "$ts_LOC"
                printf 'export TS_FFA=%q\n' "$ts_FFA"
                awk '/^### GPPI_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
            } > "$slurm_script"

            chmod +x "$slurm_script"
            if ! job_out=$(sbatch "$slurm_script" 2>&1); then
                {
                    echo ""
                    echo "========== sbatch failed $(date -Iseconds) =========="
                    echo "${job_out}"
                } >> "$run_marker"
                echo "status=SBATCH_FAILED" >> "$run_marker"
                echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
            else
                {
                    echo ""
                    echo "========== sbatch $(date -Iseconds) =========="
                    echo "${job_out}"
                } >> "$run_marker"
            fi
        else
            echo " > Skipping, marker exists: $run_marker"
        fi
    done
done
