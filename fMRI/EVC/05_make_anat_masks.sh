#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Submit one Slurm job per subject: FreeSurfer V1/V2 exvivo labels → T1w masks (ses-V2 anat, else ses-V1).
# Run before 06_make_evc_roi.sh (can run in parallel with 04_run_EVC_1st_level.sh).
#
# Usage: bash 05_make_anat_masks.sh

EVC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evc_config.sh
source "${EVC_DIR}/evc_config.sh"

JOB_FILES_ROOT="${EVC_CODE_ROOT}/job_files/run_make_anat_masks"
SLURM_TEMPLATE="${EVC_CODE_ROOT}/slurm_templates/EVC_anat_masks.sh"

# Prefer ses-V2 T1w; fall back to ses-V1. Sets T1W_ANAT_SESSION and t1w on success.
evc_resolve_t1w_anat() {
    local subj_id="$1"
    local ses
    for ses in ses-V2 ses-V1; do
        t1w="${BIDS_ROOT}/${subj_id}/${ses}/anat/${subj_id}_${ses}_acq-anat_run-1_T1w.nii.gz"
        if [ -f "$t1w" ]; then
            T1W_ANAT_SESSION="$ses"
            return 0
        fi
    done
    t1w=""
    T1W_ANAT_SESSION=""
    return 1
}

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: ${SLURM_TEMPLATE}" >&2
    exit 1
fi

if ! evc_load_subjects; then
    exit 1
fi

echo ">> Number of subjects: ${#SUBJECTS[@]} (${EVC_SUBJECT_FILTER_DESC})"

n_submitted=0
n_skipped=0
n_preflight_failed=0
n_sbatch_failed=0

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fs_subj_dir="${FREESURFER_ROOT}/${subj_id}"
    anat_dir="${EVC_ROIS_ROOT}/${subj_id}/anat"
    done_marker="${anat_dir}/.${subj_id}_V1V2_T1w.done"

    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    run_marker="${job_dir}/${subj_id}_anat_masks.run"
    slurm_script="${job_dir}/slurm_${subj_id}_anat_masks.sh"
    job_log="${job_dir}/anat_masks_${subj_id}.log"
    slurm_out="${job_dir}/slurm_${subj_id}_anat_masks_%j.out"

    if [ -f "$done_marker" ] && [ -f "${anat_dir}/${subj_id}_V1V2.nii.gz" ]; then
        n_skipped=$((n_skipped + 1))
        echo ">> ${subj_id}: submitted=0 skipped=1 preflight_failed=0 sbatch_failed=0"
        continue
    fi

    if [ -f "$run_marker" ] && grep -q '^status=SUBMITTED' "$run_marker" 2>/dev/null; then
        n_skipped=$((n_skipped + 1))
        echo ">> ${subj_id}: submitted=0 skipped=1 (job already submitted; see ${run_marker})"
        continue
    fi

    preflight_failed=0
    if [ ! -f "${fs_subj_dir}/scripts/recon-all.done" ]; then
        echo "ERROR: ${subj_id}: FreeSurfer not finished (${fs_subj_dir})" >&2
        preflight_failed=1
    elif ! evc_resolve_t1w_anat "$subj_id"; then
        echo "ERROR: ${subj_id}: missing T1w (tried ses-V2, ses-V1)" >&2
        preflight_failed=1
    fi

    if [ "$preflight_failed" -eq 1 ]; then
        n_preflight_failed=$((n_preflight_failed + 1))
        echo ">> ${subj_id}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
        continue
    fi

    cat > "$run_marker" <<EOF
# EVC anat masks — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
subj_id=${subj_id}
t1w_anat_session=${T1W_ANAT_SESSION}
anat_dir=${anat_dir}
job_log=${job_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF
    {
        awk '/^### EVC_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
        printf '#SBATCH --output=%s\n' "$slurm_out"
        printf '#SBATCH --error=%s\n' "$slurm_out"
        printf 'export RUN_MARKER=%q\n' "$run_marker"
        printf 'export JOB_LOG=%q\n' "$job_log"
        printf 'export SUBJ_ID=%q\n' "$subj_id"
        printf 'export EVC_DIR=%q\n' "$EVC_DIR"
        printf 'export T1W_ANAT_SESSION=%q\n' "$T1W_ANAT_SESSION"
        awk '/^### EVC_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
    } > "$slurm_script"

    chmod +x "$slurm_script"
    sbatch_failed=0
    if ! job_out=$(sbatch "$slurm_script" 2>&1); then
        {
            echo ""
            echo "========== sbatch failed $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        echo "status=SBATCH_FAILED" >> "$run_marker"
        echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
        sbatch_failed=1
        n_sbatch_failed=$((n_sbatch_failed + 1))
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        n_submitted=$((n_submitted + 1))
    fi

    echo ">> ${subj_id}: submitted=$((1 - sbatch_failed)) skipped=0 preflight_failed=0 sbatch_failed=${sbatch_failed}"
done

echo ">> summary: submitted=${n_submitted} skipped=${n_skipped} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"

if [ "$n_preflight_failed" -gt 0 ] || [ "$n_sbatch_failed" -gt 0 ]; then
    exit 1
fi
