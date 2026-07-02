#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Submit one Slurm job per subject: func-grid ROI, top-150+fill to 300, warp to MNI.
# Run after 04_run_EVC_1st_level.sh and 05_make_anat_masks.sh.
#
# Usage: bash 06_make_evc_roi.sh

EVC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evc_config.sh
source "${EVC_DIR}/evc_config.sh"
# shellcheck source=evc_make_evc_roi_functions.sh
source "${EVC_DIR}/evc_make_evc_roi_functions.sh"

JOB_FILES_ROOT="${EVC_CODE_ROOT}/job_files/run_make_evc_roi"
SLURM_TEMPLATE="${EVC_CODE_ROOT}/slurm_templates/EVC_make_evc_roi.sh"

EVC_ROI_FILL_N=$((EVC_ROI_TOTAL_VOXELS - EVC_ROI_TOP_N_PER_CONTRAST))

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

echo ">> subjects=${#SUBJECTS[@]} roi=${EVC_ROI_TOP_N_PER_CONTRAST}+${EVC_ROI_FILL_N}=${EVC_ROI_TOTAL_VOXELS} (${EVC_SUBJECT_FILTER_DESC})"

n_submitted=0
n_skipped=0
n_preflight_failed=0
n_sbatch_failed=0

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    feat_path="${FSLFEAT_ROOT}/EVC/${subj_id}/${subj_id}_${FEAT_BASENAME}.feat"
    roi_dir="${EVC_ROIS_ROOT}/${subj_id}"
    anat_v1v2="${roi_dir}/anat/${subj_id}_V1V2.nii.gz"
    roi_mni="${roi_dir}/${subj_id}_evc_300_V1V2.nii.gz"

    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    run_marker="${job_dir}/${subj_id}_make_evc_roi.run"
    slurm_script="${job_dir}/slurm_${subj_id}_make_evc_roi.sh"
    job_log="${job_dir}/make_evc_roi_${subj_id}.log"
    slurm_out="${job_dir}/slurm_${subj_id}_make_evc_roi_%j.out"

    if [ -f "$roi_mni" ]; then
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
    if [ ! -d "${feat_path}/stats" ] || [ ! -f "${feat_path}/report.html" ]; then
        echo "ERROR: ${subj_id}: missing or incomplete FEAT: ${feat_path}" >&2
        preflight_failed=1
    elif [ ! -f "${feat_path}/stats/zstat4.nii.gz" ] || [ ! -f "${feat_path}/stats/zstat5.nii.gz" ]; then
        echo "ERROR: ${subj_id}: missing zstat4 or zstat5 in ${feat_path}/stats" >&2
        preflight_failed=1
    elif [ ! -f "$anat_v1v2" ]; then
        echo "ERROR: ${subj_id}: missing T1w V1V2 (run 05_make_anat_masks.sh): ${anat_v1v2}" >&2
        preflight_failed=1
    else
        func_ref="${feat_path}/mean_func.nii.gz"
        if [ ! -f "$func_ref" ]; then
            func_ref="${FMRIPREP_ROOT}/${subj_id}/${EVC_SESSION}/func/${subj_id}_${EVC_SESSION}_${EVC_TASK}_space-T1w_desc-preproc_bold.nii.gz"
        fi
        if [ ! -f "$func_ref" ]; then
            echo "ERROR: ${subj_id}: no functional reference for resampling" >&2
            preflight_failed=1
        elif ! evc_find_mni_ref "$subj_id"; then
            echo "ERROR: ${subj_id}: missing MNI boldref (fMRIPrep)" >&2
            preflight_failed=1
        fi
    fi

    if [ "$preflight_failed" -eq 1 ]; then
        n_preflight_failed=$((n_preflight_failed + 1))
        echo ">> ${subj_id}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
        continue
    fi

    cat > "$run_marker" <<EOF
# EVC make_evc_roi — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
subj_id=${subj_id}
roi_dir=${roi_dir}
feat_path=${feat_path}
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
