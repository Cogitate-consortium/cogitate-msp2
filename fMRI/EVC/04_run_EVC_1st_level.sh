#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Slurm 1st-level EVC localizer FEAT (T1w space, registration off). Run after 03_make_fsf_files.sh.
#
# Usage: bash 04_run_EVC_1st_level.sh

export OPENBLAS_NUM_THREADS=1

EVC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evc_config.sh
source "${EVC_DIR}/evc_config.sh"

JOB_FILES_ROOT="${EVC_CODE_ROOT}/job_files/run_EVC_1st_level"
SLURM_TEMPLATE="${EVC_CODE_ROOT}/slurm_templates/EVC_1st_lvl.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

if ! evc_load_subjects; then
    exit 1
fi

echo ">> Number of subjects: ${#SUBJECTS[@]} (${EVC_SUBJECT_FILTER_DESC})"

fsf_path_exists() {
    local path="$1"
    [[ -e "${path}" ]] && return 0
    [[ "${path}" == *.nii.gz ]] && return 1
    [[ -e "${path}.nii.gz" ]] && return 0
    return 1
}

check_fsf_inputs() {
    local fsf="$1"
    local missing="" line t path
    if ! evc_assert_no_legacy_regressor_paths "$fsf"; then
        return 1
    fi
    while IFS= read -r line || [ -n "${line}" ]; do
        [[ "${line}" == *\"*\"* ]] || continue
        t="${line#*\"}"
        path="${t%%\"*}"
        [[ -n "${path}" ]] || continue
        if ! fsf_path_exists "${path}"; then
            missing="${missing}${missing:+$'\n'}  missing: ${path}"
        fi
    done < <(grep -E '^(set feat_files\(1\)|set confoundev_files\(1\)|set fmri\(custom[0-9]+\))' "${fsf}" 2>/dev/null || true)
    if [[ -n "${missing}" ]]; then
        printf '%s\n' "referenced paths from FSF are missing:${missing}" >&2
        if [[ "${missing}" == *"_task-EVCLoc_"*".txt" ]]; then
            printf '%s\n' \
                "Hint: run EVC/02_create_regressor_txt_files.py after 01_create_events_tsv.py (outputs: ${REGRESSOR_EVENT_ROOT}/)." >&2
        fi
        return 1
    fi
    return 0
}

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fsf_dir="${FSF_FILES_ROOT}/${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"
    n_submitted=0
    n_skipped=0
    n_sbatch_failed=0
    n_preflight_failed=0

    run_marker="${job_dir}/${subj_id}_${FEAT_BASENAME}.run"
    slurm_script="${job_dir}/slurm_${subj_id}_EVCLoc_${FEAT_SUFFIX}.sh"
    fsf_file="${fsf_dir}/${subj}_${EVC_SESSION}_${EVC_TASK}_analysis-1stROI_space-T1w.fsf"
    feat_log="${job_dir}/feat_${subj_id}_EVCLoc.log"
    slurm_out="${job_dir}/slurm_${subj_id}_EVCLoc_%j.out"
    feat_path="${FSLFEAT_ROOT}/EVC/${subj_id}/${subj_id}_${FEAT_BASENAME}.feat"

    if [ -f "$run_marker" ]; then
        n_skipped=1
        echo ">> ${subj_id}: submitted=0 skipped=1 preflight_failed=0 sbatch_failed=0"
        continue
    fi

    if [ -d "$feat_path" ] && [ -f "${feat_path}/report.html" ]; then
        n_skipped=1
        echo ">> ${subj_id}: skip (existing feat: ${feat_path})"
        continue
    fi

    if [ ! -f "$fsf_file" ]; then
        echo "ERROR: ${subj_id}: missing FSF file: ${fsf_file} (run 03_make_fsf_files.sh?)" >&2
        n_preflight_failed=1
        echo ">> ${subj_id}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
        continue
    fi

    if ! check_fsf_inputs "$fsf_file"; then
        echo "ERROR: ${subj_id}: preflight failed for ${fsf_file}" >&2
        n_preflight_failed=1
        echo ">> ${subj_id}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
        continue
    fi

    cat > "$run_marker" <<EOF
# EVC 1st-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
subj_id=${subj_id}
fsf_file=${fsf_file}
feat_log=${feat_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF
    {
        awk '/^### EVC_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
        printf '#SBATCH --output=%s\n' "$slurm_out"
        printf '#SBATCH --error=%s\n' "$slurm_out"
        printf 'export RUN_MARKER=%q\n' "$run_marker"
        printf 'export FEAT_LOG=%q\n' "$feat_log"
        printf 'export FSF_FILE=%q\n' "$fsf_file"
        awk '/^### EVC_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
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
        n_sbatch_failed=1
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        n_submitted=1
    fi

    echo ">> ${subj_id}: submitted=${n_submitted} skipped=${n_skipped} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"
done
