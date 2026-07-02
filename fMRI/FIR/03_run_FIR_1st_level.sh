#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Slurm 1st-level FEAT (combined FIR). Run after 02_make_fsf_files.sh.
#
# Usage: bash 03_run_FIR_1st_level.sh

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files"
JOB_FILES_ROOT="${FIR_CODE_ROOT}/job_files/run_FIR_1st_level"
SLURM_TEMPLATE="${FIR_CODE_ROOT}/slurm_templates/FIR_1st_lvl.sh"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

declare -a RUNS=(1 2 3 4 5 6 7 8)

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
        return 1
    fi
    return 0
}

run_variant_1st_level() {
    local variant="$1"
    if ! fir_set_variant "$variant"; then
        return 1
    fi

    _sublist_tmp="$(mktemp)"
    if ! fir_list_subjects >"$_sublist_tmp"; then
        rm -f "$_sublist_tmp"
        return 1
    fi
    mapfile -t SUBJECTS < "$_sublist_tmp"
    rm -f "$_sublist_tmp"

    if [ ${#SUBJECTS[@]} -eq 0 ]; then
        echo "ERROR: no subjects for ${variant} (${CSV_FILTER_COL})" >&2
        return 1
    fi

    echo ">> ${variant}: subjects=${#SUBJECTS[@]} (${CSV_FILTER_COL} == TRUE)"

    for subj in "${SUBJECTS[@]}"; do
        subj_id="sub-${subj}"
        fsf_dir="${FSF_FILES_ROOT}/${subj}"
        job_dir="${JOB_FILES_ROOT}/${variant}/${subj}"
        mkdir -p "$job_dir"
        n_submitted=0
        n_skipped=0
        n_sbatch_failed=0
        n_preflight_failed=0

        for i_run in "${RUNS[@]}"; do
            run_marker="${job_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.run"
            slurm_script="${job_dir}/slurm_${subj_id}_run-${i_run}_VG_${FIR_VARIANT}.sh"
            fsf_file="${fsf_dir}/${subj}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.fsf"
            feat_log="${job_dir}/feat_${subj_id}_run-${i_run}_VG_${FIR_VARIANT}.log"
            slurm_out="${job_dir}/slurm_${subj_id}_run-${i_run}_VG_${FIR_VARIANT}_%j.out"

            if [ -f "$run_marker" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi

            if [ ! -f "$fsf_file" ]; then
                echo "ERROR: ${subj_id} run-${i_run} (${variant}): missing FSF: ${fsf_file}" >&2
                n_preflight_failed=$((n_preflight_failed + 1))
                continue
            fi

            if ! check_fsf_inputs "$fsf_file"; then
                echo "ERROR: ${subj_id} run-${i_run} (${variant}): preflight failed for ${fsf_file}" >&2
                n_preflight_failed=$((n_preflight_failed + 1))
                continue
            fi

            cat >"$run_marker" <<EOF
# FIR 1st-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
subj_id=${subj_id}
run=${i_run}
variant=${FIR_VARIANT}
fsf_file=${fsf_file}
feat_log=${feat_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF
            {
                awk '/^### FIR_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
                printf '#SBATCH --output=%s\n' "$slurm_out"
                printf '#SBATCH --error=%s\n' "$slurm_out"
                printf 'export RUN_MARKER=%q\n' "$run_marker"
                printf 'export FEAT_LOG=%q\n' "$feat_log"
                printf 'export FSF_FILE=%q\n' "$fsf_file"
                awk '/^### FIR_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
            } >"$slurm_script"

            chmod +x "$slurm_script"
            if ! job_out=$(sbatch "$slurm_script" 2>&1); then
                {
                    echo ""
                    echo "========== sbatch failed $(date -Iseconds) =========="
                    echo "${job_out}"
                } >>"$run_marker"
                echo "status=SBATCH_FAILED" >>"$run_marker"
                echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
                n_sbatch_failed=$((n_sbatch_failed + 1))
            else
                {
                    echo ""
                    echo "========== sbatch $(date -Iseconds) =========="
                    echo "${job_out}"
                } >>"$run_marker"
                n_submitted=$((n_submitted + 1))
            fi
        done

        echo ">> ${subj_id} (${variant}): submitted=${n_submitted} skipped=${n_skipped} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"
    done
}

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    run_variant_1st_level "$variant" || exit 1
done
