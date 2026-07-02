#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# After 2nd-level FIR FEAT (.gfeat), create dummy reg/ at each subject's 2nd-level
# .gfeat root for 3rd-level FEAT (inputtype 1 + copeN.feat inputs). One reg per
# subject/variant serves all copes 1–42.
#
# Run after 06_run_FIR_2nd_level_VG.sh; before 08_make_fsf_files_3rdlevel.sh.
#
# Usage: bash 07_create_identity_reg_for_3rd_lvl.sh
#
# Cluster rerun (FIR 3rd level): 07 -> 08_make -> delete group .gfeat (or
# delete_fslfeat_derivatives.py --analysis FIR --level 3rd --execute) ->
# clear job_files/run_FIR_3rd_level/*/*.run -> 09_run_FIR_3rd_level.sh
#
# Requires FSL in PATH. Shared helpers: glm/identity_reg_for_2nd_lvl.sh

module load FSL

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
JOB_FILES_ROOT="${FIR_CODE_ROOT}/job_files/create_identity_reg_for_3rd_lvl"
IDENTITY_REG_LIB="${CODE_PATH}/glm/identity_reg_for_2nd_lvl.sh"

# shellcheck source=identity_reg_for_2nd_lvl.sh
source "${IDENTITY_REG_LIB}"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "${FSLDIR}/etc/flirtsch/ident.mat" ]; then
    echo "ERROR: identity matrix not found (is FSL loaded?): ${FSLDIR}/etc/flirtsch/ident.mat" >&2
    exit 1
fi

report_ok() {
    local feat_dir="$1"
    local report="${feat_dir}/report.html"
    if [ ! -f "$report" ]; then
        return 1
    fi
    if grep -qE 'Error|ERROR' "$report" 2>/dev/null; then
        return 1
    fi
    return 0
}

n_total_failed=0
n_total_created=0
n_total_skipped=0
n_total_missing=0

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    fir_set_variant "$variant" || exit 1
    job_dir="${JOB_FILES_ROOT}/${variant}"
    mkdir -p "$job_dir"

    while IFS= read -r sub_code || [ -n "${sub_code}" ]; do
        [ -n "$sub_code" ] || continue
        subj_id="sub-${sub_code}"
        parent_stem="${subj_id}_ses-V2_task-VG_analysis-2ndGLM_${variant}-MNI152NLin2009cAsym"
        gfeat_path=""
        for ext in .gfeat .feat; do
            candidate="${FSLFEAT_ROOT}/${variant}/${subj_id}/${parent_stem}${ext}"
            if [ -d "$candidate" ]; then
                gfeat_path="$candidate"
                break
            fi
        done

        marker="${job_dir}/${parent_stem}.identity_reg.run"

        if [ -z "$gfeat_path" ]; then
            n_total_missing=$((n_total_missing + 1))
            continue
        fi

        if [ -f "$marker" ]; then
            if spot_check_identity_reg_gfeat "$gfeat_path" 1 >/dev/null 2>&1; then
                n_total_skipped=$((n_total_skipped + 1))
                continue
            fi
            echo "WARN: marker present but reg/QC missing; re-creating identity reg for ${gfeat_path}" >&2
        fi

        if ! report_ok "$gfeat_path"; then
            echo "WARN: skip ${gfeat_path} (bad/missing report.html)" >&2
            n_total_missing=$((n_total_missing + 1))
            continue
        fi

        if [ ! -d "${gfeat_path}/cope1.feat" ]; then
            echo "WARN: skip ${gfeat_path} (missing cope1.feat for QC)" >&2
            n_total_missing=$((n_total_missing + 1))
            continue
        fi

        if run_identity_reg_and_qc_gfeat "$gfeat_path" 1 "$marker"; then
            n_total_created=$((n_total_created + 1))
            echo ">> ${variant} ${subj_id}: identity reg OK (${gfeat_path})"
        else
            n_total_failed=$((n_total_failed + 1))
            echo "ERROR: identity reg failed for ${gfeat_path}" >&2
        fi
    done < <(fir_list_subjects)
done

echo ">> summary: created=${n_total_created} skipped_marker=${n_total_skipped} missing_or_skip=${n_total_missing} failed=${n_total_failed}"

if [ "$n_total_failed" -gt 0 ]; then
    echo "ERROR: identity reg creation failed for ${n_total_failed} 2nd-level .gfeat run(s)" >&2
    exit 1
fi
