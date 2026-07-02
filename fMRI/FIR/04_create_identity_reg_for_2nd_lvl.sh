#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# After 1st-level FIR FEAT with registration off (fMRIPrep MNI BOLD), create dummy reg/
# in each run-level .feat so 2nd-level FEAT can merge inputs. Run before
# 05_make_fsf_files_2nd_lvl.sh / 06_run_FIR_2nd_level_VG.sh.
#
# Usage: bash 04_create_identity_reg_for_2nd_lvl.sh
#
# Requires FSL in PATH. Shared fallbacks/QC: glm/identity_reg_for_2nd_lvl.sh

module load FSL

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
JOB_FILES_ROOT="${FIR_CODE_ROOT}/job_files/create_identity_reg_for_2nd_lvl"
IDENTITY_REG_LIB="${CODE_PATH}/glm/identity_reg_for_2nd_lvl.sh"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"
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

declare -a RUNS=(1 2 3 4 5 6 7 8)

run_variant_identity_reg() {
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
        job_dir="${JOB_FILES_ROOT}/${variant}/${subj}"
        mkdir -p "$job_dir"
        n_created=0
        n_skipped=0
        n_missing_feat=0
        n_failed=0

        for i_run in "${RUNS[@]}"; do
            feat_analysis="${subj_id}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${FEAT_SUFFIX}.feat"
            feat_path="${FSLFEAT_ROOT}/${FSLFEAT_SUBDIR}/${subj_id}/${feat_analysis}"
            reg_marker="${job_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${FIR_VARIANT}-MNI152NLin2009cAsym_identity_reg.run"

            if [ ! -d "$feat_path" ]; then
                n_missing_feat=$((n_missing_feat + 1))
                continue
            fi

            if [ -f "$reg_marker" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi

            if run_identity_reg_and_qc "$feat_path" "$reg_marker"; then
                n_created=$((n_created + 1))
            else
                n_failed=$((n_failed + 1))
            fi
        done

        echo ">> ${subj_id} (${variant}): created=${n_created} skipped=${n_skipped} missing_feat=${n_missing_feat} failed=${n_failed}"
    done
}

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    run_variant_identity_reg "$variant" || exit 1
done
