#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Generate per-subject 2nd-level FIR FSFs (combined Seen/Unseen).
# Run after 04_create_identity_reg_for_2nd_lvl.sh; before 06_run_FIR_2nd_level_VG.sh.
#
# Usage: bash 05_make_fsf_files_2nd_lvl.sh

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_TEMPLATE_DIR="${FIR_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

make_2nd_fsf_for_variant() {
    local variant="$1"
    if ! fir_set_variant "$variant"; then
        return 1
    fi

    local template_fsf="${FSF_TEMPLATE_DIR}/${SECOND_GLM_TEMPLATE}"
    if [ ! -f "$template_fsf" ]; then
        echo "ERROR: missing template: $template_fsf" >&2
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
        mkdir -p "$fsf_dir"
        fsf_out="${fsf_dir}/${subj}_ses-V2_task-VG_analysis-2ndGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.fsf"

        sed -e "s/sub-PXX/sub-${subj}/g" "$template_fsf" >"$fsf_out"
        echo ">> ${subj_id} (${variant}): fsf_files=1"
    done
}

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    make_2nd_fsf_for_variant "$variant" || exit 1
done
