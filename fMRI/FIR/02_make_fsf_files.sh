#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Generate per-subject 1st-level FIR FSFs (combined Seen/Unseen).

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_TEMPLATE_DIR="${FIR_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files"
EV_FILES_ROOT="${BIDS_ROOT}/derivatives/regressoreventfiles"
FMRIPREP_ROOT="${BIDS_ROOT}/derivatives/fmriprep"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

declare -a RUNS=(1 2 3 4 5 6 7 8)

make_fsf_for_variant() {
    local variant="$1"
    if ! fir_set_variant "$variant"; then
        return 1
    fi

    local template_fsf="${FSF_TEMPLATE_DIR}/${FIRST_GLM_TEMPLATE}"
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
        n_written=0

        for i_run in "${RUNS[@]}"; do
            run_base="${subj}_ses-V2_task-VG_run-${i_run}"
            fsf_name="${run_base}_analysis-1stGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.fsf"
            fsf_tmp="${fsf_dir}/${fsf_name%.fsf}_.fsf"
            fsf_out="${fsf_dir}/${fsf_name}"
            nifti_file="${FMRIPREP_ROOT}/${subj_id}/ses-V2/func/${subj_id}_ses-V2_task-VG_run-${i_run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

            sed -e "s/sub-PXX/sub-${subj}/g" -e "s/run-X/run-${i_run}/g" "$template_fsf" >"$fsf_tmp"
            mv "$fsf_tmp" "$fsf_out"

            seen_ev="${EV_FILES_ROOT}/${subj_id}/ses-V2/FIR/${subj_id}-ses-V2_task-VG_run-${i_run}_${SEEN_EV_STEM}_Shifted.txt"
            unseen_ev="${EV_FILES_ROOT}/${subj_id}/ses-V2/FIR/${subj_id}-ses-V2_task-VG_run-${i_run}_${UNSEEN_EV_STEM}_Shifted.txt"
            sed -i "s|^set feat_files(1) .*|set feat_files(1) \"${nifti_file}\"|g" "$fsf_out"
            sed -i "s|^set fmri(custom1) .*|set fmri(custom1) \"${seen_ev}\"|g" "$fsf_out"
            sed -i "s|^set fmri(custom2) .*|set fmri(custom2) \"${unseen_ev}\"|g" "$fsf_out"

            NO_VOLUMES=$(fslhd "$nifti_file" | awk '/^dim4/ {print $2; exit}')
            sed -i "s!set fmri(npts) XXX_NTPS!set fmri(npts) ${NO_VOLUMES}!g" "$fsf_out"
            n_written=$((n_written + 1))
        done

        echo ">> ${subj_id} (${variant}): fsf_files=${n_written}"
    done
}

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    make_fsf_for_variant "$variant" || exit 1
done
