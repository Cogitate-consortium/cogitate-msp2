#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Generate per-subject EVC localizer 1st-level FSFs (T1w BOLD, registration off).
# Subjects: SYNCHRONY_min_seen == TRUE (see evc_config.sh).
# Run after 02_create_regressor_txt_files.py, before 04_run_EVC_1st_level.sh.
#
# Usage: bash 03_make_fsf_files.sh

EVC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evc_config.sh
source "${EVC_DIR}/evc_config.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if ! evc_load_subjects; then
    exit 1
fi

echo ">> Number of subjects: ${#SUBJECTS[@]} (${EVC_SUBJECT_FILTER_DESC}; ${SUBJECT_CSV})"

template_fsf="${FSF_TEMPLATE_DIR}/sub-PXX_${EVC_SESSION}_${EVC_TASK}_analysis-1stROI_space-T1w.fsf"
if [ ! -f "$template_fsf" ]; then
    echo "ERROR: missing template: $template_fsf" >&2
    exit 1
fi
if ! evc_assert_no_legacy_regressor_paths "$template_fsf"; then
    exit 1
fi

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fsf_dir="${FSF_FILES_ROOT}/${subj}"
    mkdir -p "$fsf_dir"
    fsf_out="${fsf_dir}/${subj}_${EVC_SESSION}_${EVC_TASK}_analysis-1stROI_space-T1w.fsf"
    nifti_file="${FMRIPREP_ROOT}/${subj_id}/${EVC_SESSION}/func/${subj_id}_${EVC_SESSION}_${EVC_TASK}_space-T1w_desc-preproc_bold.nii.gz"

    sed -e "s/sub-PXX/sub-${subj}/g" \
        -e "s|__REGRESSOR_EVENT_ROOT__|${REGRESSOR_EVENT_ROOT}|g" \
        "$template_fsf" > "$fsf_out"
    sed -i "s|^set feat_files(1) .*|set feat_files(1) \"${nifti_file}\"|g" "$fsf_out"
    if ! evc_assert_no_legacy_regressor_paths "$fsf_out"; then
        exit 1
    fi

    if ! NO_VOLUMES=$(fslhd "$nifti_file" 2>/dev/null | awk '/^dim4/ {print $2; exit}'); then
        echo "ERROR: ${subj_id}: cannot read dim4 from ${nifti_file}" >&2
        continue
    fi
    sed -i "s!set fmri(npts) XXX_NTPS!set fmri(npts) ${NO_VOLUMES}!g" "$fsf_out"
    echo ">> ${subj_id}: ${fsf_out} npts=${NO_VOLUMES}"
done
