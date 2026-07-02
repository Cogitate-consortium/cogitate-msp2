# Shared helpers for EVC functional ROI (06_make_evc_roi.sh). Source after evc_config.sh.
# Not executable on its own.

: "${EVC_SESSION:?}"
: "${EVC_TASK:?}"
: "${FMRIPREP_ROOT:?}"
: "${EVC_ROIS_ROOT:?}"

MNI_REF=""

evc_find_mni_ref() {
    local subj_id="$1"
    local _func_dir="${FMRIPREP_ROOT}/${subj_id}/${EVC_SESSION}/func"
    local _prefix="${subj_id}_${EVC_SESSION}_${EVC_TASK}"
    local _cand

    MNI_REF=""
    for _cand in \
        "${_func_dir}/${_prefix}_space-MNI152NLin2009cAsym_boldref.nii.gz" \
        "${_func_dir}/${subj_id}_${EVC_SESSION}_task-VG_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz" \
        "${_func_dir}/${subj_id}_ses-V1_task-Dur_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz"; do
        if [ -f "$_cand" ]; then
            MNI_REF="$_cand"
            return 0
        fi
    done
    return 1
}

evc_warp_to_mni() {
    local in_img="$1"
    local out_img="$2"
    local interp="$3"
    local subj_id="$4"
    local _anat_dir_v2="${FMRIPREP_ROOT}/${subj_id}/${EVC_SESSION}/anat"
    local _anat_dir_v1="${FMRIPREP_ROOT}/${subj_id}/ses-V1/anat"
    local _xfm=""
    local _cand

    if flirt -in "$in_img" -ref "$MNI_REF" -out "$out_img" -applyxfm -usesqform -interp "$interp"; then
        return 0
    fi

    for _cand in \
        "${_anat_dir_v2}/${subj_id}_${EVC_SESSION}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" \
        "${_anat_dir_v1}/${subj_id}_ses-V1_acq-anat_run-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"; do
        if [ -f "$_cand" ]; then
            _xfm="$_cand"
            break
        fi
    done

    if [ -z "$_xfm" ]; then
        return 1
    fi

    local _ants_interp="Linear"
    [ "$interp" = "nearestneighbour" ] && _ants_interp="NearestNeighbor"

    antsApplyTransforms \
        --input "$in_img" \
        --reference-image "$MNI_REF" \
        --output "$out_img" \
        --interpolation "$_ants_interp" \
        --transform "$_xfm"
}

evc_build_top_voxel_roi() {
    local masked_tlbr="$1"
    local masked_trbl="$2"
    local roi_out="$3"
    local top_n="$4"
    local total_vox="$5"

    MASKED_TLBR="$masked_tlbr" \
    MASKED_TRBL="$masked_trbl" \
    ROI_OUT="$roi_out" \
    EVC_ROI_TOP_N_PER_CONTRAST="$top_n" \
    EVC_ROI_TOTAL_VOXELS="$total_vox" \
    python3 <<'PY'
import os
import sys
import numpy as np
import nibabel as nib

top_n = int(os.environ["EVC_ROI_TOP_N_PER_CONTRAST"])
total_vox = int(os.environ["EVC_ROI_TOTAL_VOXELS"])
fill_n = max(0, total_vox - top_n)
paths = {
    "tlbr": os.environ["MASKED_TLBR"],
    "trbl": os.environ["MASKED_TRBL"],
}
out_path = os.environ["ROI_OUT"]

img1 = nib.load(paths["tlbr"])
d1 = img1.get_fdata().ravel()
d2 = nib.load(paths["trbl"]).get_fdata().ravel()
affine = img1.affine

mask = np.zeros(d1.size, dtype=np.float32)

k1 = min(top_n, d1.size)
idx1 = np.argsort(d1)[::-1][:k1]
mask[idx1] = 1.0

if fill_n > 0:
    d2_rank = d2.copy()
    d2_rank[mask > 0] = -np.inf
    k2 = min(fill_n, d2.size)
    idx2 = np.argsort(d2_rank)[::-1][:k2]
    idx2 = idx2[d2_rank[idx2] > -np.inf]
    mask[idx2] = 1.0

roi = mask.reshape(img1.shape)
n_vox = int(mask.sum())
nib.save(nib.Nifti1Image(roi, affine), out_path)
print(n_vox, file=sys.stderr)
PY
}
