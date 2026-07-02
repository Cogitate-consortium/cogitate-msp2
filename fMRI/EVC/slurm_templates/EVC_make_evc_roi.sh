#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=2:00:00
#SBATCH --mem=12G

### EVC_EXPORT_BLOCK_INSERT ###
# 06_make_evc_roi.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FSL/6.0.2-foss-2019a-Python-3.7.2 2>/dev/null || module load FSL
module load ANTs/2.3.2-foss-2019a-Python-3.7.2 2>/dev/null || true
module load SciPy-bundle/2024.05-gfbf-2024a 2>/dev/null || true

{
  echo ""
  echo "========== job start $(date -Iseconds) =========="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SUBJ_ID=${SUBJ_ID}"
} >> "${RUN_MARKER}"

job_fail() {
  local msg="$1"
  {
    echo ""
    echo "========== JOB FAILED $(date -Iseconds) =========="
    echo "${msg}"
    echo "status=FAILED"
  } >> "${RUN_MARKER}"
  printf '%s\n' "${msg}" | tee -a "${JOB_LOG}"
  exit 1
}

mkdir -p "$(dirname "${JOB_LOG}")" || job_fail "cannot create log dir for JOB_LOG=${JOB_LOG}"
[ -n "${SUBJ_ID}" ] || job_fail "SUBJ_ID is empty"
[ -n "${EVC_DIR}" ] || job_fail "EVC_DIR is empty"
# shellcheck source=/dev/null
source "${EVC_DIR}/evc_config.sh" || job_fail "failed to source evc_config.sh"
# shellcheck source=/dev/null
source "${EVC_DIR}/evc_make_evc_roi_functions.sh" || job_fail "failed to source evc_make_evc_roi_functions.sh"

EVC_ROI_FILL_N=$((EVC_ROI_TOTAL_VOXELS - EVC_ROI_TOP_N_PER_CONTRAST))

feat_path="${FSLFEAT_ROOT}/EVC/${SUBJ_ID}/${SUBJ_ID}_${FEAT_BASENAME}.feat"
roi_dir="${EVC_ROIS_ROOT}/${SUBJ_ID}"
anat_v1v2="${roi_dir}/anat/${SUBJ_ID}_V1V2.nii.gz"
func_dir="${roi_dir}/func"
roi_mni="${roi_dir}/${SUBJ_ID}_evc_300_V1V2.nii.gz"

[ -f "$roi_mni" ] && job_fail "MNI ROI already exists: ${roi_mni}"
[ -d "${feat_path}/stats" ] && [ -f "${feat_path}/report.html" ] || job_fail "missing or incomplete FEAT: ${feat_path}"
[ -f "${feat_path}/stats/zstat4.nii.gz" ] || job_fail "missing ${feat_path}/stats/zstat4.nii.gz"
[ -f "${feat_path}/stats/zstat5.nii.gz" ] || job_fail "missing ${feat_path}/stats/zstat5.nii.gz"
[ -f "$anat_v1v2" ] || job_fail "missing T1w V1V2 (run 05_make_anat_masks.sh): ${anat_v1v2}"

func_ref="${feat_path}/mean_func.nii.gz"
if [ ! -f "$func_ref" ]; then
  func_ref="${FMRIPREP_ROOT}/${SUBJ_ID}/${EVC_SESSION}/func/${SUBJ_ID}_${EVC_SESSION}_${EVC_TASK}_space-T1w_desc-preproc_bold.nii.gz"
fi
[ -f "$func_ref" ] || job_fail "no functional reference for resampling: ${func_ref}"

evc_find_mni_ref "$SUBJ_ID" || job_fail "missing MNI boldref (fMRIPrep) for ${SUBJ_ID}"

mkdir -p "$func_dir" "$roi_dir"

v1v2_func="${func_dir}/${SUBJ_ID}_V1V2_on_func.nii.gz"
if ! flirt -in "$anat_v1v2" -ref "$func_ref" -out "$v1v2_func" -applyxfm -usesqform -interp nearestneighbour >>"${JOB_LOG}" 2>&1; then
  job_fail "flirt resample anat→func failed"
fi

masked_tlbr="${func_dir}/_${SUBJ_ID}_evc_zstat_TLBR_vs_TRBL_V1V2_masked.nii.gz"
masked_trbl="${func_dir}/_${SUBJ_ID}_evc_zstat_TRBL_vs_TLBR_V1V2_masked.nii.gz"
fslmaths "${feat_path}/stats/zstat4" -mas "$v1v2_func" "$masked_tlbr" >>"${JOB_LOG}" 2>&1
fslmaths "${feat_path}/stats/zstat5" -mas "$v1v2_func" "$masked_trbl" >>"${JOB_LOG}" 2>&1

roi_func="${func_dir}/${SUBJ_ID}_evc_300_V1V2_on_func.nii.gz"
if ! n_vox=$(evc_build_top_voxel_roi "$masked_tlbr" "$masked_trbl" "$roi_func" \
  "$EVC_ROI_TOP_N_PER_CONTRAST" "$EVC_ROI_TOTAL_VOXELS" 2>&1); then
  job_fail "top-voxel ROI failed: ${n_vox}"
fi

evc_warp_to_mni "$roi_func" "$roi_mni" nearestneighbour "$SUBJ_ID" >>"${JOB_LOG}" 2>&1 \
  || job_fail "warp ROI to MNI failed"

cp -f "${feat_path}/stats/zstat4.nii.gz" "${roi_dir}/${SUBJ_ID}_evc_zstat_TLBR_vs_TRBL.nii.gz"
cp -f "${feat_path}/stats/zstat5.nii.gz" "${roi_dir}/${SUBJ_ID}_evc_zstat_TRBL_vs_TLBR.nii.gz"
for out_label in TLBR_vs_TRBL TRBL_vs_TLBR; do
  z_src="${roi_dir}/${SUBJ_ID}_evc_zstat_${out_label}.nii.gz"
  z_mni="${roi_dir}/${SUBJ_ID}_evc_zstat_${out_label}_MNI152NLin2009cAsym.nii.gz"
  if evc_warp_to_mni "$z_src" "$z_mni" trilinear "$SUBJ_ID" >>"${JOB_LOG}" 2>&1; then
    fslmaths "$z_mni" -mas "$roi_mni" -thr 0 "${roi_dir}/_${SUBJ_ID}_evc_zstat_${out_label}_V1V2_masked" >>"${JOB_LOG}" 2>&1
  fi
done

{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "roi_mni=${roi_mni}"
  echo "n_vox=${n_vox}"
  echo "status=OK"
} >> "${RUN_MARKER}"
echo ">> ${SUBJ_ID}: ${roi_mni} (n_vox=${n_vox})" | tee -a "${JOB_LOG}"
