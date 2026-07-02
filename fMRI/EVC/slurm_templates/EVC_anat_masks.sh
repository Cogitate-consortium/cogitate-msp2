#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=4:00:00
#SBATCH --mem=16G

### EVC_EXPORT_BLOCK_INSERT ###
# 05_make_anat_masks.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FreeSurfer/6.0.1-centos6_x86_64 2>/dev/null || module load FreeSurfer
module load FSL/6.0.2-foss-2019a-Python-3.7.2 2>/dev/null || module load FSL

LABELS=(V1 V2)

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

fs_subj_dir="${FREESURFER_ROOT}/${SUBJ_ID}"
anat_dir="${EVC_ROIS_ROOT}/${SUBJ_ID}/anat"
done_marker="${anat_dir}/.${SUBJ_ID}_V1V2_T1w.done"
t1w="${BIDS_ROOT}/${SUBJ_ID}/${T1W_ANAT_SESSION}/anat/${SUBJ_ID}_${T1W_ANAT_SESSION}_acq-anat_run-1_T1w.nii.gz"

[ -f "${fs_subj_dir}/scripts/recon-all.done" ] || job_fail "FreeSurfer not finished: ${fs_subj_dir}"
[ -f "$t1w" ] || job_fail "missing T1w: ${t1w}"

mkdir -p "$anat_dir"
export SUBJECTS_DIR="${FREESURFER_ROOT}"

if [ ! -f "${fs_subj_dir}/register.dat" ]; then
  echo ">> ${SUBJ_ID}: bbregister" | tee -a "${JOB_LOG}"
  if ! bbregister --s "${SUBJ_ID}" --mov "$t1w" --reg "${fs_subj_dir}/register.dat" --init-fsl --t1 >>"${JOB_LOG}" 2>&1; then
    job_fail "bbregister failed"
  fi
fi

subj_ok=1
for label in "${LABELS[@]}"; do
  lh_out="${anat_dir}/${SUBJ_ID}_lh_${label}.nii.gz"
  rh_out="${anat_dir}/${SUBJ_ID}_rh_${label}.nii.gz"
  merged="${anat_dir}/${SUBJ_ID}_${label}.nii.gz"

  if [ -f "$merged" ]; then
    continue
  fi

  if [ ! -f "$lh_out" ]; then
    if ! mri_label2vol \
      --label "${fs_subj_dir}/label/lh.${label}_exvivo.label" \
      --temp "${fs_subj_dir}/mri/rawavg.mgz" \
      --subject "${SUBJ_ID}" \
      --hemi lh \
      --o "$lh_out" \
      --proj frac 0 1 .1 \
      --fillthresh 0 \
      --reg "${fs_subj_dir}/register.dat" >>"${JOB_LOG}" 2>&1; then
      job_fail "mri_label2vol failed lh ${label}"
    fi
  fi

  if [ ! -f "$rh_out" ]; then
    if ! mri_label2vol \
      --label "${fs_subj_dir}/label/rh.${label}_exvivo.label" \
      --temp "${fs_subj_dir}/mri/rawavg.mgz" \
      --subject "${SUBJ_ID}" \
      --hemi rh \
      --o "$rh_out" \
      --proj frac 0 1 .1 \
      --fillthresh 0 \
      --reg "${fs_subj_dir}/register.dat" >>"${JOB_LOG}" 2>&1; then
      job_fail "mri_label2vol failed rh ${label}"
    fi
  fi

  fslmaths "$rh_out" -add "$lh_out" -bin "$merged" >>"${JOB_LOG}" 2>&1
done

fslmaths "${anat_dir}/${SUBJ_ID}_V1" -add "${anat_dir}/${SUBJ_ID}_V2" -bin "${anat_dir}/${SUBJ_ID}_V1V2.nii.gz" >>"${JOB_LOG}" 2>&1
date -Iseconds >"$done_marker"

{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "anat_dir=${anat_dir}"
  echo "status=OK"
} >> "${RUN_MARKER}"
echo ">> ${SUBJ_ID}: T1w V1/V2/V1V2 ok → ${anat_dir}" | tee -a "${JOB_LOG}"
