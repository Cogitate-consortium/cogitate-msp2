#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=3:00:00
#SBATCH --mem=8G

### FIR_3RD_EXPORT_BLOCK_INSERT ###
# 09_run_FIR_3rd_level.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FSL
module load SciPy-bundle/2024.05-gfbf-2024a

export OPENBLAS_NUM_THREADS=1
export FSLPARALLEL=0
export FSLOUTPUTTYPE=NIFTI_GZ
export FSLMULTIFILEQUIT=TRUE

{
  echo ""
  echo "========== job start $(date -Iseconds) =========="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
} >> "${RUN_MARKER}"

preflight_fail() {
  local msg="$1"
  {
    echo ""
    echo "========== PREFLIGHT FAILED $(date -Iseconds) =========="
    echo "${msg}"
    echo "status=PREFLIGHT_FAILED"
  } >> "${RUN_MARKER}"
  printf '%s\n' "${msg}" | tee -a "${FEAT_LOG}"
  exit 1
}

mkdir -p "$(dirname "${FEAT_LOG}")" || preflight_fail "cannot create log directory for FEAT_LOG=${FEAT_LOG}"
[ -n "${FSF_FILE}" ] || preflight_fail "FSF_FILE is empty"
[ -f "${FSF_FILE}" ] || preflight_fail "missing FSF file: ${FSF_FILE} (run 08_make_fsf_files_3rdlevel.sh?)"
[ -r "${FSF_FILE}" ] || preflight_fail "unreadable FSF file: ${FSF_FILE}"
command -v feat >/dev/null 2>&1 || preflight_fail "feat not in PATH (is FSL loaded above?)"

_parse_feat_files_path() {
  local line="$1"
  local path=""
  if [[ "$line" =~ ^set[[:space:]]+feat_files\([0-9]+\)[[:space:]]+\{(.+)\}[[:space:]]*$ ]]; then
    path="${BASH_REMATCH[1]}"
  elif [[ "$line" == *\"*\"* ]]; then
    local t="${line#*\"}"
    path="${t%%\"*}"
  fi
  printf '%s' "$path"
}

missing=""
while IFS= read -r line || [ -n "${line}" ]; do
  path="$(_parse_feat_files_path "$line")"
  [[ -n "${path}" ]] || continue
  if [[ ! -e "${path}" ]]; then
    missing="${missing}${missing:+$'\n'}  missing: ${path}"
  fi
done < <(grep -E '^set feat_files\(' "${FSF_FILE}" 2>/dev/null || true)
[[ -z "${missing}" ]] || preflight_fail "referenced paths from FSF are missing:${missing}"

missing_cope_stats=""
while IFS= read -r line || [ -n "${line}" ]; do
  path="$(_parse_feat_files_path "$line")"
  [[ -n "${path}" ]] || continue
  if [[ "${path}" =~ /cope([0-9]+)\.feat$ ]]; then
    cope_nii="${path}/stats/cope1.nii.gz"
    if [[ ! -f "${cope_nii}" ]]; then
      cope_nii="${path}/stats/cope${BASH_REMATCH[1]}.nii.gz"
    fi
    if [[ ! -f "${cope_nii}" ]]; then
      missing_cope_stats="${missing_cope_stats}${missing_cope_stats:+$'\n'}  missing stats in cope feat: ${path}/stats/cope1.nii.gz"
    fi
  fi
done < <(grep -E '^set feat_files\(' "${FSF_FILE}" 2>/dev/null || true)
[[ -z "${missing_cope_stats}" ]] || preflight_fail "cope .feat inputs lack required stats:${missing_cope_stats}"

missing_gfeat_reg=""
while IFS= read -r line || [ -n "${line}" ]; do
  path="$(_parse_feat_files_path "$line")"
  [[ -n "${path}" ]] || continue
  if [[ "${path}" =~ /cope[0-9]+\.feat$ ]]; then
    parent_gfeat="${path%/cope*.feat}"
    if [[ ! -f "${parent_gfeat}/reg/example_func2standard.mat" ]]; then
      missing_gfeat_reg="${missing_gfeat_reg}${missing_gfeat_reg:+$'\n'}  missing 2nd-level identity reg: ${parent_gfeat}/reg/example_func2standard.mat (run 07_create_identity_reg_for_3rd_lvl.sh?)"
    fi
  fi
done < <(grep -E '^set feat_files\(' "${FSF_FILE}" 2>/dev/null || true)
[[ -z "${missing_gfeat_reg}" ]] || preflight_fail "parent .gfeat lacks identity reg:${missing_gfeat_reg}"

set +e
feat "${FSF_FILE}" 2>&1 | tee -a "${FEAT_LOG}"
exit_feat=$?
set -e
{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "feat_exit_code=${exit_feat}"
  tail -n 150 "${FEAT_LOG}" 2>/dev/null || true
} >> "${RUN_MARKER}"
if [ "${exit_feat}" -eq 0 ]; then
  echo "status=FEAT_OK" >> "${RUN_MARKER}"
else
  echo "status=FEAT_FAILED" >> "${RUN_MARKER}"
fi
exit "${exit_feat}"
