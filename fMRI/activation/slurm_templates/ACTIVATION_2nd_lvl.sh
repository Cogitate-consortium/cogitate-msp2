#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=8:00:00
#SBATCH --mem=8G

### ACTIVATION_EXPORT_BLOCK_INSERT ###
# 04_run_2nd_level.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FSL
module load SciPy-bundle/2024.05-gfbf-2024a

export OPENBLAS_NUM_THREADS=1

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
[ -f "${FSF_FILE}" ] || preflight_fail "missing FSF file: ${FSF_FILE} (run 03_make_fsf_files.sh?)"
[ -r "${FSF_FILE}" ] || preflight_fail "unreadable FSF file: ${FSF_FILE}"
command -v feat >/dev/null 2>&1 || preflight_fail "feat not in PATH (is FSL loaded above?)"

missing=""
while IFS= read -r line || [ -n "${line}" ]; do
  [[ "${line}" == *\"*\"* ]] || continue
  t="${line#*\"}"
  path="${t%%\"*}"
  [[ -n "${path}" ]] || continue
  if [[ ! -e "${path}" ]]; then
    missing="${missing}${missing:+$'\n'}  missing: ${path}"
    continue
  fi
  if [[ "${path}" == *.feat && ! -f "${path}/reg/example_func2standard.mat" ]]; then
    missing="${missing}${missing:+$'\n'}  missing identity reg: ${path}/reg/example_func2standard.mat (run 02_create_identity_reg_for_2nd_lvl.sh?)"
  fi
done < <(grep -E '^set feat_files\(' "${FSF_FILE}" 2>/dev/null || true)
[[ -z "${missing}" ]] || preflight_fail "referenced paths from FSF are missing:${missing}"

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
