#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=6:00:00
#SBATCH --mem=6G

### GPPI_EXPORT_BLOCK_INSERT ###
# 04_run_PPI_1st_level.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FSL
module load SciPy-bundle/2024.05-gfbf-2024a

{
  echo ""
  echo "========== job start $(date -Iseconds) =========="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
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
[ -f "${FSF_FILE}" ] || preflight_fail "missing FSF file: ${FSF_FILE}"
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
  fi
done < <(grep -E '^(set feat_files\(1\)|set confoundev_files\(1\)|set fmri\(custom[0-9]+\))' "${FSF_FILE}" 2>/dev/null || true)
[[ -z "${missing}" ]] || preflight_fail "referenced paths from FSF are missing:${missing}"

set +e
feat "${FSF_FILE}" 2>&1 | tee -a "${FEAT_LOG}"
exit_feat=$?
set -e
{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "feat_exit_code=${exit_feat}"
  echo "---- last 150 lines of ${FEAT_LOG} (feat stdout/stderr) ----"
  tail -n 150 "${FEAT_LOG}" 2>/dev/null || echo "(could not read FEAT_LOG)"
} >> "${RUN_MARKER}"
if [ "${exit_feat}" -eq 0 ]; then
  echo "status=FEAT_OK" >> "${RUN_MARKER}"
else
  echo "status=FEAT_FAILED" >> "${RUN_MARKER}"
fi
exit "${exit_feat}"
