#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=2:00:00
#SBATCH --mem=8G

### LOGFILE_EXPORT_BLOCK_INSERT ###
# 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh inserts #SBATCH --output/--error and exports.

set -euo pipefail

module load SciPy-bundle/2024.05-gfbf-2024a

export OPENBLAS_NUM_THREADS=1

{
  echo ""
  echo "========== job start $(date -Iseconds) =========="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SUB_CODE=${SUB_CODE:-}"
  echo "EXTRACTION_SCRIPT=${EXTRACTION_SCRIPT:-}"
  echo "JOB_LOG=${JOB_LOG:-}"
} >> "${RUN_MARKER}"

preflight_fail() {
  local msg="$1"
  {
    echo ""
    echo "========== PREFLIGHT FAILED $(date -Iseconds) =========="
    echo "${msg}"
    echo "status=PREFLIGHT_FAILED"
  } >> "${RUN_MARKER}"
  printf '%s\n' "${msg}" | tee -a "${JOB_LOG}"
  exit 1
}

mkdir -p "$(dirname "${JOB_LOG}")" || preflight_fail "cannot create log directory"
[ -n "${SUB_CODE:-}" ] || preflight_fail "SUB_CODE is empty"
[ -n "${EXTRACTION_SCRIPT:-}" ] || preflight_fail "EXTRACTION_SCRIPT is empty"
[ -f "${EXTRACTION_SCRIPT}" ] || preflight_fail "missing script: ${EXTRACTION_SCRIPT}"

set +e
{
  echo ""
  echo "========== python $(date -Iseconds) =========="
  python3 "${EXTRACTION_SCRIPT}" --sub-code "${SUB_CODE}"
  exit_py=$?
  echo "python_exit_code=${exit_py}"
  exit "${exit_py}"
} 2>&1 | tee -a "${JOB_LOG}"
exit_py=${PIPESTATUS[0]}
set -e

{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "python_exit_code=${exit_py}"
  echo "---- last 40 lines of ${JOB_LOG} ----"
  tail -n 40 "${JOB_LOG}" 2>/dev/null || echo "(could not read JOB_LOG)"
} >> "${RUN_MARKER}"

if [ "${exit_py}" -eq 0 ]; then
  echo "status=LOGFILE_EXTRACTION_OK" >> "${RUN_MARKER}"
else
  echo "status=LOGFILE_EXTRACTION_FAILED" >> "${RUN_MARKER}"
fi
exit "${exit_py}"
