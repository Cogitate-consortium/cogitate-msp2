#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=octopus
#SBATCH --time=2:00:00
#SBATCH --mem=4G

### FEAT_CHECK_EXPORT_BLOCK_INSERT ###
# submit_check_FSLFeat.sh inserts #SBATCH --output/--error and exports before the body runs.

set -euo pipefail

{
  echo ""
  echo "========== job start $(date -Iseconds) =========="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "ANALYSIS=${ANALYSIS:-}"
  echo "DERIVATIVES_SUBDIR=${DERIVATIVES_SUBDIR:-}"
  echo "CHECK_SCRIPT=${CHECK_SCRIPT:-}"
  echo "OUTPUT_FILE=${OUTPUT_FILE:-}"
  echo "FEAT_LEVEL=${FEAT_LEVEL:-all}"
} >> "${RUN_MARKER}"

preflight_fail() {
  local msg="$1"
  {
    echo ""
    echo "========== PREFLIGHT FAILED $(date -Iseconds) =========="
    echo "${msg}"
    echo "status=PREFLIGHT_FAILED"
  } >> "${RUN_MARKER}"
  printf '%s\n' "${msg}" >&2
  exit 1
}

[ -n "${CHECK_SCRIPT:-}" ] || preflight_fail "CHECK_SCRIPT is empty"
[ -f "${CHECK_SCRIPT}" ] || preflight_fail "missing check script: ${CHECK_SCRIPT}"
[ -n "${DERIVATIVES_SUBDIR:-}" ] || preflight_fail "DERIVATIVES_SUBDIR is empty"
[ -d "${DERIVATIVES_SUBDIR}" ] || preflight_fail "derivatives subfolder not found: ${DERIVATIVES_SUBDIR}"
[ -n "${OUTPUT_FILE:-}" ] || preflight_fail "OUTPUT_FILE is empty"

mkdir -p "$(dirname "${OUTPUT_FILE}")" || preflight_fail "cannot create output directory for ${OUTPUT_FILE}"

set +e
python3 "${CHECK_SCRIPT}" "${DERIVATIVES_SUBDIR}" --level "${FEAT_LEVEL:-all}" -o "${OUTPUT_FILE}" >> "${CHECK_LOG}" 2>&1
exit_check=$?
set -e

{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "check_exit_code=${exit_check}"
  echo "output_file=${OUTPUT_FILE}"
  tail -n 80 "${CHECK_LOG}" 2>/dev/null || true
} >> "${RUN_MARKER}"

if [ "${exit_check}" -eq 0 ]; then
  echo "status=CHECK_OK" >> "${RUN_MARKER}"
else
  echo "status=CHECK_FAILED" >> "${RUN_MARKER}"
fi
exit "${exit_check}"
