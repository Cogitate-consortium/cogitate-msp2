#!/bin/bash
#SBATCH --nodes=1

### GPPI_EXPORT_BLOCK_INSERT ###
# 02_make_timecourse_files.sh inserts #SBATCH --output/--error and exports before the body runs.

module load FSL
module load SciPy-bundle/2024.05-gfbf-2024a

export OPENBLAS_NUM_THREADS=1

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
  printf '%s\n' "${msg}" | tee -a "${JOB_LOG}"
  exit 1
}

mkdir -p "$(dirname "${JOB_LOG}")" || preflight_fail "cannot create log directory for JOB_LOG=${JOB_LOG}"
command -v fslmeants >/dev/null 2>&1 || preflight_fail "fslmeants not in PATH (is FSL loaded above?)"
[ -n "${FUNC_DATA}" ] || preflight_fail "FUNC_DATA is empty"
[ -f "${FUNC_DATA}" ] || preflight_fail "missing functional data: ${FUNC_DATA}"
[ -n "${SEED_LOC}" ] || preflight_fail "SEED_LOC is empty"
[ -f "${SEED_LOC}" ] || preflight_fail "missing LOC seed mask: ${SEED_LOC}"
[ -n "${SEED_FFA}" ] || preflight_fail "SEED_FFA is empty"
[ -f "${SEED_FFA}" ] || preflight_fail "missing FFA seed mask: ${SEED_FFA}"
[ -n "${TS_LOC}" ] || preflight_fail "TS_LOC is empty"
[ -n "${TS_FFA}" ] || preflight_fail "TS_FFA is empty"
mkdir -p "$(dirname "${TS_LOC}")" || preflight_fail "cannot create output directory for timecourses"

set +e
{
  echo ""
  echo "========== fslmeants $(date -Iseconds) =========="
  echo "FUNC_DATA=${FUNC_DATA}"
  echo "SEED_LOC=${SEED_LOC}"
  echo "SEED_FFA=${SEED_FFA}"
  echo "TS_LOC=${TS_LOC}"
  echo "TS_FFA=${TS_FFA}"

  exit_ts=0
  if [ -f "${TS_LOC}" ]; then
    echo ">> TS_LOC already exists, skipping"
  else
    fslmeants -i "${FUNC_DATA}" -o "${TS_LOC}" -m "${SEED_LOC}"
    st=$?
    echo "fslmeants LOC exit_code=${st}"
    [ "${st}" -eq 0 ] || exit_ts=${st}
  fi

  if [ -f "${TS_FFA}" ]; then
    echo ">> TS_FFA already exists, skipping"
  else
    fslmeants -i "${FUNC_DATA}" -o "${TS_FFA}" -m "${SEED_FFA}"
    st=$?
    echo "fslmeants FFA exit_code=${st}"
    [ "${st}" -eq 0 ] || exit_ts=${st}
  fi
  exit "${exit_ts}"
} 2>&1 | tee -a "${JOB_LOG}"
exit_ts=${PIPESTATUS[0]}
set -e

{
  echo ""
  echo "========== job finished $(date -Iseconds) =========="
  echo "timecourse_exit_code=${exit_ts}"
  echo "---- last 80 lines of ${JOB_LOG} ----"
  tail -n 80 "${JOB_LOG}" 2>/dev/null || echo "(could not read JOB_LOG)"
} >> "${RUN_MARKER}"

if [ "${exit_ts}" -eq 0 ]; then
  echo "status=TIMECOURSE_OK" >> "${RUN_MARKER}"
else
  echo "status=TIMECOURSE_FAILED" >> "${RUN_MARKER}"
fi
exit "${exit_ts}"
