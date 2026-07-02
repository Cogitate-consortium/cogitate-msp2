#!/usr/bin/env bash

# Submit one Slurm job per analysis folder under fslFeat (activation, FIR, gPPI, …).
# Each job runs glm/check_FSLFeat_outputs.py on that folder.
#
# Usage:
#   bash submit_check_FSLFeat.sh [--level all|1st|2nd]
#
#   --level all  (default)  scan every .feat/.gfeat under each analysis folder
#   --level 1st             only run-level first-level (_run-N_ + 1stGLM / 1stROI in name)
#   --level 2nd             only outputs whose basename includes analysis-2ndGLM
#
# Env FEAT_CHECK_LEVEL is used as default when --level is omitted (same values).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash submit_check_FSLFeat.sh [--level all|1st|2nd]

  --level   Restrict check_FSLFeat_outputs.py to 1st- or 2nd-level FEAT dirs only.
            Default: all. Same default from env FEAT_CHECK_LEVEL if set.

Examples:
  bash submit_check_FSLFeat.sh --level 1st
  bash submit_check_FSLFeat.sh --level 2nd
  FEAT_CHECK_LEVEL=2nd bash submit_check_FSLFeat.sh
EOF
}

FEAT_LEVEL="${FEAT_CHECK_LEVEL:-all}"
while [ $# -gt 0 ]; do
    case "$1" in
        --level)
            FEAT_LEVEL="${2:?--level requires all, 1st, or 2nd}"
            shift 2
            ;;
        --level=*)
            FEAT_LEVEL="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${FEAT_LEVEL}" in
    all|1st|2nd) ;;
    *)
        echo "ERROR: --level / FEAT_CHECK_LEVEL must be all, 1st, or 2nd (got: ${FEAT_LEVEL})" >&2
        exit 1
        ;;
esac

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
GLM_CODE_ROOT="${CODE_PATH}/glm"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
CHECK_SCRIPT="${GLM_CODE_ROOT}/check_FSLFeat_outputs.py"
SLURM_TEMPLATE="${GLM_CODE_ROOT}/slurm_templates/check_FSLFeat.sh"
if [ "${FEAT_LEVEL}" = "all" ]; then
    JOB_FILES_ROOT="${GLM_CODE_ROOT}/job_files/check_fslfeat"
else
    JOB_FILES_ROOT="${GLM_CODE_ROOT}/job_files/check_fslfeat_${FEAT_LEVEL}"
fi

if [ ! -f "${CHECK_SCRIPT}" ]; then
    echo "ERROR: check script not found: ${CHECK_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${SLURM_TEMPLATE}" ]; then
    echo "ERROR: Slurm template not found: ${SLURM_TEMPLATE}" >&2
    exit 1
fi

if [ ! -d "${FSLFEAT_ROOT}" ]; then
    echo "ERROR: fslFeat root not found: ${FSLFEAT_ROOT}" >&2
    exit 1
fi

ANALYSIS_DIRS=()
for _d in "${FSLFEAT_ROOT}"/*/; do
    [ -d "${_d}" ] || continue
    _name="$(basename "${_d}")"
    [[ "${_name}" == sub-* ]] && continue
    ANALYSIS_DIRS+=("${_name}")
done
IFS=$'\n' ANALYSIS_DIRS=($(printf '%s\n' "${ANALYSIS_DIRS[@]}" | sort))
unset IFS

if [ ${#ANALYSIS_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no analysis folders found under ${FSLFEAT_ROOT}" >&2
    echo "       (expected e.g. activation/, FIR/, gPPI/ — not sub-*)" >&2
    exit 1
fi

echo ">> fslFeat root: ${FSLFEAT_ROOT}"
echo ">> FEAT level: ${FEAT_LEVEL}"
echo ">> job files: ${JOB_FILES_ROOT}"
echo ">> analysis folders: ${ANALYSIS_DIRS[*]}"

n_submitted=0
n_skipped=0
n_sbatch_failed=0

for analysis in "${ANALYSIS_DIRS[@]}"; do
    derivatives_subdir="${FSLFEAT_ROOT}/${analysis}"
    job_dir="${JOB_FILES_ROOT}/${analysis}"
    mkdir -p "${job_dir}"

    run_marker="${job_dir}/check_${analysis}.run"
    slurm_script="${job_dir}/slurm_check_${analysis}.sh"
    check_log="${job_dir}/check_${analysis}.log"
    output_file="${job_dir}/feat_failed_outputs_${analysis}_${FEAT_LEVEL}_$(date +%Y%m%d_%H%M%S).txt"
    slurm_out="${job_dir}/slurm_check_${analysis}_%j.out"

    if [ -f "${run_marker}" ]; then
        echo ">> ${analysis}: skipped (marker exists: ${run_marker})"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    cat > "${run_marker}" <<EOF
# FEAT output check — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
feat_level=${FEAT_LEVEL}
analysis=${analysis}
derivatives_subdir=${derivatives_subdir}
check_script=${CHECK_SCRIPT}
check_log=${check_log}
output_file=${output_file}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF

    {
        awk '/^### FEAT_CHECK_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "${SLURM_TEMPLATE}"
        printf '#SBATCH --output=%s\n' "${slurm_out}"
        printf '#SBATCH --error=%s\n' "${slurm_out}"
        printf 'export RUN_MARKER=%q\n' "${run_marker}"
        printf 'export ANALYSIS=%q\n' "${analysis}"
        printf 'export DERIVATIVES_SUBDIR=%q\n' "${derivatives_subdir}"
        printf 'export CHECK_SCRIPT=%q\n' "${CHECK_SCRIPT}"
        printf 'export CHECK_LOG=%q\n' "${check_log}"
        printf 'export OUTPUT_FILE=%q\n' "${output_file}"
        printf 'export FEAT_LEVEL=%q\n' "${FEAT_LEVEL}"
        awk '/^### FEAT_CHECK_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "${SLURM_TEMPLATE}"
    } > "${slurm_script}"

    chmod +x "${slurm_script}"

    if ! job_out=$(sbatch "${slurm_script}" 2>&1); then
        {
            echo ""
            echo "========== sbatch failed $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "${run_marker}"
        echo "status=SBATCH_FAILED" >> "${run_marker}"
        echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
        n_sbatch_failed=$((n_sbatch_failed + 1))
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "${run_marker}"
        echo ">> ${analysis}: ${job_out}"
        n_submitted=$((n_submitted + 1))
    fi
done

echo ">> summary: level=${FEAT_LEVEL} submitted=${n_submitted} skipped=${n_skipped} sbatch_failed=${n_sbatch_failed}"
