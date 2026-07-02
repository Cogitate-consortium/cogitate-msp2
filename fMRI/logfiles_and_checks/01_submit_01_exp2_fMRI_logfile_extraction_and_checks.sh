#!/usr/bin/env bash

# Submit one Slurm job per subject for exp.2 logfile extraction and checks.
# Each job runs exp2_fMRI_logfile_extraction_and_checks.py --sub-code SUB.
#
# Usage: bash 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh [--force]

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids}"
CODE_PATH="${CODE_PATH:-${BIDS_ROOT}/code}"
SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri.csv}"
LOGFILES_ROOT="${CODE_PATH}/logfiles_and_checks"
EXTRACTION_SCRIPT="${LOGFILES_ROOT}/exp2_fMRI_logfile_extraction_and_checks.py"
JOB_FILES_ROOT="${LOGFILES_ROOT}/job_files/exp2_logfile_extraction"
SLURM_TEMPLATE="${LOGFILES_ROOT}/slurm_templates/exp2_logfile_extraction.sh"

FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help)
            echo "Usage: bash 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh [--force]"
            echo "  --force  submit even if run marker exists"
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$EXTRACTION_SCRIPT" ]; then
    echo "ERROR: extraction script not found: $EXTRACTION_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

_sublist_tmp="$(mktemp)"
LOGFILES_ROOT="$LOGFILES_ROOT" SUBJECT_CSV="$SUBJECT_CSV" python3 <<'PY' >"$_sublist_tmp"
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

logfiles_root = os.environ["LOGFILES_ROOT"]
script = os.path.join(logfiles_root, "exp2_fMRI_logfile_extraction_and_checks.py")
spec = spec_from_file_location("logext", script)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
for sub in mod.load_subjects():
    print(sub)
PY
_py_st=$?
if [ "$_py_st" -ne 0 ]; then
    rm -f "$_sublist_tmp"
    exit "$_py_st"
fi
mapfile -t SUBJECTS < "$_sublist_tmp"
rm -f "$_sublist_tmp"

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: no subjects after loading/filtering $SUBJECT_CSV" >&2
    exit 1
fi

echo ">> subjects=${#SUBJECTS[@]} csv=${SUBJECT_CSV}"
echo ">> job_files=${JOB_FILES_ROOT}"

n_submitted=0
n_skipped=0
n_sbatch_failed=0

for subj in "${SUBJECTS[@]}"; do
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"

    run_marker="${job_dir}/${subj}_logfile_extraction.run"
    slurm_script="${job_dir}/slurm_${subj}_logfile_extraction.sh"
    job_log="${job_dir}/logfile_extraction_${subj}.log"
    slurm_out="${job_dir}/slurm_${subj}_logfile_extraction_%j.out"

    if [ -f "$run_marker" ] && [ "$FORCE" -eq 0 ]; then
        echo ">> ${subj}: skipped (marker exists: ${run_marker})"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    cat > "$run_marker" <<EOF
# exp.2 logfile extraction — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
sub_code=${subj}
extraction_script=${EXTRACTION_SCRIPT}
job_log=${job_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF

    {
        awk '/^### LOGFILE_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
        printf '#SBATCH --output=%s\n' "$slurm_out"
        printf '#SBATCH --error=%s\n' "$slurm_out"
        printf 'export RUN_MARKER=%q\n' "$run_marker"
        printf 'export JOB_LOG=%q\n' "$job_log"
        printf 'export SUB_CODE=%q\n' "$subj"
        printf 'export EXTRACTION_SCRIPT=%q\n' "$EXTRACTION_SCRIPT"
        printf 'export BIDS_ROOT=%q\n' "$BIDS_ROOT"
        printf 'export CODE_PATH=%q\n' "$CODE_PATH"
        printf 'export SUBJECT_CSV=%q\n' "$SUBJECT_CSV"
        awk '/^### LOGFILE_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
    } > "$slurm_script"

    chmod +x "$slurm_script"

    if ! job_out=$(sbatch "$slurm_script" 2>&1); then
        {
            echo ""
            echo "========== sbatch failed $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        echo "status=SBATCH_FAILED" >> "$run_marker"
        echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
        n_sbatch_failed=$((n_sbatch_failed + 1))
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        echo ">> ${subj}: ${job_out}"
        n_submitted=$((n_submitted + 1))
    fi
done

echo ">> summary: submitted=${n_submitted} skipped=${n_skipped} sbatch_failed=${n_sbatch_failed}"
[ "$n_sbatch_failed" -eq 0 ]
