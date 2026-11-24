#!/usr/bin/env bash
# Local (no Slurm) job helper for gPPI_code/no_slurm/AT-seen-go.
# Source from 02 / 04 / 07 / 10 after demo_paths.sh.
#
# GPPI_NJOBS  — concurrent jobs (default 8). Override in the environment.

gppi_local_init() {
    export GPPI_NJOBS="${GPPI_NJOBS:-8}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export FSLPARALLEL="${FSLPARALLEL:-0}"
    export FSLOUTPUTTYPE="${FSLOUTPUTTYPE:-NIFTI_GZ}"
    export FSLMULTIFILEQUIT="${FSLMULTIFILEQUIT:-TRUE}"
    if ! [[ "${GPPI_NJOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GPPI_NJOBS must be a positive integer (got: ${GPPI_NJOBS})" >&2
        return 1
    fi
}

# Assemble a job script from a job_templates/* file (### GPPI_*_INSERT ### marker).
# Extra export lines are written between the header and the body.
gppi_assemble_job() {
    local dest="$1"
    local template="$2"
    local insert_marker="$3"
    shift 3
    if [ ! -f "$template" ]; then
        echo "ERROR: job template not found: $template" >&2
        return 1
    fi
    {
        awk -v m="$insert_marker" '$0 == m {exit} {print}' "$template"
        printf '%s\n' "$@"
        awk -v m="$insert_marker" '$0 == m {f=1; next} f' "$template"
    } > "$dest"
    chmod +x "$dest"
}

# Run one bash job script per line in LIST_FILE, up to GPPI_NJOBS at a time.
# Returns xargs status (GNU xargs: 123 if any command failed).
gppi_run_job_list() {
    local list="$1"
    local n=0
    if [ -f "$list" ]; then
        n="$(grep -cve '^[[:space:]]*$' "$list" || true)"
    fi
    if [ "${n:-0}" -eq 0 ]; then
        echo ">> no queued local jobs"
        return 0
    fi
    echo ">> running ${n} local job(s) with GPPI_NJOBS=${GPPI_NJOBS}"
    xargs -d '\n' -n 1 -P "$GPPI_NJOBS" bash < "$list"
}
