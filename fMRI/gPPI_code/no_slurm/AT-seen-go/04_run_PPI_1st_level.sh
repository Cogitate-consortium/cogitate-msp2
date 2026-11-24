#!/usr/bin/env bash


#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Local (no Slurm) 1st-level gPPI FEAT. Uses job_templates/PPI_1st_lvl.sh.
#"""

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"
# shellcheck source=local_parallel.sh
source "${_SCRIPT_DIR}/local_parallel.sh"
gppi_local_init || exit 1

echo "$FSLDIR"

# ---------------------------------------------------------------------------
# Paths (this no_slurm copy; FEAT outputs still under BIDS derivatives)
# ---------------------------------------------------------------------------
GPPI_CODE_ROOT="${_SCRIPT_DIR}"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/run_PPI_1st_level"
JOB_TEMPLATE="${GPPI_CODE_ROOT}/job_templates/PPI_1st_lvl.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$JOB_TEMPLATE" ]; then
    echo "ERROR: job template not found: $JOB_TEMPLATE" >&2
    exit 1
fi

_sublist_tmp="$(mktemp)"
SUBJECT_CSV="$SUBJECT_CSV" python3 <<'PY' >"$_sublist_tmp"
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

col = "SYNCHRONY_min_seen"
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
    sys.exit(1)
subj_df = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE")]

for sub in subj_df["sub_code"].values:
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

echo ">> Number of subjects to use: ${#SUBJECTS[@]} (SYNCHRONY_min_seen == TRUE; list: $SUBJECT_CSV)"

# To test/run just one participant, uncomment and set the line below (use the sub_code, e.g., SUBJECTS=(SC108))
#SUBJECTS=(SC108)

declare -a RUNS=(1 2 3 4)
declare -a ANALYSES=(PPI_LOC PPI_FFA)

mkdir -p "$JOB_FILES_ROOT"
job_list="${JOB_FILES_ROOT}/_queued_jobs.txt"
: > "$job_list"
n_queued=0
n_skipped=0

for subj in "${SUBJECTS[@]}" ; do
	subj_id="sub-${subj}"
	fsf_dir="${FSF_FILES_ROOT}/${subj}"
	job_dir="${JOB_FILES_ROOT}/${subj}"
	mkdir -p "$job_dir"
	n_subj_queued=0
	n_subj_skipped=0

	for analysis in "${ANALYSES[@]}" ; do
	for i_run in "${RUNS[@]}" ; do
		run_marker="${job_dir}/${subj_id}_ses-V2_task-Replay_run-${i_run}_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.run"
		job_script="${job_dir}/local_${subj_id}_run-${i_run}_Replay_${analysis}.sh"
		fsf_file="${fsf_dir}/${subj}_ses-V2_task-Replay_run-${i_run}_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.fsf"
		feat_log="${job_dir}/feat_${subj_id}_run-${i_run}_Replay_${analysis}.log"

    	if [ -f "$run_marker" ] ; then
			n_subj_skipped=$((n_subj_skipped + 1))
			n_skipped=$((n_skipped + 1))
			continue
		fi

			cat > "$run_marker" <<EOF
# gPPI 1st-level FEAT — marker / local job summary
status=QUEUED
date_queued=$(date -Iseconds)
submit_host=$(hostname)
user=${USER:-unknown}
subj_id=${subj_id}
subj_code=${subj}
analysis=${analysis}
run=${i_run}
fsf_file=${fsf_file}
feat_log=${feat_log}
job_script=${job_script}
---
EOF

			gppi_assemble_job "$job_script" "$JOB_TEMPLATE" "### GPPI_EXPORT_BLOCK_INSERT ###" \
				"$(printf 'export RUN_MARKER=%q' "$run_marker")" \
				"$(printf 'export FEAT_LOG=%q' "$feat_log")" \
				"$(printf 'export FSF_FILE=%q' "$fsf_file")"
			printf '%s\n' "$job_script" >> "$job_list"
			n_subj_queued=$((n_subj_queued + 1))
			n_queued=$((n_queued + 1))
	done
	done
	echo ">> ${subj_id}: queued=${n_subj_queued} skipped=${n_subj_skipped}"
done

echo ">> 1st-level queued=${n_queued} skipped=${n_skipped} GPPI_NJOBS=${GPPI_NJOBS}"
n_run_failed=0
if ! gppi_run_job_list "$job_list"; then
    n_run_failed=1
fi
echo ">> 1st-level local run finished (run_failed=${n_run_failed})"
if [ "$n_run_failed" -ne 0 ]; then
    echo "ERROR: one or more 1st-level FEAT jobs failed (see job_files/run_PPI_1st_level/)" >&2
    exit 1
fi
