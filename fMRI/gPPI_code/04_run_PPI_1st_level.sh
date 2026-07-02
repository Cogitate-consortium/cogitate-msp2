#!/usr/bin/env bash


#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
#"""


#module load FSL

echo "$FSLDIR"
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Paths (nested: BIDS → fMRI_exp2 → gPPI_code → outputs / job files)
# ---------------------------------------------------------------------------
BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/run_PPI_1st_level"
SLURM_TEMPLATE="${GPPI_CODE_ROOT}/slurm_templates/PPI_1st_lvl.sh"

# Use the same subject list/filtering as in 02_make_timecourse_files.sh and 03_make_fsf_files.sh
if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
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

declare -a RUNS=(1 2 3 4 5 6 7 8)
declare -a ANALYSES=(PPI_LOC PPI_FFA)

for subj in "${SUBJECTS[@]}" ; do
	subj_id="sub-${subj}"
	fsf_dir="${FSF_FILES_ROOT}/${subj}"
	job_dir="${JOB_FILES_ROOT}/${subj}"
	mkdir -p "$job_dir"
	n_submitted=0
	n_skipped=0
	n_sbatch_failed=0

	for analysis in "${ANALYSES[@]}" ; do
	for i_run in "${RUNS[@]}" ; do
		run_marker="${job_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.run"
		slurm_script="${job_dir}/slurm_${subj_id}_run-${i_run}_VG_${analysis}.sh"
		fsf_file="${fsf_dir}/${subj}_ses-V2_task-VG_run-${i_run}_analysis-1stGLM_${analysis}-MNI152NLin2009cAsym.fsf"
		feat_log="${job_dir}/feat_${subj_id}_run-${i_run}_VG_${analysis}.log"
		slurm_out="${job_dir}/slurm_${subj_id}_run-${i_run}_VG_${analysis}_%j.out"

    	if [ ! -f "$run_marker" ] ; then

			cat > "$run_marker" <<EOF
# gPPI 1st-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
submit_host=$(hostname)
user=${USER:-unknown}
subj_id=${subj_id}
subj_code=${subj}
analysis=${analysis}
run=${i_run}
fsf_file=${fsf_file}
feat_log=${feat_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF

			# Job body (preflight, feat, .run footer) lives in slurm_templates/PPI_1st_lvl.sh after ### GPPI_EXPORT_BLOCK_INSERT ###
			{
				awk '/^### GPPI_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
				printf '#SBATCH --output=%s\n' "$slurm_out"
				printf '#SBATCH --error=%s\n' "$slurm_out"
				printf 'export RUN_MARKER=%q\n' "$run_marker"
				printf 'export FEAT_LOG=%q\n' "$feat_log"
				printf 'export FSF_FILE=%q\n' "$fsf_file"
				awk '/^### GPPI_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
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
				n_submitted=$((n_submitted + 1))
			fi

	    else
	          n_skipped=$((n_skipped + 1))
	    fi
	done
	done
	echo ">> ${subj_id}: submitted=${n_submitted} skipped=${n_skipped} sbatch_failed=${n_sbatch_failed}"
done
