#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Slurm 2nd-level gPPI FEAT. Run after 06_make_fsf_files_2ndlevel.sh and
# 05_create_identity_reg_for_2nd_lvl.sh.
#
# Builds a per-subject FSF including only successfully completed 1st-level
# runs (report.html OK + identity reg). Writes one combined run-inclusion
# report under job_files/run_PPI_2nd_level/.
#
# Usage: bash 07_run_PPI_2nd_level_VG.sh

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/run_PPI_2nd_level"
SLURM_TEMPLATE="${GPPI_CODE_ROOT}/slurm_templates/PPI_2nd_lvl.sh"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
MAX_RUNS=8
RUNS_REPORT="${JOB_FILES_ROOT}/included_runs_report.txt"

declare -a ANALYSES=(PPI_LOC PPI_FFA)

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

mkdir -p "$JOB_FILES_ROOT"
{
    echo "# gPPI 2nd-level — 1st-level run inclusion report (all subjects)"
    echo "# date=$(date -Iseconds)"
    echo "# max_runs=${MAX_RUNS}"
    echo ""
} > "$RUNS_REPORT"

load_subjects() {
    local _tmp
    _tmp="$(mktemp)"
    SUBJECT_CSV="$SUBJECT_CSV" python3 <<'PY' >"$_tmp"
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
    local _st=$?
    if [ "$_st" -ne 0 ]; then
        rm -f "$_tmp"
        return "$_st"
    fi
    SUBJECTS=()
    mapfile -t SUBJECTS < "$_tmp"
    rm -f "$_tmp"
    return 0
}

# Patch base 2nd-level FSF to include only good 1st-level runs; append to RUNS_REPORT.
# Prints patched FSF path on stdout; exits non-zero on failure.
prepare_fsf_for_good_runs() {
    local subj="$1"
    local analysis="$2"
    local base_fsf="$3"
    local patched_fsf="$4"
    local submit_status="${5:-pending}"
    local feat_suffix="analysis-1stGLM_${analysis}-MNI152NLin2009cAsym"

    FSLFEAT_ROOT="$FSLFEAT_ROOT" FEAT_SUFFIX="$feat_suffix" MAX_RUNS="$MAX_RUNS" \
        RUNS_REPORT="$RUNS_REPORT" SUBMIT_STATUS="$submit_status" \
        PATCH_SECOND_LEVEL_FSF_PY="${CODE_PATH}/glm/patch_second_level_fsf.py" \
        python3 - "$subj" "$analysis" "$base_fsf" "$patched_fsf" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

subj, analysis_label, base_fsf_s, patched_fsf_s = sys.argv[1:5]
base_fsf = Path(base_fsf_s)
patched_fsf = Path(patched_fsf_s)
runs_report = Path(os.environ["RUNS_REPORT"])
submit_status = os.environ.get("SUBMIT_STATUS", "pending")

feat_root = Path(os.environ["FSLFEAT_ROOT"])
feat_suffix = os.environ["FEAT_SUFFIX"]
max_runs = int(os.environ["MAX_RUNS"])

subj_id = f"sub-{subj}"
included = []
excluded = []

for run in range(1, max_runs + 1):
    feat_path = (
        feat_root / "gPPI" / subj_id
        / f"{subj_id}_ses-V2_task-VG_run-{run}_{feat_suffix}.feat"
    )
    reason = None
    if not feat_path.is_dir():
        reason = "missing feat dir"
    elif not (feat_path / "report.html").is_file():
        reason = "missing report.html"
    else:
        report_text = (feat_path / "report.html").read_text(encoding="utf-8", errors="replace")
        if "Error" in report_text or "ERROR" in report_text:
            reason = "error in report.html"
    if reason is None and not (feat_path / "reg" / "example_func2standard.mat").is_file():
        reason = "missing identity reg (example_func2standard.mat)"
    if reason is None:
        included.append((run, str(feat_path)))
    else:
        excluded.append((run, reason, str(feat_path)))

inc_ids = ",".join(str(r) for r, _ in included)
exc_ids = ",".join(str(r) for r, _, _ in excluded)

lines = [
    f"## analysis={analysis_label}\tsubject={subj_id}\tstatus={submit_status}",
    f"included_run_ids={inc_ids}",
    f"excluded_run_ids={exc_ids}",
    f"n_included={len(included)}",
    f"n_excluded={len(excluded)}",
    f"patched_fsf={patched_fsf_s}",
    "",
    "# Included runs",
]
for run, path in included:
    lines.append(f"run-{run}\t{path}")
lines.extend(["", "# Excluded runs"])
for run, reason, path in excluded:
    lines.append(f"run-{run}\t{reason}\t{path}")
lines.append("")

runs_report.parent.mkdir(parents=True, exist_ok=True)
with runs_report.open("a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

if not included:
    print("ERROR: no successful 1st-level runs to include", file=sys.stderr)
    sys.exit(2)

_mod_path = Path(os.environ["PATCH_SECOND_LEVEL_FSF_PY"])
_spec = importlib.util.spec_from_file_location(
    "patch_second_level_fsf", _mod_path
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

paths = [p for _, p in included]
text = base_fsf.read_text(encoding="utf-8")
try:
    text = _mod.apply_run_count_patch(
        text,
        len(included),
        paths,
        insert_marker="# Add confound EVs",
    )
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

patched_fsf.parent.mkdir(parents=True, exist_ok=True)
patched_fsf.write_text(text, encoding="utf-8")
print(f"META included_run_ids={inc_ids}")
print(f"META n_included={len(included)}")
print(str(patched_fsf))
PY
}

append_submission_status() {
    local analysis="$1"
    local subj="$2"
    local status="$3"
    {
        echo "## submission_update analysis=${analysis}\tsubject=sub-${subj}\tsubmission_status=${status}"
        echo ""
    } >> "$RUNS_REPORT"
}

append_run_report_status() {
    local analysis="$1"
    local subj="$2"
    local status="$3"
    {
        echo "## analysis=${analysis}\tsubject=sub-${subj}\tstatus=${status}"
        echo "included_run_ids="
        echo "excluded_run_ids="
        echo "n_included=0"
        echo "n_excluded="
        echo ""
    } >> "$RUNS_REPORT"
}

if ! load_subjects; then
    exit 1
fi

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: no subjects after loading/filtering $SUBJECT_CSV" >&2
    exit 1
fi

echo ">> subjects=${#SUBJECTS[@]} (SYNCHRONY_min_seen == TRUE; list: $SUBJECT_CSV)"

# To test/run just one participant, uncomment and set the line below (use the sub_code, e.g., SUBJECTS=(SC108))
#SUBJECTS=(SC108)

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    fsf_dir="${FSF_FILES_ROOT}/${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    mkdir -p "$job_dir"

    for analysis in "${ANALYSES[@]}"; do
        n_submitted=0
        n_skipped=0
        n_sbatch_failed=0
        n_preflight_failed=0

        analysis_suffix="analysis-2ndGLM_${analysis}-MNI152NLin2009cAsym"
        run_marker="${job_dir}/${subj_id}_ses-V2_task-VG_${analysis_suffix}.run"
        slurm_script="${job_dir}/slurm_${subj_id}_2nd_VG_${analysis}.sh"
        base_fsf="${fsf_dir}/${subj}_ses-V2_task-VG_${analysis_suffix}.fsf"
        patched_fsf="${job_dir}/${subj}_ses-V2_task-VG_${analysis_suffix}_included_runs.fsf"
        feat_log="${job_dir}/feat_${subj_id}_2nd_VG_${analysis}.log"
        slurm_out="${job_dir}/slurm_${subj_id}_2nd_VG_${analysis}_%j.out"

        if [ -f "$run_marker" ]; then
            n_skipped=1
            append_run_report_status "$analysis" "$subj" "skipped_marker_exists"
            echo ">> ${subj_id} ${analysis}: submitted=0 skipped=1 preflight_failed=0 sbatch_failed=0"
            continue
        fi

        if [ ! -f "$base_fsf" ]; then
            echo "ERROR: ${subj_id} ${analysis}: missing FSF file: ${base_fsf} (run 06_make_fsf_files_2ndlevel.sh?)" >&2
            n_preflight_failed=1
            append_run_report_status "$analysis" "$subj" "missing_base_fsf"
            echo ">> ${subj_id} ${analysis}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
            continue
        fi

        _prep_tmp="$(mktemp)"
        if ! prepare_fsf_for_good_runs "$subj" "$analysis" "$base_fsf" "$patched_fsf" "preflight_ok" >"$_prep_tmp" 2>&1; then
            cat "$_prep_tmp" >&2
            rm -f "$_prep_tmp"
            echo "ERROR: ${subj_id} ${analysis}: could not build FSF from successful 1st-level runs (see ${RUNS_REPORT})" >&2
            n_preflight_failed=1
            echo ">> ${subj_id} ${analysis}: submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
            continue
        fi
        included_list=$(awk -F= '/^META included_run_ids=/{print $2}' "$_prep_tmp")
        n_included=$(awk -F= '/^META n_included=/{print $2}' "$_prep_tmp")
        fsf_file=$(grep -v '^META ' "$_prep_tmp" | tail -1)
        rm -f "$_prep_tmp"

        echo ">> ${subj_id} ${analysis}: using ${n_included} run(s) (${included_list})"

        cat > "$run_marker" <<EOF
# gPPI 2nd-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
submit_host=$(hostname)
user=${USER:-unknown}
subj_id=${subj_id}
subj_code=${subj}
analysis=${analysis}
base_fsf=${base_fsf}
fsf_file=${fsf_file}
included_runs=${included_list}
runs_report=${RUNS_REPORT}
feat_log=${feat_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF

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
            n_sbatch_failed=1
            append_submission_status "$analysis" "$subj" "sbatch_failed"
        else
            {
                echo ""
                echo "========== sbatch $(date -Iseconds) =========="
                echo "${job_out}"
            } >> "$run_marker"
            n_submitted=1
            append_submission_status "$analysis" "$subj" "submitted"
        fi

        echo ">> ${subj_id} ${analysis}: submitted=${n_submitted} skipped=${n_skipped} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"
    done
done

echo ">> run inclusion report: ${RUNS_REPORT}"
