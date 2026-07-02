#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Slurm 2nd-level FEAT (combined FIR). Run after 05_make_fsf_files_2nd_lvl.sh
# and 04_create_identity_reg_for_2nd_lvl.sh.
#
# Usage: bash 06_run_FIR_2nd_level_VG.sh

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files"
JOB_FILES_ROOT="${FIR_CODE_ROOT}/job_files/run_FIR_2nd_level"
SLURM_TEMPLATE="${FIR_CODE_ROOT}/slurm_templates/FIR_2nd_lvl.sh"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
MAX_RUNS=8

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

prepare_fsf_for_good_runs() {
    local subj="$1"
    local analysis_label="$2"
    local base_fsf="$3"
    local patched_fsf="$4"
    local submit_status="${5:-pending}"

    FSLFEAT_ROOT="$FSLFEAT_ROOT" \
        FSLFEAT_SUBDIR="$FSLFEAT_SUBDIR" \
        FEAT_SUFFIX="$FEAT_SUFFIX" \
        MAX_RUNS="$MAX_RUNS" \
        RUNS_REPORT="$RUNS_REPORT" \
        SUBMIT_STATUS="$submit_status" \
        PATCH_SECOND_LEVEL_FSF_PY="${CODE_PATH}/glm/patch_second_level_fsf.py" \
        python3 - "$subj" "$analysis_label" "$base_fsf" "$patched_fsf" <<'PY'
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
fslfeat_subdir = os.environ["FSLFEAT_SUBDIR"]
feat_suffix = os.environ["FEAT_SUFFIX"]
max_runs = int(os.environ["MAX_RUNS"])

subj_id = f"sub-{subj}"
included = []
excluded = []

for run in range(1, max_runs + 1):
    feat_path = (
        feat_root / fslfeat_subdir / subj_id
        / f"{subj_id}_ses-V2_task-VG_run-{run}_analysis-1stGLM_{feat_suffix}.feat"
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
_spec = importlib.util.spec_from_file_location("patch_second_level_fsf", _mod_path)
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
    } >>"$RUNS_REPORT"
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
    } >>"$RUNS_REPORT"
}

run_variant_2nd_level() {
    local variant="$1"
    if ! fir_set_variant "$variant"; then
        return 1
    fi

    RUNS_REPORT="${JOB_FILES_ROOT}/included_runs_report_${FIR_VARIANT}.txt"

    _sublist_tmp="$(mktemp)"
    if ! fir_list_subjects >"$_sublist_tmp"; then
        rm -f "$_sublist_tmp"
        return 1
    fi
    mapfile -t SUBJECTS < "$_sublist_tmp"
    rm -f "$_sublist_tmp"

    if [ ${#SUBJECTS[@]} -eq 0 ]; then
        echo "ERROR: no subjects for ${variant} (${CSV_FILTER_COL})" >&2
        return 1
    fi

    echo ">> ${variant}: subjects=${#SUBJECTS[@]} (${CSV_FILTER_COL} == TRUE)"

    mkdir -p "$JOB_FILES_ROOT"
    {
        echo "# FIR 2nd-level — 1st-level run inclusion report (${FIR_VARIANT})"
        echo "# date=$(date -Iseconds)"
        echo "# fslfeat_subdir=${FSLFEAT_SUBDIR}"
        echo "# feat_suffix=${FEAT_SUFFIX}"
        echo "# max_runs=${MAX_RUNS}"
        echo ""
    } >"$RUNS_REPORT"

    for subj in "${SUBJECTS[@]}"; do
        subj_id="sub-${subj}"
        fsf_dir="${FSF_FILES_ROOT}/${subj}"
        job_dir="${JOB_FILES_ROOT}/${variant}/${subj}"
        mkdir -p "$job_dir"
        n_submitted=0
        n_skipped=0
        n_sbatch_failed=0
        n_preflight_failed=0

        analysis_suffix="analysis-2ndGLM_${FIR_VARIANT}-MNI152NLin2009cAsym"
        run_marker="${job_dir}/${subj_id}_ses-V2_task-VG_${analysis_suffix}.run"
        slurm_script="${job_dir}/slurm_${subj_id}_2nd_VG_${FIR_VARIANT}.sh"
        base_fsf="${fsf_dir}/${subj}_ses-V2_task-VG_${analysis_suffix}.fsf"
        patched_fsf="${job_dir}/${subj}_ses-V2_task-VG_${analysis_suffix}_included_runs.fsf"
        feat_log="${job_dir}/feat_${subj_id}_2nd_VG_${FIR_VARIANT}.log"
        slurm_out="${job_dir}/slurm_${subj_id}_2nd_VG_${FIR_VARIANT}_%j.out"

        if [ -f "$run_marker" ]; then
            n_skipped=1
            append_run_report_status "$FIR_VARIANT" "$subj" "skipped_marker_exists"
            echo ">> ${subj_id} (${variant}): submitted=0 skipped=1 preflight_failed=0 sbatch_failed=0"
            continue
        fi

        if [ ! -f "$base_fsf" ]; then
            echo "ERROR: ${subj_id} (${variant}): missing FSF: ${base_fsf}" >&2
            n_preflight_failed=1
            append_run_report_status "$FIR_VARIANT" "$subj" "missing_base_fsf"
            echo ">> ${subj_id} (${variant}): submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
            continue
        fi

        _prep_tmp="$(mktemp)"
        if ! prepare_fsf_for_good_runs "$subj" "$FIR_VARIANT" "$base_fsf" "$patched_fsf" "preflight_ok" >"$_prep_tmp" 2>&1; then
            cat "$_prep_tmp" >&2
            rm -f "$_prep_tmp"
            echo "ERROR: ${subj_id} (${variant}): could not build FSF (see ${RUNS_REPORT})" >&2
            n_preflight_failed=1
            echo ">> ${subj_id} (${variant}): submitted=0 skipped=0 preflight_failed=1 sbatch_failed=0"
            continue
        fi
        included_list=$(awk -F= '/^META included_run_ids=/{print $2}' "$_prep_tmp")
        n_included=$(awk -F= '/^META n_included=/{print $2}' "$_prep_tmp")
        fsf_file=$(grep -v '^META ' "$_prep_tmp" | tail -1)
        rm -f "$_prep_tmp"

        echo ">> ${subj_id} (${variant}): using ${n_included} run(s) (${included_list})"

        cat >"$run_marker" <<EOF
# FIR 2nd-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
subj_id=${subj_id}
variant=${FIR_VARIANT}
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
            awk '/^### FIR_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
            printf '#SBATCH --output=%s\n' "$slurm_out"
            printf '#SBATCH --error=%s\n' "$slurm_out"
            printf 'export RUN_MARKER=%q\n' "$run_marker"
            printf 'export FEAT_LOG=%q\n' "$feat_log"
            printf 'export FSF_FILE=%q\n' "$fsf_file"
            awk '/^### FIR_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
        } >"$slurm_script"

        chmod +x "$slurm_script"
        if ! job_out=$(sbatch "$slurm_script" 2>&1); then
            {
                echo ""
                echo "========== sbatch failed $(date -Iseconds) =========="
                echo "${job_out}"
            } >>"$run_marker"
            echo "status=SBATCH_FAILED" >>"$run_marker"
            echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
            n_sbatch_failed=1
            append_submission_status "$FIR_VARIANT" "$subj" "sbatch_failed"
        else
            {
                echo ""
                echo "========== sbatch $(date -Iseconds) =========="
                echo "${job_out}"
            } >>"$run_marker"
            n_submitted=1
            append_submission_status "$FIR_VARIANT" "$subj" "submitted"
        fi

        echo ">> ${subj_id} (${variant}): submitted=${n_submitted} skipped=${n_skipped} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"
    done

    echo ">> run inclusion report (${variant}): ${RUNS_REPORT}"
}

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    run_variant_2nd_level "$variant" || exit 1
done
