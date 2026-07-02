#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Submit Slurm 3rd-level gPPI FEAT jobs. Run after 08_create_identity_reg_for_3rd_lvl.sh
# and 09_make_fsf_files_3rdlevel.sh (inputtype 1; feat_files -> subject copeN.feat).
# Re-run 08 and 09_make after template or path changes before submitting.
# Runs inspect_2nd_level_for_3rd.py strict checks on each FSF before sbatch (same gates
# as inspect_2nd_level_for_3rd.sh; re-run 09_make if subjects fail). Refreshes cohort
# inspect report under gPPI_code/3rd level report/ at start.
# Submits every analysis/cope that has a valid FSF; warns
# and skips missing FSFs.
# Writes submission_inclusion_report.txt (subjects per analysis/cope, from 09_make
# included_subjects_report.txt / FSF). Writes a trace-back inclusion report under
# gPPI_code/3rd level report/ (see make_3rd_level_inclusion_report.sh).
# Exits 1 only on sbatch failure or if nothing was submitted.
#
# Usage: bash 10_run_PPI_3rd_level.sh

export OPENBLAS_NUM_THREADS=1

module load FSL 2>/dev/null || true

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files/group"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/run_PPI_3rd_level"
SLURM_TEMPLATE="${GPPI_CODE_ROOT}/slurm_templates/PPI_3rd_lvl.sh"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
SUBMISSION_REPORT="${JOB_FILES_ROOT}/submission_inclusion_report.txt"
REPORT_3RD_PY="${CODE_PATH}/glm/report_3rd_level_submission.py"
MAKE_INCLUSION_REPORT="${GPPI_CODE_ROOT}/make_3rd_level_inclusion_report.sh"
INSPECT_PY="${GPPI_CODE_ROOT}/inspect_2nd_level_for_3rd.py"
REPORT_3RD_DIR="${GPPI_CODE_ROOT}/3rd level report"

export PYTHONPATH="${GPPI_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck source=gppi_3rd_level.sh
source "${GPPI_CODE_ROOT}/gppi_3rd_level.sh"
ANALYSES=("${GPPI_3RD_ANALYSES[@]}")

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

if [ ! -f "$REPORT_3RD_PY" ]; then
    echo "ERROR: report helper not found: $REPORT_3RD_PY" >&2
    exit 1
fi

if [ ! -f "$INSPECT_PY" ]; then
    echo "ERROR: inspect helper not found: $INSPECT_PY" >&2
    exit 1
fi

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

mkdir -p "$JOB_FILES_ROOT" "$REPORT_3RD_DIR"

echo ">> 2nd-level strict preflight: refresh cohort inspect report"
python3 "$INSPECT_PY" scan \
    --bids-root "$BIDS_ROOT" \
    --subject-csv "$SUBJECT_CSV" \
    --feat-root "$FSLFEAT_ROOT" \
    --out-dir "$REPORT_3RD_DIR"

python3 "$REPORT_3RD_PY" init --report "$SUBMISSION_REPORT" \
    --title "gPPI 3rd-level — subjects included at submission"

query_3rd_subjects() {
    python3 "$REPORT_3RD_PY" query \
        --analysis "$1" \
        --cope "$2" \
        --fsf "$3" \
        --inclusion-report "$INCLUSION_REPORT"
}

record_3rd_submission() {
    python3 "$REPORT_3RD_PY" record \
        --report "$SUBMISSION_REPORT" \
        --pipeline gPPI \
        --analysis "$1" \
        --cope "$2" \
        --fsf "$3" \
        --inclusion-report "$INCLUSION_REPORT" \
        --status "$4"
}

n_submitted=0
n_skipped=0
n_missing_fsf=0
n_preflight_failed=0
n_sbatch_failed=0

for analysis in "${ANALYSES[@]}"; do
    cope_idx="$(gppi_3rd_cope_for "$analysis")" || exit 1
    cope_tag="cope${cope_idx}"
    third_suffix="analysis-3rdGLM_${analysis}-MNI152NLin2009cAsym"

    fsf_glob="${FSF_FILES_ROOT}/N*_ses-V2_task-VG_${third_suffix}_desc-${cope_tag}.fsf"
    shopt -s nullglob
    fsf_matches=( $fsf_glob )
    shopt -u nullglob

    job_label="${analysis}_${cope_tag}"

    if [ ${#fsf_matches[@]} -eq 0 ]; then
        echo "WARN: no FSF for ${analysis} ${cope_tag}; skipping (0 subjects in 08?)" >&2
        echo "      expected: ${fsf_glob}" >&2
        record_3rd_submission "$analysis" "$cope_tag" "" "no_fsf" >/dev/null
        n_missing_fsf=$((n_missing_fsf + 1))
        continue
    fi
    if [ ${#fsf_matches[@]} -gt 1 ]; then
        echo "WARN: multiple FSFs for ${analysis} ${cope_tag}; skipping (re-run 08):" >&2
        printf '      %s\n' "${fsf_matches[@]}" >&2
        record_3rd_submission "$analysis" "$cope_tag" "" "multiple_fsf" >/dev/null
        n_missing_fsf=$((n_missing_fsf + 1))
        continue
    fi

    fsf_file="${fsf_matches[0]}"
    fsf_base="$(basename "$fsf_file" .fsf)"
    job_dir="${JOB_FILES_ROOT}/${analysis}/${cope_tag}"
    mkdir -p "$job_dir"

    run_marker="${job_dir}/${fsf_base}.run"
    slurm_script="${job_dir}/slurm_${job_label}_3rd.sh"
    feat_log="${job_dir}/feat_${job_label}_3rd.log"
    slurm_out="${job_dir}/slurm_${job_label}_3rd_%j.out"

    if [ -f "$run_marker" ]; then
        echo ">> ${analysis} ${cope_tag}: skipped (marker exists)"
        record_3rd_submission "$analysis" "$cope_tag" "$fsf_file" "skipped_marker" >/dev/null
        n_skipped=$((n_skipped + 1))
        continue
    fi

    _meta_tmp="$(mktemp)"
    query_3rd_subjects "$analysis" "$cope_tag" "$fsf_file" >"$_meta_tmp"
    n_included=$(awk -F= '/^META n_included=/{print $2}' "$_meta_tmp")
    included_list=$(awk -F= '/^META included_subject_ids=/{print $2}' "$_meta_tmp")
    rm -f "$_meta_tmp"

    echo ">> ${analysis} ${cope_tag}: ${n_included} subject(s) (${included_list})"

    if ! python3 "$INSPECT_PY" validate-fsf \
        --fsf "$fsf_file" \
        --analysis "$analysis"; then
        echo "ERROR: ${analysis} ${cope_tag}: strict 2nd-level preflight failed; not submitting" >&2
        echo "       fix 2nd-level outputs and re-run 09_make_fsf_files_3rdlevel.sh" >&2
        echo "       see ${REPORT_3RD_DIR}/inspect_2nd_level_report.txt" >&2
        record_3rd_submission "$analysis" "$cope_tag" "$fsf_file" "preflight_strict_failed" >/dev/null
        n_preflight_failed=$((n_preflight_failed + 1))
        continue
    fi

    cat > "$run_marker" <<EOF
# gPPI 3rd-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
analysis=${analysis}
cope=${cope_tag}
fsf_file=${fsf_file}
n_included=${n_included}
included_subject_ids=${included_list}
inclusion_report=${INCLUSION_REPORT}
submission_report=${SUBMISSION_REPORT}
feat_log=${feat_log}
slurm_script=${slurm_script}
slurm_out=${slurm_out}
---
EOF

    {
        awk '/^### GPPI_3RD_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
        printf '#SBATCH --output=%s\n' "$slurm_out"
        printf '#SBATCH --error=%s\n' "$slurm_out"
        printf 'export RUN_MARKER=%q\n' "$run_marker"
        printf 'export FEAT_LOG=%q\n' "$feat_log"
        printf 'export FSF_FILE=%q\n' "$fsf_file"
        awk '/^### GPPI_3RD_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
    } > "$slurm_script"

    chmod +x "$slurm_script"
    if ! job_out=$(sbatch "$slurm_script" 2>&1); then
        {
            echo ""
            echo "========== sbatch failed $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        echo "status=SBATCH_FAILED" >> "$run_marker"
        record_3rd_submission "$analysis" "$cope_tag" "$fsf_file" "sbatch_failed" >/dev/null
        echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
        n_sbatch_failed=$((n_sbatch_failed + 1))
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        n_submitted=$((n_submitted + 1))
        record_3rd_submission "$analysis" "$cope_tag" "$fsf_file" "submitted" >/dev/null
        echo ">> ${analysis} ${cope_tag}: submitted (${fsf_base})"
    fi
done

echo ">> 3rd-level summary: submitted=${n_submitted} skipped=${n_skipped} missing_fsf=${n_missing_fsf} preflight_failed=${n_preflight_failed} sbatch_failed=${n_sbatch_failed}"
echo ">> 2nd-level inspect report: ${REPORT_3RD_DIR}/inspect_2nd_level_report.txt"
echo ">> submission inclusion report: ${SUBMISSION_REPORT}"
if [ -f "$INCLUSION_REPORT" ]; then
    echo ">> 08 inclusion report (source): ${INCLUSION_REPORT}"
fi

if [ -f "$MAKE_INCLUSION_REPORT" ]; then
    if bash "$MAKE_INCLUSION_REPORT"; then
        echo ">> 3rd-level inclusion report: ${GPPI_CODE_ROOT}/3rd level report/3rd_level_inclusion_report.txt"
    else
        echo "WARN: failed to write 3rd-level inclusion report (see make_3rd_level_inclusion_report.sh)" >&2
    fi
else
    echo "WARN: missing ${MAKE_INCLUSION_REPORT}; skipping 3rd-level inclusion report" >&2
fi

if [ "$n_missing_fsf" -gt 0 ]; then
    echo "WARN: ${n_missing_fsf} analysis/cope job(s) had no usable FSF (see messages above)" >&2
fi

if [ "$n_preflight_failed" -gt 0 ]; then
    echo "ERROR: ${n_preflight_failed} job(s) failed strict 2nd-level preflight (re-run 09_make after fixes)" >&2
fi

if [ "$n_preflight_failed" -gt 0 ] || [ "$n_sbatch_failed" -gt 0 ]; then
    exit 1
fi

if [ "$n_submitted" -eq 0 ] && [ "$n_skipped" -eq 0 ]; then
    echo "ERROR: no 3rd-level jobs submitted (all missing FSF or invalid FSF glob)" >&2
    exit 1
fi
