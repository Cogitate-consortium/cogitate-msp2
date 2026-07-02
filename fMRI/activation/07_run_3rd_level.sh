#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Submit Slurm 3rd-level activation FEAT jobs. Run after 06_make_fsf_files_3rdlevel.sh
# (inputtype 1; feat_files -> subject copeN.feat).
# Submits every analysis that has a valid FSF; warns and skips contrasts with no FSF
# (e.g. zero subjects in 06). Writes submission_inclusion_report.txt (subjects per
# contrast, from 06 included_subjects_report.txt / FSF). Writes a trace-back inclusion
# report under activation/3rd level report/ (see make_3rd_level_inclusion_report.sh).
# Exits 1 only on sbatch failure or if nothing was submitted.
#
# Usage: bash 07_run_3rd_level.sh

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
ACTIVATION_CODE_ROOT="${CODE_PATH}/activation"
FSF_FILES_ROOT="${ACTIVATION_CODE_ROOT}/fsf_files/group"
JOB_FILES_ROOT="${ACTIVATION_CODE_ROOT}/job_files/run_activation_3rd_level"
SLURM_TEMPLATE="${ACTIVATION_CODE_ROOT}/slurm_templates/ACTIVATION_3rd_lvl.sh"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
SUBMISSION_REPORT="${JOB_FILES_ROOT}/submission_inclusion_report.txt"
REPORT_3RD_PY="${CODE_PATH}/glm/report_3rd_level_submission.py"
MAKE_INCLUSION_REPORT="${ACTIVATION_CODE_ROOT}/make_3rd_level_inclusion_report.sh"

declare -a ANALYSES=(
    "seen_face_vs_unseen_face:analysis-3rdGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym:1"
    "seen_object_vs_unseen_object:analysis-3rdGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym:2"
)

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: Slurm template not found: $SLURM_TEMPLATE" >&2
    exit 1
fi

if [ ! -f "$REPORT_3RD_PY" ]; then
    echo "ERROR: report helper not found: $REPORT_3RD_PY" >&2
    exit 1
fi

mkdir -p "$JOB_FILES_ROOT"

python3 "$REPORT_3RD_PY" init --report "$SUBMISSION_REPORT" \
    --title "Activation 3rd-level — subjects included at submission"

# query_3rd_subjects ANALYSIS COPE_TAG FSF_FILE — META lines only (no report write)
query_3rd_subjects() {
    python3 "$REPORT_3RD_PY" query \
        --analysis "$1" \
        --cope "$2" \
        --fsf "$3" \
        --inclusion-report "$INCLUSION_REPORT"
}

# record_3rd_submission ANALYSIS COPE_TAG FSF_FILE STATUS
record_3rd_submission() {
    python3 "$REPORT_3RD_PY" record \
        --report "$SUBMISSION_REPORT" \
        --pipeline activation \
        --analysis "$1" \
        --cope "$2" \
        --fsf "$3" \
        --inclusion-report "$INCLUSION_REPORT" \
        --status "$4"
}

n_submitted=0
n_skipped=0
n_missing_fsf=0
n_sbatch_failed=0

for analysis_spec in "${ANALYSES[@]}"; do
    IFS=':' read -r analysis_label third_suffix cope_idx <<< "$analysis_spec"
    cope_tag="cope${cope_idx}"

    fsf_glob="${FSF_FILES_ROOT}/N*_ses-V2_task-VG_${third_suffix}_desc-${cope_tag}.fsf"
    shopt -s nullglob
    fsf_matches=( $fsf_glob )
    shopt -u nullglob

    if [ ${#fsf_matches[@]} -eq 0 ]; then
        echo "WARN: no FSF for ${analysis_label}; skipping submission (0 subjects in 06 or run 06_make_fsf_files_3rdlevel.sh?)" >&2
        echo "      expected: ${fsf_glob}" >&2
        record_3rd_submission "$analysis_label" "$cope_tag" "" "no_fsf" >/dev/null
        n_missing_fsf=$((n_missing_fsf + 1))
        continue
    fi
    if [ ${#fsf_matches[@]} -gt 1 ]; then
        echo "WARN: multiple FSFs match ${analysis_label}; skipping submission (re-run 05 to refresh):" >&2
        printf '      %s\n' "${fsf_matches[@]}" >&2
        record_3rd_submission "$analysis_label" "$cope_tag" "" "multiple_fsf" >/dev/null
        n_missing_fsf=$((n_missing_fsf + 1))
        continue
    fi

    fsf_file="${fsf_matches[0]}"
    fsf_base="$(basename "$fsf_file" .fsf)"
    job_dir="${JOB_FILES_ROOT}/${analysis_label}"
    mkdir -p "$job_dir"

    run_marker="${job_dir}/${fsf_base}.run"
    slurm_script="${job_dir}/slurm_${analysis_label}_3rd.sh"
    feat_log="${job_dir}/feat_${analysis_label}_3rd.log"
    slurm_out="${job_dir}/slurm_${analysis_label}_3rd_%j.out"

    if [ -f "$run_marker" ]; then
        echo ">> ${analysis_label}: skipped (marker exists)"
        record_3rd_submission "$analysis_label" "$cope_tag" "$fsf_file" "skipped_marker" >/dev/null
        n_skipped=$((n_skipped + 1))
        continue
    fi

    _meta_tmp="$(mktemp)"
    query_3rd_subjects "$analysis_label" "$cope_tag" "$fsf_file" >"$_meta_tmp"
    n_included=$(awk -F= '/^META n_included=/{print $2}' "$_meta_tmp")
    included_list=$(awk -F= '/^META included_subject_ids=/{print $2}' "$_meta_tmp")
    rm -f "$_meta_tmp"

    echo ">> ${analysis_label}: ${n_included} subject(s) in FSF (${included_list})"

    cat > "$run_marker" <<EOF
# Activation 3rd-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
analysis=${analysis_label}
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
        awk '/^### ACTIVATION_3RD_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
        printf '#SBATCH --output=%s\n' "$slurm_out"
        printf '#SBATCH --error=%s\n' "$slurm_out"
        printf 'export RUN_MARKER=%q\n' "$run_marker"
        printf 'export FEAT_LOG=%q\n' "$feat_log"
        printf 'export FSF_FILE=%q\n' "$fsf_file"
        awk '/^### ACTIVATION_3RD_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
    } > "$slurm_script"

    chmod +x "$slurm_script"
    if ! job_out=$(sbatch "$slurm_script" 2>&1); then
        {
            echo ""
            echo "========== sbatch failed $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        echo "status=SBATCH_FAILED" >> "$run_marker"
        record_3rd_submission "$analysis_label" "$cope_tag" "$fsf_file" "sbatch_failed" >/dev/null
        echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
        n_sbatch_failed=$((n_sbatch_failed + 1))
    else
        {
            echo ""
            echo "========== sbatch $(date -Iseconds) =========="
            echo "${job_out}"
        } >> "$run_marker"
        n_submitted=$((n_submitted + 1))
        record_3rd_submission "$analysis_label" "$cope_tag" "$fsf_file" "submitted" >/dev/null
        echo ">> ${analysis_label}: submitted (${fsf_base})"
    fi
done

echo ">> 3rd-level summary: submitted=${n_submitted} skipped=${n_skipped} missing_fsf=${n_missing_fsf} sbatch_failed=${n_sbatch_failed}"
echo ">> submission inclusion report: ${SUBMISSION_REPORT}"
if [ -f "$INCLUSION_REPORT" ]; then
    echo ">> 05 inclusion report (source): ${INCLUSION_REPORT}"
fi

if [ -f "$MAKE_INCLUSION_REPORT" ]; then
    if bash "$MAKE_INCLUSION_REPORT"; then
        echo ">> 3rd-level inclusion report: ${ACTIVATION_CODE_ROOT}/3rd level report/3rd_level_inclusion_report.txt"
    else
        echo "WARN: failed to write 3rd-level inclusion report (see make_3rd_level_inclusion_report.sh)" >&2
    fi
else
    echo "WARN: missing ${MAKE_INCLUSION_REPORT}; skipping 3rd-level inclusion report" >&2
fi

if [ "$n_missing_fsf" -gt 0 ]; then
    echo "WARN: ${n_missing_fsf} analysis/analyses had no usable FSF and were not submitted (see messages above)" >&2
fi

if [ "$n_sbatch_failed" -gt 0 ]; then
    exit 1
fi

if [ "$n_submitted" -eq 0 ] && [ "$n_skipped" -eq 0 ]; then
    echo "ERROR: no 3rd-level jobs submitted (all analyses missing FSF or invalid FSF glob)" >&2
    exit 1
fi
