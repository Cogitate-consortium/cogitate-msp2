#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Submit Slurm 3rd-level FIR FEAT jobs. Run after 07_create_identity_reg_for_3rd_lvl.sh
# and 08_make_fsf_files_3rdlevel.sh (inputtype 1; feat_files -> copeN.feat).
# Submits every variant/cope that has a valid FSF; warns and skips missing FSFs.
#
# Usage: bash 09_run_FIR_3rd_level.sh

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files/group"
JOB_FILES_ROOT="${FIR_CODE_ROOT}/job_files/run_FIR_3rd_level"
SLURM_TEMPLATE="${FIR_CODE_ROOT}/slurm_templates/FIR_3rd_lvl.sh"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
SUBMISSION_REPORT="${JOB_FILES_ROOT}/submission_inclusion_report.txt"
REPORT_3RD_PY="${CODE_PATH}/glm/report_3rd_level_submission.py"
MAKE_INCLUSION_REPORT="${FIR_CODE_ROOT}/make_fir_3rd_level_inclusion_report.sh"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

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
    --title "FIR 3rd-level — subjects included at submission"

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
        --pipeline fir \
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

for variant in "${FIR_ALL_VARIANTS[@]}"; do
    for cope_idx in "${FIR_3RD_COPES[@]}"; do
        cope_tag="cope${cope_idx}"
        third_suffix="analysis-3rdGLM_${variant}-MNI152NLin2009cAsym"

        fsf_glob="${FSF_FILES_ROOT}/N*_ses-V2_task-VG_${third_suffix}_desc-${cope_tag}.fsf"
        shopt -s nullglob
        fsf_matches=( $fsf_glob )
        shopt -u nullglob

        job_label="${variant}_${cope_tag}"

        if [ ${#fsf_matches[@]} -eq 0 ]; then
            echo "WARN: no FSF for ${variant} ${cope_tag}; skipping (0 subjects in 08?)" >&2
            echo "      expected: ${fsf_glob}" >&2
            record_3rd_submission "$variant" "$cope_tag" "" "no_fsf" >/dev/null
            n_missing_fsf=$((n_missing_fsf + 1))
            continue
        fi
        if [ ${#fsf_matches[@]} -gt 1 ]; then
            echo "WARN: multiple FSFs for ${variant} ${cope_tag}; skipping (re-run 07):" >&2
            printf '      %s\n' "${fsf_matches[@]}" >&2
            record_3rd_submission "$variant" "$cope_tag" "" "multiple_fsf" >/dev/null
            n_missing_fsf=$((n_missing_fsf + 1))
            continue
        fi

        fsf_file="${fsf_matches[0]}"
        fsf_base="$(basename "$fsf_file" .fsf)"
        job_dir="${JOB_FILES_ROOT}/${variant}/${cope_tag}"
        mkdir -p "$job_dir"

        run_marker="${job_dir}/${fsf_base}.run"
        slurm_script="${job_dir}/slurm_${job_label}_3rd.sh"
        feat_log="${job_dir}/feat_${job_label}_3rd.log"
        slurm_out="${job_dir}/slurm_${job_label}_3rd_%j.out"

        if [ -f "$run_marker" ]; then
            echo ">> ${variant} ${cope_tag}: skipped (marker exists)"
            record_3rd_submission "$variant" "$cope_tag" "$fsf_file" "skipped_marker" >/dev/null
            n_skipped=$((n_skipped + 1))
            continue
        fi

        _meta_tmp="$(mktemp)"
        query_3rd_subjects "$variant" "$cope_tag" "$fsf_file" >"$_meta_tmp"
        n_included=$(awk -F= '/^META n_included=/{print $2}' "$_meta_tmp")
        included_list=$(awk -F= '/^META included_subject_ids=/{print $2}' "$_meta_tmp")
        rm -f "$_meta_tmp"

        echo ">> ${variant} ${cope_tag}: ${n_included} subject(s) (${included_list})"

        cat >"$run_marker" <<EOF
# FIR 3rd-level FEAT — marker / submission + job summary
status=SUBMITTED
date_submitted=$(date -Iseconds)
variant=${variant}
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
            awk '/^### FIR_3RD_EXPORT_BLOCK_INSERT ###$/ {exit} {print}' "$SLURM_TEMPLATE"
            printf '#SBATCH --output=%s\n' "$slurm_out"
            printf '#SBATCH --error=%s\n' "$slurm_out"
            printf 'export RUN_MARKER=%q\n' "$run_marker"
            printf 'export FEAT_LOG=%q\n' "$feat_log"
            printf 'export FSF_FILE=%q\n' "$fsf_file"
            awk '/^### FIR_3RD_EXPORT_BLOCK_INSERT ###$/ {f=1; next} f' "$SLURM_TEMPLATE"
        } >"$slurm_script"

        chmod +x "$slurm_script"
        if ! job_out=$(sbatch "$slurm_script" 2>&1); then
            {
                echo ""
                echo "========== sbatch failed $(date -Iseconds) =========="
                echo "${job_out}"
            } >>"$run_marker"
            echo "status=SBATCH_FAILED" >>"$run_marker"
            record_3rd_submission "$variant" "$cope_tag" "$fsf_file" "sbatch_failed" >/dev/null
            echo "ERROR: sbatch failed for ${slurm_script}: ${job_out}" >&2
            n_sbatch_failed=$((n_sbatch_failed + 1))
        else
            {
                echo ""
                echo "========== sbatch $(date -Iseconds) =========="
                echo "${job_out}"
            } >>"$run_marker"
            n_submitted=$((n_submitted + 1))
            record_3rd_submission "$variant" "$cope_tag" "$fsf_file" "submitted" >/dev/null
            echo ">> ${variant} ${cope_tag}: submitted (${fsf_base})"
        fi
    done
done

echo ">> 3rd-level summary: submitted=${n_submitted} skipped=${n_skipped} missing_fsf=${n_missing_fsf} sbatch_failed=${n_sbatch_failed}"
echo ">> submission inclusion report: ${SUBMISSION_REPORT}"
if [ -f "$INCLUSION_REPORT" ]; then
    echo ">> 08 inclusion report (source): ${INCLUSION_REPORT}"
fi

if [ -f "$MAKE_INCLUSION_REPORT" ]; then
    if bash "$MAKE_INCLUSION_REPORT"; then
        echo ">> 3rd-level inclusion report: ${FIR_CODE_ROOT}/3rd level report/fir_3rd_level_inclusion_report.txt"
    else
        echo "WARN: failed to write 3rd-level inclusion report (see make_fir_3rd_level_inclusion_report.sh)" >&2
    fi
else
    echo "WARN: missing ${MAKE_INCLUSION_REPORT}; skipping 3rd-level inclusion report" >&2
fi

if [ "$n_missing_fsf" -gt 0 ]; then
    echo "WARN: ${n_missing_fsf} variant/cope job(s) had no usable FSF (see messages above)" >&2
fi

if [ "$n_sbatch_failed" -gt 0 ]; then
    exit 1
fi

if [ "$n_submitted" -eq 0 ] && [ "$n_skipped" -eq 0 ]; then
    echo "ERROR: no 3rd-level jobs submitted (all missing FSF or invalid FSF glob)" >&2
    exit 1
fi
