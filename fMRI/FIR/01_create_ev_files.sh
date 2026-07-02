#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Build combined Seen/Unseen FIR *_Shifted.txt event files per run.
# Subjects: BASELINE_min_seen_unseen == TRUE (see fir_variants.sh).

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
EV_FILES_ROOT="${BIDS_ROOT}/derivatives/regressoreventfiles"

# HRF onset shift (seconds) for FIR EVs referenced as *_Shifted in the FSF template
FIR_ONSET_SHIFT=4.5

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

if ! fir_set_variant FIR; then
    exit 1
fi

_sublist_tmp="$(mktemp)"
if ! fir_list_subjects >"$_sublist_tmp"; then
    rm -f "$_sublist_tmp"
    exit 1
fi
mapfile -t SUBJECTS < "$_sublist_tmp"
rm -f "$_sublist_tmp"

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: no subjects after filter (${CSV_FILTER_COL})" >&2
    exit 1
fi

echo ">> Number of subjects to use: ${#SUBJECTS[@]} (${CSV_FILTER_COL} == TRUE; list: $SUBJECT_CSV)"

declare -a RUNS=(1 2 3 4 5 6 7 8)

# Placeholder rows (0 0 0) in upstream EVs become negative onsets after FIR_ONSET_SHIFT.
filter_zero_triplets() {
    local f="$1"
    awk 'NF >= 3 && ($1 + 0) == 0 && ($2 + 0) == 0 && ($3 + 0) == 0 { next }
         { print }' "$f" >"${f}.tmp" && mv "${f}.tmp" "$f"
}

shift_ev() {
    local src="$1"
    local dst="$2"
    awk -v s="${FIR_ONSET_SHIFT}" '
        NF >= 3 {
            t = $1 - s
            if (t < 0) next
            printf "%.2f %.2f %.2f\n", t, $2, $3
            next
        }
        { print }
    ' "$src" >"$dst"
}

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    ev_dir="${EV_FILES_ROOT}/${subj_id}/ses-V2"
    EVENTFILES_ROOT="${EV_FILES_ROOT}/${subj_id}/ses-V2/FIR"
    mkdir -p "$EVENTFILES_ROOT"
    n_runs=0

    for i_run in "${RUNS[@]}"; do
        run_base="${subj_id}-ses-V2_task-VG_run-${i_run}"

        unseen_out="${EVENTFILES_ROOT}/${run_base}_probedUnseen.txt"
        seen_out="${EVENTFILES_ROOT}/${run_base}_probedSeen.txt"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedUnseenObjectRight_EV.txt" >"$unseen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedUnseenObjectLeft_EV.txt" >>"$unseen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedUnseenFaceRight_EV.txt" >>"$unseen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedUnseenFaceLeft_EV.txt" >>"$unseen_out"

        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedSeenObjectRight_EV.txt" >"$seen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedSeenObjectLeft_EV.txt" >>"$seen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedSeenFaceRight_EV.txt" >>"$seen_out"
        cat "${ev_dir}/${subj_id}_ses-V2_task-VG_run-${i_run}_probedSeenFaceLeft_EV.txt" >>"$seen_out"

        for evf in "$unseen_out" "$seen_out"; do
            filter_zero_triplets "$evf"
        done

        shift_ev "$seen_out" "${EVENTFILES_ROOT}/${run_base}_probedSeen_Shifted.txt"
        shift_ev "$unseen_out" "${EVENTFILES_ROOT}/${run_base}_probedUnseen_Shifted.txt"

        n_runs=$((n_runs + 1))
    done

    echo ">> ${subj_id}: event_runs=${n_runs}"
done
