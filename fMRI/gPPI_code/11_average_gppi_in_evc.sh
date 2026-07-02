#!/usr/bin/env bash
# Author: Yamil Vidal
# Mean 2nd-level VG GPPI (FFA seed → cope3, LOC seed → cope4) inside EVC ROI masks.
# Writes one table from inner stats/cope1 and one from stats/zstat1 per subject.
#
# Usage: bash 11_average_gppi_in_evc.sh

# Load FSL
module load FSL

BIDS_ROOT="${BIDS_ROOT:-/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids}"
CODE_PATH="${CODE_PATH:-${BIDS_ROOT}/code}"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
EVC_ROIS_ROOT="${BIDS_ROOT}/derivatives/evc_rois"

analysis="SYNCHRONY_min_seen"
SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri.csv}"
SUBJECT_CSV_COLUMN="${SUBJECT_CSV_COLUMN:-${analysis}}"

# shellcheck source=../EVC/evc_config.sh
source "${GPPI_CODE_ROOT}/../EVC/evc_config.sh"

if ! evc_load_subjects; then
    exit 1
fi

echo "Total number of participants: ${#SUBJECTS[@]} (${EVC_SUBJECT_FILTER_DESC})"

# Resolve 2nd-level stat map (cope3 = ppi_FFA, cope4 = ppi_LOC); try .feat then .gfeat.
# stat_name: cope1 or zstat1 inside copeN.feat/stats/
gppi_2nd_stat() {
    local sub_code="$1"
    local glm_analysis="$2"
    local cope_num="$3"
    local stat_name="$4"
    local subj_id="sub-${sub_code}"
    local stem="${subj_id}_ses-V2_task-VG_analysis-2ndGLM_${glm_analysis}-MNI152NLin2009cAsym"
    local ext nii

    for ext in feat gfeat; do
        nii="${FSLFEAT_ROOT}/gPPI/${subj_id}/${stem}.${ext}/cope${cope_num}.feat/stats/${stat_name}.nii.gz"
        if [ -f "$nii" ]; then
            printf '%s\n' "$nii"
            return 0
        fi
    done
    return 1
}

out_dir="${BIDS_ROOT}/derivatives/gppi/"
mkdir -p "$out_dir"

write_evc_table() {
    local stat_name="$1"
    local stat_label="$2"
    local out_csv="${out_dir}/gppi_evc_roi_means_${stat_label}_${analysis}.csv"
    local n_written=0
    local n_skipped=0

    echo "sub_code;FFA_gppi;LOC_gppi" >"$out_csv"

    for sub_code in "${SUBJECTS[@]}"; do
        local subj_id="sub-${sub_code}"
        local evc_roi="${EVC_ROIS_ROOT}/${subj_id}/${subj_id}_evc_300_V1V2.nii.gz"
        local ffa_nii loc_nii ffa_mean loc_mean

        if [ ! -f "$evc_roi" ]; then
            echo "Skip ${sub_code} (${stat_label}): missing EVC mask: ${evc_roi}" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi

        if ! ffa_nii=$(gppi_2nd_stat "$sub_code" "PPI_FFA" 3 "$stat_name"); then
            echo "Skip ${sub_code} (${stat_label}): missing FFA 2nd-level GPPI (cope3/${stat_name})" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if ! loc_nii=$(gppi_2nd_stat "$sub_code" "PPI_LOC" 4 "$stat_name"); then
            echo "Skip ${sub_code} (${stat_label}): missing LOC 2nd-level GPPI (cope4/${stat_name})" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi

        if ! ffa_mean=$(fslstats "$ffa_nii" -k "$evc_roi" -m 2>/dev/null); then
            echo "Skip ${sub_code} (${stat_label}): fslstats failed for FFA map" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if ! loc_mean=$(fslstats "$loc_nii" -k "$evc_roi" -m 2>/dev/null); then
            echo "Skip ${sub_code} (${stat_label}): fslstats failed for LOC map" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ -z "$ffa_mean" ] || [ -z "$loc_mean" ]; then
            echo "Skip ${sub_code} (${stat_label}): fslstats returned empty mean" >&2
            n_skipped=$((n_skipped + 1))
            continue
        fi

        echo "${sub_code};${ffa_mean};${loc_mean}" >>"$out_csv"
        n_written=$((n_written + 1))
    done

    echo "Wrote ${n_written} rows to ${out_csv} (${stat_label}; skipped ${n_skipped})"
}

write_evc_table "cope1" "cope"
write_evc_table "zstat1" "zstat"
