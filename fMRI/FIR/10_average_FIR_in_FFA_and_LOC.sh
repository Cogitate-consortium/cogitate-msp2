#!/usr/bin/env bash
# Author: Yamil Vidal
# Mean 2nd-level FIR COPE within FFA / LOC per time bin; seen vs unseen.
# Seen: cope 1–14; unseen: cope 15–28 (within sub-level gfeat).
# Combined FIR (Seen/Unseen) → FFA and LOC ROI bin averages.
# Writes summary CSVs plus fir_roi_averaging_inclusion_report.txt (per-table inclusion
# and trace-back for exclusions: seed, 2nd level, anatomical, 1st level, etc.).
# Run after 06_run_FIR_2nd_level_VG.sh (needs 2nd-level .gfeat; 3rd level not required).
#
# Usage: bash 10_average_FIR_in_FFA_and_LOC.sh

module load FSL

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
SEEDS_ROOT="${BIDS_ROOT}/derivatives/gppi_seeds"
FIR_DERIV_ROOT="${BIDS_ROOT}/derivatives/fir"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
REPORT_DIR="${FIR_CODE_ROOT}/inclusion report"
REPORT_PY="${FIR_CODE_ROOT}/report_fir_roi_inclusion.py"
INCLUSION_REPORT_NAME="fir_roi_averaging_inclusion_report.txt"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

FFA_rois_template="${SEEDS_ROOT}/sub-SUBCODE/sub-SUBCODE_rel_irrel_bh_face_FFA_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"
LOC_rois_template="${SEEDS_ROOT}/sub-SUBCODE/sub-SUBCODE_rel_irrel_bh_object_LOC_n_voxels_300_space-MNI152NLin2009cAsym.nii.gz"

out_dir="${FIR_DERIV_ROOT}"
error_log="${out_dir}/fir_averaging_error_log.txt"
inclusion_report="${out_dir}/${INCLUSION_REPORT_NAME}"
seen_range=(1 14)
unseen_range=(15 28)

build_header_seen() {
  local h="sub_code"
  local i
  for i in $(seq "${seen_range[@]}"); do
    h+=";Bin_${i}"
  done
  echo "$h"
}

build_header_unseen() {
  local h="sub_code"
  local i n_unseen=$(( unseen_range[1] - unseen_range[0] + 1 ))
  for i in $(seq 1 "$n_unseen"); do
    h+=";Bin_${i}"
  done
  echo "$h"
}

log_skip() {
  echo "$*" | tee -a "$error_log" >&2
}

build_fir_row() {
  local sub=$1 mask=$2 tpl=$3 start=$4 end=$5 label=$6
  local row="${sub}"
  local c m path base="${tpl//SUBCODE/$sub}"

  if [ ! -f "$mask" ]; then
    log_skip "Skip ${sub} (pass ${PASS_LABEL}) — ${label}: missing ROI mask: ${mask}"
    return 1
  fi

  for ((c = start; c <= end; c++)); do
    path="${base//COPECODE/cope${c}}"
    if [ ! -f "$path" ]; then
      log_skip "Skip ${sub} (pass ${PASS_LABEL}) — ${label}: missing FIR map cope${c}: ${path}"
      return 1
    fi
    if ! m=$(fslstats "$path" -k "$mask" -m 2>/dev/null); then
      log_skip "Skip ${sub} (pass ${PASS_LABEL}) — ${label}: fslstats failed cope${c}: ${path}"
      return 1
    fi
    if [ -z "$m" ]; then
      log_skip "Skip ${sub} (pass ${PASS_LABEL}) — ${label}: empty mean cope${c}: ${path}"
      return 1
    fi
    row+=";${m}"
  done
  echo "$row"
}

load_participants() {
  local col="$1"
  SUBJECT_CSV="$SUBJECT_CSV" ANALYSIS_COL="$col" python3 <<'PY'
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
col = os.environ["ANALYSIS_COL"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")
if col not in subj_df.columns:
    sys.exit(1)
for sub in subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE"), "sub_code"]:
    print(sub)
PY
}

run_averaging_pass() {
  local pass_label="$1"
  local fslfeat_subdir="$2"
  local glm_stem="$3"
  local csv_col="$4"
  local out_suffix="$5"
  local do_ffa="$6"
  local do_loc="$7"

  PASS_LABEL="$pass_label"
  local participants
  participants=$(load_participants "$csv_col")
  if [ -z "$participants" ]; then
    echo "WARN: no participants for pass ${pass_label} (${csv_col})" >&2
    return 0
  fi

  local n_list
  n_list=$(printf '%s\n' "$participants" | awk 'NF { c++ } END { print c + 0 }')
  echo ">> Pass ${pass_label}: ${n_list} participants (${csv_col}); fslFeat/${fslfeat_subdir}"

  local fir_tpl="${FSLFEAT_ROOT}/${fslfeat_subdir}/sub-SUBCODE/sub-SUBCODE_ses-V2_task-VG_${glm_stem}.gfeat/COPECODE.feat/stats/cope1.nii.gz"

  local out_seen_ffa="" out_unseen_ffa="" out_seen_loc="" out_unseen_loc=""
  if [ "$do_ffa" = "1" ]; then
    out_seen_ffa="${out_dir}/fir_FFA_seen_bins_${out_suffix}.csv"
    out_unseen_ffa="${out_dir}/fir_FFA_unseen_bins_${out_suffix}.csv"
    build_header_seen >"$out_seen_ffa"
    build_header_unseen >"$out_unseen_ffa"
  fi
  if [ "$do_loc" = "1" ]; then
    out_seen_loc="${out_dir}/fir_LOC_seen_bins_${out_suffix}.csv"
    out_unseen_loc="${out_dir}/fir_LOC_unseen_bins_${out_suffix}.csv"
    build_header_seen >"$out_seen_loc"
    build_header_unseen >"$out_unseen_loc"
  fi

  for sub_code in $participants; do
    local ffa_mask loc_mask
    ffa_mask="${FFA_rois_template//SUBCODE/$sub_code}"
    loc_mask="${LOC_rois_template//SUBCODE/$sub_code}"

    if [ "$do_ffa" = "1" ]; then
      local row_sf row_uf
      if row_sf=$(build_fir_row "$sub_code" "$ffa_mask" "$fir_tpl" "${seen_range[0]}" "${seen_range[1]}" "FFA_seen"); then
        if row_uf=$(build_fir_row "$sub_code" "$ffa_mask" "$fir_tpl" "${unseen_range[0]}" "${unseen_range[1]}" "FFA_unseen"); then
          echo "$row_sf" >>"$out_seen_ffa"
          echo "$row_uf" >>"$out_unseen_ffa"
        fi
      fi
    fi

    if [ "$do_loc" = "1" ]; then
      local row_sl row_ul
      if row_sl=$(build_fir_row "$sub_code" "$loc_mask" "$fir_tpl" "${seen_range[0]}" "${seen_range[1]}" "LOC_seen"); then
        if row_ul=$(build_fir_row "$sub_code" "$loc_mask" "$fir_tpl" "${unseen_range[0]}" "${unseen_range[1]}" "LOC_unseen"); then
          echo "$row_sl" >>"$out_seen_loc"
          echo "$row_ul" >>"$out_unseen_loc"
        fi
      fi
    fi
  done

  for f in "$out_seen_ffa" "$out_unseen_ffa" "$out_seen_loc" "$out_unseen_loc"; do
    [ -n "$f" ] || continue
    local n=$(( $(wc -l <"$f") - 1 ))
    echo "Wrote ${n} data rows to ${f}"
  done
}

mkdir -p "$out_dir"
{
  echo "fir_averaging error log"
  echo "Started: $(date -Iseconds 2>/dev/null || date)"
  echo "participants_list=${SUBJECT_CSV}"
  echo "---"
} >"$error_log"

run_averaging_pass \
  "combined" \
  "FIR" \
  "analysis-2ndGLM_FIR-MNI152NLin2009cAsym" \
  "${FIR_SUBJECT_FILTER_COL}" \
  "BASELINE_min_seen_unseen" \
  1 1

echo "Error log (excluded rows only): ${error_log}"

if [ -f "$REPORT_PY" ]; then
    mkdir -p "$REPORT_DIR"
    if BIDS_ROOT="$BIDS_ROOT" \
        FIR_CODE_ROOT="$FIR_CODE_ROOT" \
        SUBJECT_CSV="$SUBJECT_CSV" \
        REPORT_DIR="$REPORT_DIR" \
        FIR_DERIV_ROOT="$out_dir" \
        ERROR_LOG="$error_log" \
        python3 "$REPORT_PY"; then
        echo ">> inclusion report: ${inclusion_report}"
        echo ">> inclusion report (copy): ${REPORT_DIR}/${INCLUSION_REPORT_NAME}"
    else
        echo "WARN: failed to write FIR inclusion report (${REPORT_PY})" >&2
    fi
else
    echo "WARN: missing ${REPORT_PY}; skipping inclusion report" >&2
fi
