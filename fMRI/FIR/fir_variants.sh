#!/usr/bin/env bash
# FIR analysis: combined Seen/Unseen (face + object) regressors.
# Source this file, then call fir_set_variant and use exported variables.
#
# All pipeline steps (01–10) use the same subject cohort:
#   BASELINE_min_seen_unseen == TRUE

FIR_ALL_VARIANTS=(FIR)
FIR_SUBJECT_FILTER_COL=BASELINE_min_seen_unseen

# 3rd-level group FEAT: pool 2nd-level copes 1–42 per variant (07–09).
FIR_3RD_COPES=($(seq 1 42))

fir_set_variant() {
    local v="$1"
    FIR_VARIANT="$v"
    case "$v" in
        FIR)
            FSLFEAT_SUBDIR=FIR
            FEAT_SUFFIX=FIR_space-MNI152NLin2009cAsym
            SEEN_EV_STEM=probedSeen
            UNSEEN_EV_STEM=probedUnseen
            EV_TITLE_SEEN=Seen
            EV_TITLE_UNSEEN=Unseen
            ;;
        *)
            echo "ERROR: unknown FIR variant: $v" >&2
            return 1
            ;;
    esac
    CSV_FILTER_COL="${FIR_SUBJECT_FILTER_COL}"
    FIRST_GLM_TEMPLATE="sub-PXX_ses-V2_task-VG_run-X_analysis-1stGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.fsf"
    SECOND_GLM_TEMPLATE="sub-PXX_ses-V2_task-VG_analysis-2ndGLM_${FIR_VARIANT}-MNI152NLin2009cAsym.fsf"
    return 0
}

# Write subject codes (one per line) for FIR_SUBJECT_FILTER_COL to stdout.
fir_list_subjects() {
    SUBJECT_CSV="${SUBJECT_CSV:?SUBJECT_CSV not set}" \
    FILTER_COL="${CSV_FILTER_COL:?call fir_set_variant first}" \
    python3 <<'PY'
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
col = os.environ["FILTER_COL"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in str(subj_df.columns[0]):
    subj_df = pd.read_csv(path, sep=";")
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
    sys.exit(1)
subj_df = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE")]
for sub in subj_df["sub_code"].values:
    print(sub)
PY
}
