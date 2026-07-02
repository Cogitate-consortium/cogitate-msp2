#!/usr/bin/env bash
# Inspect all gPPI 2nd-level .gfeat outputs for issues that can cause all-zero 3rd-level maps.
#
# PPI_FFA -> cope3.feat (inner stats/cope1), PPI_LOC -> cope4.feat.
# Writes gPPI_code/3rd level report/inspect_2nd_level_summary.csv and
# inspect_2nd_level_report.txt
#
# Usage:
#   bash inspect_2nd_level_for_3rd.sh
#   bash inspect_2nd_level_for_3rd.sh --simulate-mask   # also run fslmaths mask -Tmin simulation

set -euo pipefail

module load FSL 2>/dev/null || true

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
REPORT_DIR="${GPPI_CODE_ROOT}/3rd level report"

SIMULATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --simulate-mask)
            SIMULATE="--simulate-mask-intersection"
            shift
            ;;
        -h|--help)
            echo "Usage: bash inspect_2nd_level_for_3rd.sh [--simulate-mask]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

mkdir -p "$REPORT_DIR"

export PYTHONPATH="${GPPI_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${GPPI_CODE_ROOT}/inspect_2nd_level_for_3rd.py" scan \
    --bids-root "$BIDS_ROOT" \
    --subject-csv "$SUBJECT_CSV" \
    --feat-root "$FSLFEAT_ROOT" \
    --out-dir "$REPORT_DIR" \
    $SIMULATE

SUMMARY_CSV="${REPORT_DIR}/inspect_2nd_level_summary.csv"
for analysis in PPI_FFA PPI_LOC; do
    list_path="${REPORT_DIR}/inputs_${analysis}_strict_ok.txt"
    python3 - "${SUMMARY_CSV}" "${analysis}" "${list_path}" <<'PY'
import csv
import sys

summary, analysis, out_path = sys.argv[1:4]
paths = []
with open(summary, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["analysis"] == analysis and row["ok_3rd_strict"] == "1" and row["cope_feat_path"]:
            paths.append(row["cope_feat_path"])
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(paths) + ("\n" if paths else ""))
print(f">> {analysis}: strict_ok={len(paths)} -> {out_path}")
PY
done
