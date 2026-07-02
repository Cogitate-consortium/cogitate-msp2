#!/usr/bin/env bash

#@author: matthias.ekman
#
# Modified by Yamil Vidal (05/2026)
# Overlap binarized 3rd-level thresholded z-stats (seen face vs unseen F and seen object vs unseen O).
# Run after 07_run_3rd_level.sh.
#
# Usage: bash 08_overlap.sh

module load FSL

export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
ACTIVATION_CODE_ROOT="${CODE_PATH}/activation"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
RESULTS_DIR="${ACTIVATION_CODE_ROOT}/results"

mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

file_O="${FSLFEAT_ROOT}/activation/group/seen_vs_unseen_O.gfeat/cope1.feat/thresh_zstat1.nii.gz"
file_F="${FSLFEAT_ROOT}/activation/group/seen_vs_unseen_F.gfeat/cope1.feat/thresh_zstat1.nii.gz"

if [ ! -f "$file_O" ] || [ ! -f "$file_F" ]; then
    echo "ERROR: missing group-level threshold maps." >&2
    echo "  O: ${file_O}" >&2
    echo "  F: ${file_F}" >&2
    exit 1
fi

cp -f "$file_O" O_thresh_zstat1.nii.gz
cp -f "$file_F" F_thresh_zstat1.nii.gz

fslmaths O_thresh_zstat1.nii.gz -bin O_thresh_zstat1_bin
fslmaths F_thresh_zstat1.nii.gz -bin F_thresh_zstat1_bin
fslmaths O_thresh_zstat1_bin -add F_thresh_zstat1_bin -thr 1.1 -bin overlap_O_and_F

echo ">> wrote ${RESULTS_DIR}/overlap_O_and_F.nii.gz"
