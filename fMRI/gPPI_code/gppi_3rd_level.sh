#!/usr/bin/env bash
# gPPI 3rd-level group FEAT (08–10). Source this file in 3rd-level scripts.
#
# Each analysis uses one 2nd-level cope (lower-level 1st-level PPI contrast):
#   PPI_FFA → cope 3 (ppi_FFA)
#   PPI_LOC → cope 4 (ppi_LOC)

GPPI_3RD_ANALYSES=(PPI_LOC PPI_FFA)

gppi_3rd_cope_for() {
    case "$1" in
        PPI_FFA) echo 3 ;;
        PPI_LOC) echo 4 ;;
        *)
            echo "ERROR: unknown gPPI 3rd-level analysis: $1" >&2
            return 1
            ;;
    esac
}
