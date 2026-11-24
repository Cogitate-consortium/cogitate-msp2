#!/usr/bin/env bash
# Compare new vs old gPPI seed masks (FFA/LOC NIfTI) per subject.
#
# For each sub-* under NEW_ROOT, load that subject's seed masks and compute
# overlap with the matching filename under OLD_ROOT. Overlap is the smaller of
#   100 * |new ∩ old| / |new|
#   100 * |new ∩ old| / |old|
# (binary voxels). Pairs below THRESH percent, plus missing/failed pairs, are
# written to low_overlap.txt next to this script.
#
# Needs FSL (fslmaths, fslstats). On Wish:  module load fsl/6.0.7.23

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEW_ROOT="${NEW_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/new/gppi_seeds}"
OLD_ROOT="${OLD_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/old/gppi_seeds}"
OUT="${OUT:-${_SCRIPT_DIR}/low_overlap.txt}"
THRESH="${THRESH:-95}"

if ! command -v fslmaths >/dev/null 2>&1 || ! command -v fslstats >/dev/null 2>&1; then
    echo "ERROR: fslmaths/fslstats not in PATH. Load FSL (e.g. module load fsl/6.0.7.23)." >&2
    exit 1
fi

if [ ! -d "$NEW_ROOT" ]; then
    echo "ERROR: new seeds root not found: $NEW_ROOT" >&2
    exit 1
fi
if [ ! -d "$OLD_ROOT" ]; then
    echo "ERROR: old seeds root not found: $OLD_ROOT" >&2
    exit 1
fi

_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gppi_seed_overlap.XXXXXX")"
trap 'rm -rf "$_TMP"' EXIT

stem_of() {
    local f="$1"
    local b
    b="$(basename "$f")"
    b="${b%.nii.gz}"
    b="${b%.nii}"
    printf '%s' "$b"
}

# Prefer .nii.gz when both uncompressed and gzipped copies exist.
list_masks() {
    local root="$1"
    local seen_gz=""
    local f stem
    if [ ! -d "$root" ]; then
        return 0
    fi
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$f" in
            *.nii.gz) seen_gz="${seen_gz}$(stem_of "$f")"$'\n' ;;
        esac
    done < <(find "$root" -type f \( -name '*.nii.gz' -o -name '*.nii' \) ! -name '*timecourse*' | sort)

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        stem="$(stem_of "$f")"
        if [[ "$f" == *.nii && "$f" != *.nii.gz ]]; then
            if printf '%s' "$seen_gz" | grep -Fxq "$stem"; then
                continue
            fi
        fi
        printf '%s\n' "$f"
    done < <(find "$root" -type f \( -name '*.nii.gz' -o -name '*.nii' \) ! -name '*timecourse*' | sort)
}

find_old_mask() {
    local sub="$1"
    local stem="$2"
    local sub_dir="${OLD_ROOT}/${sub}"
    local hits
    if [ ! -d "$sub_dir" ]; then
        return 1
    fi
    hits="$(find "$sub_dir" -type f \( -name "${stem}.nii.gz" -o -name "${stem}.nii" \) | sort)"
    if [ -z "$hits" ]; then
        return 1
    fi
    # Prefer gzipped
    local gz
    gz="$(printf '%s\n' "$hits" | grep '\.nii\.gz$' | head -n 1 || true)"
    if [ -n "$gz" ]; then
        printf '%s' "$gz"
    else
        printf '%s' "$(printf '%s\n' "$hits" | head -n 1)"
    fi
}

fsl_nvox() {
    # fslstats -V: "<nvoxels> <volume_mm3>"
    fslstats "$1" -V | awk '{printf "%d", $1+0}'
}

fsl_dims() {
    fslval "$1" dim1; printf ' '
    fslval "$1" dim2; printf ' '
    fslval "$1" dim3
}

record_low() {
    printf '%s\n' "$1" >>"$OUT"
}

: >"$OUT"
{
    echo "# new vs old gPPI seed overlap (threshold ${THRESH}%)"
    echo "# overlap_pct = min(100*|∩|/|new|, 100*|∩|/|old|); Dice also reported"
    echo "# NEW_ROOT=${NEW_ROOT}"
    echo "# OLD_ROOT=${OLD_ROOT}"
    echo "# date=$(date -Iseconds)"
    echo "sub	mask	overlap_pct	dice_pct	n_new	n_old	n_intersect	reason	new_path	old_path"
} >>"$OUT"

n_subs=0
n_pairs=0
n_low=0
n_ok=0
n_missing=0
n_grid=0

echo ">> new: $NEW_ROOT"
echo ">> old: $OLD_ROOT"
echo ">> write pairs with overlap < ${THRESH}% to: $OUT"
echo

printf '%-12s %-8s %10s %10s %8s %8s %8s  %s\n' \
    "sub" "mask" "overlap%" "dice%" "n_new" "n_old" "n_int" "status"

shopt -s nullglob
for sub_dir in "${NEW_ROOT}"/sub-*; do
    [ -d "$sub_dir" ] || continue
    sub="$(basename "$sub_dir")"
    n_subs=$((n_subs + 1))

    mapfile -t new_masks < <(list_masks "$sub_dir")
    if [ ${#new_masks[@]} -eq 0 ]; then
        echo "WARN: no NIfTI masks under $sub_dir" >&2
        record_low "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
            "$sub" "ALL" "NA" "NA" "NA" "NA" "NA" "NO_NEW_MASKS" "$sub_dir" "")"
        n_low=$((n_low + 1))
        continue
    fi

    for new_nii in "${new_masks[@]}"; do
        n_pairs=$((n_pairs + 1))
        stem="$(stem_of "$new_nii")"
        mask_label="$stem"
        case "$stem" in
            *FFA*) mask_label="FFA" ;;
            *LOC*) mask_label="LOC" ;;
        esac

        old_nii=""
        if ! old_nii="$(find_old_mask "$sub" "$stem")"; then
            printf '%-12s %-8s %10s %10s %8s %8s %8s  %s\n' \
                "$sub" "$mask_label" "NA" "NA" "?" "?" "?" "MISSING_OLD"
            record_low "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
                "$sub" "$mask_label" "NA" "NA" "NA" "NA" "NA" "MISSING_OLD" "$new_nii" "")"
            n_missing=$((n_missing + 1))
            n_low=$((n_low + 1))
            continue
        fi

        new_dims="$(fsl_dims "$new_nii" | tr -d '\n' | tr -s ' ')"
        old_dims="$(fsl_dims "$old_nii" | tr -d '\n' | tr -s ' ')"
        if [ "$new_dims" != "$old_dims" ]; then
            printf '%-12s %-8s %10s %10s %8s %8s %8s  %s\n' \
                "$sub" "$mask_label" "NA" "NA" "?" "?" "?" "GRID_MISMATCH"
            record_low "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
                "$sub" "$mask_label" "NA" "NA" "NA" "NA" "NA" "GRID_MISMATCH ${new_dims} vs ${old_dims}" "$new_nii" "$old_nii")"
            n_grid=$((n_grid + 1))
            n_low=$((n_low + 1))
            continue
        fi

        bin_new="${_TMP}/new_bin.nii.gz"
        bin_old="${_TMP}/old_bin.nii.gz"
        bin_ov="${_TMP}/overlap.nii.gz"

        fslmaths "$new_nii" -bin "$bin_new"
        fslmaths "$old_nii" -bin "$bin_old"
        fslmaths "$bin_new" -mul "$bin_old" "$bin_ov"

        n_new="$(fsl_nvox "$bin_new")"
        n_old="$(fsl_nvox "$bin_old")"
        n_int="$(fsl_nvox "$bin_ov")"

        read -r overlap_pct dice_pct < <(awk -v a="$n_new" -v b="$n_old" -v i="$n_int" 'BEGIN{
            pct_a = (a>0) ? 100.0*i/a : 0
            pct_b = (b>0) ? 100.0*i/b : 0
            ov = (pct_a < pct_b) ? pct_a : pct_b
            dice = (a+b>0) ? 200.0*i/(a+b) : 0
            printf "%.2f %.2f", ov, dice
        }')

        below="$(awk -v p="$overlap_pct" -v t="$THRESH" 'BEGIN{print (p+0 < t+0) ? 1 : 0}')"
        if [ "$below" -eq 1 ]; then
            status="LOW_OVERLAP"
            printf '%-12s %-8s %10s %10s %8s %8s %8s  %s\n' \
                "$sub" "$mask_label" "$overlap_pct" "$dice_pct" "$n_new" "$n_old" "$n_int" "$status"
            record_low "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
                "$sub" "$mask_label" "$overlap_pct" "$dice_pct" "$n_new" "$n_old" "$n_int" "$status" "$new_nii" "$old_nii")"
            n_low=$((n_low + 1))
        else
            status="OK"
            printf '%-12s %-8s %10s %10s %8s %8s %8s  %s\n' \
                "$sub" "$mask_label" "$overlap_pct" "$dice_pct" "$n_new" "$n_old" "$n_int" "$status"
            n_ok=$((n_ok + 1))
        fi
    done
done

echo
if [ "$n_subs" -eq 0 ]; then
    echo "ERROR: no sub-* directories in $NEW_ROOT" >&2
    exit 1
fi
echo ">> subjects=${n_subs}  pairs=${n_pairs}  ok=${n_ok}  written_to_low_overlap=${n_low} (missing_old=${n_missing} grid_mismatch=${n_grid})"
echo ">> low-overlap file: $OUT"
