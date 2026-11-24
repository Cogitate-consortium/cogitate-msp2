#!/usr/bin/env bash
# Compare new vs old gPPI seed masks (FFA/LOC NIfTI) per subject.
#
# Overlap is the smaller of
#   100 * |new ∩ old| / |new|
#   100 * |new ∩ old| / |old|
# (binary voxels).
#
# Writes:
#   all_overlap.txt   every pair, overlap descending (NA / missing last)
#   low_overlap.txt   pairs with overlap < THRESH, same order
#
# Needs FSL (fslmaths, fslstats). On Wish:  module load fsl/6.0.7.23

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEW_ROOT="${NEW_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/new/gppi_seeds}"
OLD_ROOT="${OLD_ROOT:-/wishData/cogitate-data/fMRI/bids/derivatives/exclude/old/gppi_seeds}"
ALL_OUT="${ALL_OUT:-${_SCRIPT_DIR}/all_overlap.txt}"
LOW_OUT="${LOW_OUT:-${_SCRIPT_DIR}/low_overlap.txt}"
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
ROWS="${_TMP}/rows.tsv"

stem_of() {
    local f="$1"
    local b
    b="$(basename "$f")"
    b="${b%.nii.gz}"
    b="${b%.nii}"
    printf '%s' "$b"
}

nifti_list() {
    find "$1" -type f \( -name '*.nii.gz' -o -name '*.nii' \) ! -name '*timecourse*' | sort
}

# Unique sub-* ids from directory names and from NIfTI basenames.
list_subjects() {
    local root="$1"
    local d f base
    while IFS= read -r d; do
        [ -n "$d" ] || continue
        basename "$d"
    done < <(find "$root" -type d -name 'sub-*' | sort)
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        base="$(basename "$f")"
        if [[ "$base" =~ ^(sub-[A-Za-z0-9]+) ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
        fi
    done < <(nifti_list "$root")
}

# Prefer .nii.gz when both uncompressed and gzipped copies exist.
list_masks() {
    local root="$1"
    local sub="$2"
    local seen_gz="" f stem
    local files="${_TMP}/masks_${sub}.txt"
    : >"$files"
    nifti_list "$root" | while IFS= read -r f; do
        [ -n "$f" ] || continue
        base="$(basename "$f")"
        case "$base" in
            "${sub}"_*) printf '%s\n' "$f" ;;
            *)
                case "$f" in
                    */"${sub}"/*) printf '%s\n' "$f" ;;
                esac
                ;;
        esac
    done | sort -u >"$files"

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$f" in
            *.nii.gz) seen_gz="${seen_gz}$(stem_of "$f")"$'\n' ;;
        esac
    done <"$files"

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        stem="$(stem_of "$f")"
        if [[ "$f" == *.nii && "$f" != *.nii.gz ]]; then
            if printf '%s' "$seen_gz" | grep -Fxq "$stem"; then
                continue
            fi
        fi
        printf '%s\n' "$f"
    done <"$files"
}

find_old_mask() {
    local stem="$1"
    local hits gz
    hits="$(find "$OLD_ROOT" -type f \( -name "${stem}.nii.gz" -o -name "${stem}.nii" \) | sort || true)"
    if [ -z "$hits" ]; then
        return 1
    fi
    gz="$(printf '%s\n' "$hits" | grep '\.nii\.gz$' | head -n 1 || true)"
    if [ -n "$gz" ]; then
        printf '%s' "$gz"
    else
        printf '%s' "$(printf '%s\n' "$hits" | head -n 1)"
    fi
}

fsl_nvox() {
    fslstats "$1" -V | awk '{printf "%d", $1+0}'
}

fsl_dims() {
    fslval "$1" dim1; printf ' '
    fslval "$1" dim2; printf ' '
    fslval "$1" dim3
}

# sort_key <tab> sub mask overlap dice n_new n_old n_int reason new_path old_path
record_row() {
    local sort_key="$1"
    shift
    local IFS=$'\t'
    printf '%s\t%s\n' "$sort_key" "$*" >>"$ROWS"
}

n_new_nii="$(nifti_list "$NEW_ROOT" | wc -l | tr -d ' ')"
n_old_nii="$(nifti_list "$OLD_ROOT" | wc -l | tr -d ' ')"

echo ">> new: $NEW_ROOT"
echo ">> old: $OLD_ROOT"
echo ">> NIfTI files: new=${n_new_nii}  old=${n_old_nii}"
echo ">> listing NEW_ROOT:"
ls -la "$NEW_ROOT" | head -n 30
if [ "$n_new_nii" -eq 0 ]; then
    echo "ERROR: no .nii/.nii.gz masks under NEW_ROOT" >&2
    exit 1
fi

mapfile -t SUBJECTS < <(list_subjects "$NEW_ROOT" | sort -u)
n_subs=${#SUBJECTS[@]}
echo ">> subjects found: ${n_subs}"
printf '   %s\n' "${SUBJECTS[@]}"
echo

if [ "$n_subs" -eq 0 ]; then
    echo "ERROR: no sub-* ids under $NEW_ROOT" >&2
    exit 1
fi

: >"$ROWS"
n_pairs=0
n_ok=0
n_low=0
n_missing=0
n_grid=0
n_nomask=0

for sub in "${SUBJECTS[@]}"; do
    echo ">> ${sub}"
    mapfile -t new_masks < <(list_masks "$NEW_ROOT" "$sub")
    if [ ${#new_masks[@]} -eq 0 ]; then
        echo "   WARN: no NIfTI masks for ${sub}" >&2
        record_row "-1" "$sub" "ALL" "NA" "NA" "NA" "NA" "NA" "NO_NEW_MASKS" "" ""
        n_nomask=$((n_nomask + 1))
        continue
    fi
    echo "   masks: ${#new_masks[@]}"

    for new_nii in "${new_masks[@]}"; do
        n_pairs=$((n_pairs + 1))
        stem="$(stem_of "$new_nii")"
        mask_label="$stem"
        case "$stem" in
            *FFA*) mask_label="FFA" ;;
            *LOC*) mask_label="LOC" ;;
        esac

        old_nii=""
        if ! old_nii="$(find_old_mask "$stem")"; then
            echo "   ${mask_label}  MISSING_OLD  $(basename "$new_nii")"
            record_row "-1" "$sub" "$mask_label" "NA" "NA" "NA" "NA" "NA" "MISSING_OLD" "$new_nii" ""
            n_missing=$((n_missing + 1))
            continue
        fi

        new_dims="$(fsl_dims "$new_nii" | tr -d '\n' | tr -s ' ')"
        old_dims="$(fsl_dims "$old_nii" | tr -d '\n' | tr -s ' ')"
        if [ "$new_dims" != "$old_dims" ]; then
            echo "   ${mask_label}  GRID_MISMATCH  ${new_dims} vs ${old_dims}"
            record_row "-1" "$sub" "$mask_label" "NA" "NA" "NA" "NA" "NA" "GRID_MISMATCH ${new_dims} vs ${old_dims}" "$new_nii" "$old_nii"
            n_grid=$((n_grid + 1))
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
            n_low=$((n_low + 1))
        else
            status="OK"
            n_ok=$((n_ok + 1))
        fi
        echo "   ${mask_label}  ${overlap_pct}%  dice=${dice_pct}  n=${n_new}/${n_old}/${n_int}  ${status}"
        record_row "$overlap_pct" "$sub" "$mask_label" "$overlap_pct" "$dice_pct" "$n_new" "$n_old" "$n_int" "$status" "$new_nii" "$old_nii"
    done
done

header_lines() {
    echo "# new vs old gPPI seed overlap (threshold ${THRESH}%)"
    echo "# overlap_pct = min(100*|intersect|/|new|, 100*|intersect|/|old|); Dice also reported"
    echo "# sorted by overlap_pct descending (NA / missing last)"
    echo "# NEW_ROOT=${NEW_ROOT}"
    echo "# OLD_ROOT=${OLD_ROOT}"
    echo "# date=$(date -Iseconds)"
    printf '%s\n' $'sub\tmask\toverlap_pct\tdice_pct\tn_new\tn_old\tn_intersect\treason\tnew_path\told_path'
}

SORTED="${_TMP}/sorted.tsv"
# numeric descending on sort_key; strip the key
sort -t $'\t' -k1,1nr -k2,2 "$ROWS" | cut -f2- >"$SORTED"

{
    header_lines
    cat "$SORTED"
} >"$ALL_OUT"

{
    header_lines
    awk -F '\t' -v t="$THRESH" '
        $8 == "LOW_OVERLAP" {print; next}
        $3 == "NA" {print; next}
        ($3+0) < (t+0) {print}
    ' "$SORTED"
} >"$LOW_OUT"

echo
echo ">> subjects=${n_subs}  pairs=${n_pairs}  ok=${n_ok}  low=<${THRESH}%:${n_low}  missing_old=${n_missing}  grid=${n_grid}  no_new_masks=${n_nomask}"
echo ">> all overlaps (desc): $ALL_OUT"
echo ">> low overlap:         $LOW_OUT"
if [ ! -s "$ROWS" ]; then
    echo "ERROR: no comparison rows were written (discovery or FSL step produced nothing)" >&2
    exit 1
fi
