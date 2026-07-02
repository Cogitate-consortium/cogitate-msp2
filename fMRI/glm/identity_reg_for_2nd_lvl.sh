#!/usr/bin/env bash
# Shared helpers for 2nd-level FEAT when 1st-level used regstandard_yn=0 (fMRIPrep MNI BOLD).
# Source from pipeline scripts (FIR/04, activation/02, gPPI/05); requires FSL in PATH.
#
# Provides: resolve_func_ref, create_identity_reg, spot_check_identity_reg,
#           run_identity_reg_and_qc, write_identity_reg_marker,
#           resolve_func_ref_gfeat, spot_check_identity_reg_gfeat,
#           create_identity_reg_gfeat, run_identity_reg_and_qc_gfeat

: "${MNI_BOLD_PATH_RE:=space-MNI152NLin2009cAsym_desc-preproc_bold}"

# True if path exists, is non-empty, and FSL can read volume metadata.
nifti_is_readable() {
    local img="$1"
    [ -f "$img" ] || return 1
    [ -s "$img" ] || return 1
    if ! command -v fslnvols >/dev/null 2>&1; then
        echo "ERROR: fslnvols not in PATH (load FSL before running this script)" >&2
        return 1
    fi
    fslnvols "$img" >/dev/null 2>&1
}

# Append candidate paths (raw, .nii.gz, alternate fmriprep roots) to array name in $1.
_bold_path_candidates() {
    local -n _out=$1
    local raw="$2"
    local p alt

    _out=()
    [ -n "$raw" ] || return 0

    _out+=("$raw")
    if [[ "$raw" != *.nii.gz ]]; then
        _out+=("${raw}.nii.gz")
    fi
    if [[ "$raw" == *"/derivatives/fmriprep/"* ]]; then
        alt="${raw//\/derivatives\/fmriprep\//\/derivatives\/exclude\/new\/fmriprep\/}"
        _out+=("$alt")
        if [[ "$alt" != *.nii.gz ]]; then
            _out+=("${alt}.nii.gz")
        fi
    elif [[ "$raw" == *"/derivatives/fmriprep/"* ]]; then
        alt="${raw//\/derivatives\/exclude\/new\/fmriprep\//\/derivatives\/fmriprep\/}"
        _out+=("$alt")
        if [[ "$alt" != *.nii.gz ]]; then
            _out+=("${alt}.nii.gz")
        fi
    fi
}

# Resolve fMRIPrep BOLD path from design.fsf (may omit .nii.gz or use legacy roots).
resolve_bold_path() {
    local raw="$1"
    local -a candidates=()
    local p

    _bold_path_candidates candidates "$raw"
    for p in "${candidates[@]}"; do
        if nifti_is_readable "$p"; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    return 1
}

# Resolve a 3D functional reference for reg/standard.nii.gz (mean_func may be absent).
resolve_func_ref() {
    local feat_path="$1"
    local candidate design bold resolved

    for candidate in \
        "${feat_path}/mean_func.nii.gz" \
        "${feat_path}/prefiltered_func_data.nii.gz" \
        "${feat_path}/filtered_func_data.nii.gz"; do
        if nifti_is_readable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    design="${feat_path}/design.fsf"
    if [ -f "$design" ]; then
        bold=$(grep -E '^set feat_files\(1\)' "$design" | sed -n 's/.*"\([^"]*\)".*/\1/p' | head -n 1)
        if [ -n "$bold" ]; then
            resolved=$(resolve_bold_path "$bold") && {
                printf '%s\n' "$resolved"
                return 0
            }
        fi
    fi

    if nifti_is_readable "${feat_path}/stats/cope1.nii.gz"; then
        echo "WARNING: ${feat_path}: using stats/cope1.nii.gz as functional reference (feat intermediates/BOLD unavailable)" >&2
        printf '%s\n' "${feat_path}/stats/cope1.nii.gz"
        return 0
    fi

    echo "ERROR: no readable functional reference in ${feat_path} (tried mean_func, prefiltered/filtered func, design.fsf feat_files(1), stats/cope1.nii.gz)" >&2
    return 1
}

# Integer volume count from fslnvols (defaults to 1 if unreadable).
nifti_volume_count() {
    local img="$1"
    local nvol

    nvol=$(fslnvols "$img" 2>/dev/null | head -n 1 | tr -d '[:space:]')
    if [[ -z "$nvol" || ! "$nvol" =~ ^[0-9]+$ ]]; then
        echo 1
        return 0
    fi
    echo "$nvol"
}

# FSL may append .nii.gz when the requested output name does not end with it.
resolve_fsl_output_path() {
    local intended="$1"

    if [ -f "$intended" ]; then
        printf '%s\n' "$intended"
        return 0
    fi
    if [[ "$intended" != *.nii.gz ]] && [ -f "${intended}.nii.gz" ]; then
        printf '%s\n' "${intended}.nii.gz"
        return 0
    fi
    return 1
}

write_standard_ref() {
    local func_ref="$1"
    local standard_out="$2"
    local nvol actual_out

    if ! command -v fslnvols >/dev/null 2>&1; then
        echo "ERROR: fslnvols not in PATH (load FSL before running this script)" >&2
        return 1
    fi

    if [[ "$standard_out" != *.nii.gz ]]; then
        standard_out="${standard_out}.nii.gz"
    fi

    nvol=$(nifti_volume_count "$func_ref")
    if [ "$nvol" -gt 1 ]; then
        if ! fslmaths "$func_ref" -Tmean "$standard_out"; then
            echo "ERROR: fslmaths -Tmean failed for ${func_ref}" >&2
            return 1
        fi
    else
        if ! cp "$func_ref" "$standard_out"; then
            echo "ERROR: cp failed for ${func_ref} -> ${standard_out}" >&2
            return 1
        fi
    fi

    actual_out=$(resolve_fsl_output_path "$standard_out") || {
        echo "ERROR: standard reference not written (expected ${standard_out})" >&2
        return 1
    }
    if [ "$actual_out" != "$standard_out" ]; then
        mv -f "$actual_out" "$standard_out" || return 1
    fi
    if ! nifti_is_readable "$standard_out"; then
        echo "ERROR: standard reference not readable: ${standard_out}" >&2
        return 1
    fi
    return 0
}

fslhd_field() {
    local img="$1" key="$2"
    fslhd "$img" 2>/dev/null | awk -v k="$key" '$1 == k { print $2; exit }'
}

pixdim_match() {
    local a="$1" b="$2"
    awk -v a="$a" -v b="$b" 'BEGIN {
        if (a == "" || b == "") exit 1
        if (a == b) exit 0
        if ((a - b) * (a - b) < 1e-8) exit 0
        exit 1
    }'
}

design_bold_from_fsf() {
    local feat_path="$1"
    local design="${feat_path}/design.fsf"
    local bold=""

    if [ ! -f "$design" ]; then
        echo "ERROR: missing design.fsf in ${feat_path}" >&2
        return 1
    fi

    bold=$(grep -E '^set feat_files\(1\)' "$design" | sed -n 's/.*"\([^"]*\)".*/\1/p' | head -n 1)
    if [ -z "$bold" ]; then
        echo "ERROR: no feat_files(1) in ${design}" >&2
        return 1
    fi
    printf '%s\n' "$bold"
}

# Validate identity assumption: cope grid == reg/standard; input BOLD is fMRIPrep MNI.
spot_check_identity_reg() {
    local feat_path="$1"
    local cope="${feat_path}/stats/cope1.nii.gz"
    local standard="${feat_path}/reg/standard.nii.gz"
    local bold bold_resolved dim d cope_dim std_dim cope_pix std_pix bold_note=""

    if [ ! -f "$cope" ]; then
        echo "ERROR: QC missing stats/cope1.nii.gz in ${feat_path}" >&2
        return 1
    fi
    if [ ! -f "$standard" ]; then
        echo "ERROR: QC missing reg/standard.nii.gz in ${feat_path}" >&2
        return 1
    fi

    for d in 1 2 3; do
        cope_dim=$(fslhd_field "$cope" "dim${d}")
        std_dim=$(fslhd_field "$standard" "dim${d}")
        if [ -z "$cope_dim" ] || [ -z "$std_dim" ]; then
            echo "ERROR: QC could not read dim${d} (cope1=${cope_dim:-?} standard=${std_dim:-?}) in ${feat_path}" >&2
            return 1
        fi
        if [ "$cope_dim" != "$std_dim" ]; then
            echo "ERROR: QC dim${d} mismatch cope1=${cope_dim} standard=${std_dim} in ${feat_path}" >&2
            return 1
        fi

        cope_pix=$(fslhd_field "$cope" "pixdim${d}")
        std_pix=$(fslhd_field "$standard" "pixdim${d}")
        if [ -z "$cope_pix" ] || [ -z "$std_pix" ]; then
            echo "ERROR: QC could not read pixdim${d} in ${feat_path}" >&2
            return 1
        fi
        if ! pixdim_match "$cope_pix" "$std_pix"; then
            echo "ERROR: QC pixdim${d} mismatch cope1=${cope_pix} standard=${std_pix} in ${feat_path}" >&2
            return 1
        fi
    done

    bold=$(design_bold_from_fsf "$feat_path") || return 1
    if [[ "$bold" != *"${MNI_BOLD_PATH_RE}"* ]]; then
        echo "ERROR: QC feat_files(1) is not fMRIPrep MNI preproc BOLD (expected *${MNI_BOLD_PATH_RE}*): ${bold}" >&2
        return 1
    fi

    bold_resolved=$(resolve_bold_path "$bold") || true
    if [ -n "$bold_resolved" ]; then
        bold_note="bold=${bold_resolved}"
    else
        bold_note="bold_fsf=${bold} (not on disk; geometry QC only)"
    fi

    dim="$(fslhd_field "$cope" dim1)x$(fslhd_field "$cope" dim2)x$(fslhd_field "$cope" dim3)"
    echo ">> QC OK ${feat_path}: cope1==standard (${dim}); ${bold_note}"
    return 0
}

create_identity_reg() {
    local feat_path="$1"
    local reg_dir="${feat_path}/reg"
    local func_ref standard_out tmp_standard
    local ident_mat="${IDENT_MAT:-${FSLDIR}/etc/flirtsch/ident.mat}"

    if [ ! -f "$ident_mat" ]; then
        echo "ERROR: identity matrix not found (is FSL loaded?): ${ident_mat}" >&2
        return 1
    fi

    func_ref=$(resolve_func_ref "$feat_path") || return 1

    if [ -d "${feat_path}/reg_standard" ]; then
        rm -rf "${feat_path}/reg_standard_bkp"
        mv "${feat_path}/reg_standard" "${feat_path}/reg_standard_bkp"
    fi

    if [ -d "$reg_dir" ]; then
        if [ ! -d "${feat_path}/reg_bkp" ]; then
            cp -R "$reg_dir" "${feat_path}/reg_bkp"
        fi
        shopt -s nullglob
        rm -f "${reg_dir}/"*.mat
        shopt -u nullglob
        rm -f "${reg_dir}/standard.nii.gz"
    else
        mkdir -p "$reg_dir"
    fi

    cp "$ident_mat" "${reg_dir}/example_func2standard.mat"
    standard_out="${reg_dir}/standard.nii.gz"
    rm -f "$standard_out"

    tmp_standard=$(mktemp "${reg_dir}/standard_XXXXXX.nii.gz")
    if ! write_standard_ref "$func_ref" "$tmp_standard"; then
        rm -f "$tmp_standard"
        return 1
    fi
    if ! mv -f "$tmp_standard" "$standard_out"; then
        echo "ERROR: failed to move ${tmp_standard} -> ${standard_out}" >&2
        rm -f "$tmp_standard"
        return 1
    fi
    if ! nifti_is_readable "$standard_out"; then
        echo "ERROR: missing ${standard_out} after identity reg build" >&2
        return 1
    fi

    if [ ! -f "${feat_path}/mean_func.nii.gz" ]; then
        cp "$standard_out" "${feat_path}/mean_func.nii.gz" || return 1
    fi
}

write_identity_reg_marker() {
    local marker_path="$1"
    local feat_path="$2"

    {
        echo "status=done"
        echo "qc=pass"
        echo "feat_path=${feat_path}"
        echo "date=$(date -Iseconds 2>/dev/null || date)"
    } >"$marker_path"
}

run_identity_reg_and_qc() {
    local feat_path="$1"
    local marker_path="$2"

    if create_identity_reg "$feat_path" && spot_check_identity_reg "$feat_path"; then
        write_identity_reg_marker "$marker_path" "$feat_path"
        return 0
    fi
    return 1
}

# --- 2nd-level .gfeat (3rd-level copeN.feat inputs): identity reg at gfeat root ---
#
# Inside parent .gfeat/copeN.feat the sub-FEAT has a single contrast → stats/cope1.nii.gz
# (not stats/copeN.nii.gz). cope_idx selects the subdirectory only.

gfeat_child_cope_stats_nii() {
    local cope_feat_dir="$1"
    local parent_cope_idx="$2"
    local candidate

    for candidate in \
        "${cope_feat_dir}/stats/cope1.nii.gz" \
        "${cope_feat_dir}/stats/cope${parent_cope_idx}.nii.gz"; do
        if nifti_is_readable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_func_ref_gfeat() {
    local gfeat_path="$1"
    local cope_idx="$2"
    local cope_feat="${gfeat_path}/cope${cope_idx}.feat"
    local candidate cope_stats

    for candidate in \
        "${gfeat_path}/mean_func.nii.gz" \
        "${cope_feat}/mean_func.nii.gz"; do
        if nifti_is_readable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    cope_stats=$(gfeat_child_cope_stats_nii "$cope_feat" "$cope_idx") || true
    if [ -n "$cope_stats" ]; then
        printf '%s\n' "$cope_stats"
        return 0
    fi

    echo "ERROR: no readable functional reference in ${gfeat_path} for cope${cope_idx} (tried gfeat/mean_func, cope${cope_idx}.feat/mean_func, cope${cope_idx}.feat/stats/cope1.nii.gz)" >&2
    return 1
}

spot_check_identity_reg_gfeat() {
    local gfeat_path="$1"
    local cope_idx="$2"
    local cope_feat="${gfeat_path}/cope${cope_idx}.feat"
    local cope
    cope=$(gfeat_child_cope_stats_nii "$cope_feat" "$cope_idx") || cope=""
    local standard="${gfeat_path}/reg/standard.nii.gz"
    local dim d cope_dim std_dim cope_pix std_pix

    if [ -z "$cope" ] || [ ! -f "$cope" ]; then
        echo "ERROR: QC missing ${cope_feat}/stats/cope1.nii.gz in ${gfeat_path}" >&2
        return 1
    fi
    if [ ! -f "$standard" ]; then
        echo "ERROR: QC missing reg/standard.nii.gz in ${gfeat_path}" >&2
        return 1
    fi

    for d in 1 2 3; do
        cope_dim=$(fslhd_field "$cope" "dim${d}")
        std_dim=$(fslhd_field "$standard" "dim${d}")
        if [ -z "$cope_dim" ] || [ -z "$std_dim" ]; then
            echo "ERROR: QC could not read dim${d} (cope${cope_idx}=${cope_dim:-?} standard=${std_dim:-?}) in ${gfeat_path}" >&2
            return 1
        fi
        if [ "$cope_dim" != "$std_dim" ]; then
            echo "ERROR: QC dim${d} mismatch cope${cope_idx}=${cope_dim} standard=${std_dim} in ${gfeat_path}" >&2
            return 1
        fi

        cope_pix=$(fslhd_field "$cope" "pixdim${d}")
        std_pix=$(fslhd_field "$standard" "pixdim${d}")
        if [ -z "$cope_pix" ] || [ -z "$std_pix" ]; then
            echo "ERROR: QC could not read pixdim${d} in ${gfeat_path}" >&2
            return 1
        fi
        if ! pixdim_match "$cope_pix" "$std_pix"; then
            echo "ERROR: QC pixdim${d} mismatch cope${cope_idx}=${cope_pix} standard=${std_pix} in ${gfeat_path}" >&2
            return 1
        fi
    done

    dim="$(fslhd_field "$cope" dim1)x$(fslhd_field "$cope" dim2)x$(fslhd_field "$cope" dim3)"
    echo ">> QC OK ${gfeat_path}: cope${cope_idx}==standard (${dim})"
    return 0
}

create_identity_reg_gfeat() {
    local gfeat_path="$1"
    local cope_idx="$2"
    local reg_dir="${gfeat_path}/reg"
    local func_ref standard_out tmp_standard
    local ident_mat="${IDENT_MAT:-${FSLDIR}/etc/flirtsch/ident.mat}"

    if [ ! -f "$ident_mat" ]; then
        echo "ERROR: identity matrix not found (is FSL loaded?): ${ident_mat}" >&2
        return 1
    fi

    func_ref=$(resolve_func_ref_gfeat "$gfeat_path" "$cope_idx") || return 1

    if [ -d "${gfeat_path}/reg_standard" ]; then
        rm -rf "${gfeat_path}/reg_standard_bkp"
        mv "${gfeat_path}/reg_standard" "${gfeat_path}/reg_standard_bkp"
    fi

    if [ -d "$reg_dir" ]; then
        if [ ! -d "${gfeat_path}/reg_bkp" ]; then
            cp -R "$reg_dir" "${gfeat_path}/reg_bkp"
        fi
        shopt -s nullglob
        rm -f "${reg_dir}/"*.mat
        shopt -u nullglob
        rm -f "${reg_dir}/standard.nii.gz"
    else
        mkdir -p "$reg_dir"
    fi

    cp "$ident_mat" "${reg_dir}/example_func2standard.mat"
    standard_out="${reg_dir}/standard.nii.gz"
    rm -f "$standard_out"

    tmp_standard=$(mktemp "${reg_dir}/standard_XXXXXX.nii.gz")
    if ! write_standard_ref "$func_ref" "$tmp_standard"; then
        rm -f "$tmp_standard"
        return 1
    fi
    if ! mv -f "$tmp_standard" "$standard_out"; then
        echo "ERROR: failed to move ${tmp_standard} -> ${standard_out}" >&2
        rm -f "$tmp_standard"
        return 1
    fi
    if ! nifti_is_readable "$standard_out"; then
        echo "ERROR: missing ${standard_out} after identity reg build" >&2
        return 1
    fi

    if [ ! -f "${gfeat_path}/mean_func.nii.gz" ]; then
        cp "$standard_out" "${gfeat_path}/mean_func.nii.gz" || return 1
    fi
}

run_identity_reg_and_qc_gfeat() {
    local gfeat_path="$1"
    local cope_idx="$2"
    local marker_path="$3"

    if create_identity_reg_gfeat "$gfeat_path" "$cope_idx" && spot_check_identity_reg_gfeat "$gfeat_path" "$cope_idx"; then
        write_identity_reg_marker "$marker_path" "$gfeat_path"
        return 0
    fi
    return 1
}
