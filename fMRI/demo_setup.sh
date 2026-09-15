#!/usr/bin/env bash
# Activate demo mode for the fMRI sample dataset under ./data_demo.
#
# Usage (from the fMRI/ directory, or any cwd):
#   source /path/to/fMRI/demo_setup.sh
#
# Sets DEMO=1, BIDS_ROOT, CODE_PATH, SUBJECT_CSV; creates ses-V2 name aliases
# for raw (and derivative) ses-2 trees; ensures the demo subject CSV exists.
#
# Must be sourced (not executed) so exports persist in your shell.

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "ERROR: source this script instead of executing it:" >&2
    echo "  source ${0}" >&2
    exit 1
fi

_DEMO_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEMO=1
export CODE_PATH="${_DEMO_SETUP_DIR}"
export BIDS_ROOT="${CODE_PATH}/data_demo"
export SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri_demo.csv"
export SESSION="ses-V2"
export REGRESSOR_EVENT_ROOT="${BIDS_ROOT}/derivatives/regressoreventfiles"

# shellcheck source=demo_paths.sh
source "${CODE_PATH}/demo_paths.sh"

if [ ! -d "${BIDS_ROOT}" ]; then
    echo "ERROR: demo BIDS root not found: ${BIDS_ROOT}" >&2
    echo "  Download sample data from https://www.arc-cogitate.com/data-user" >&2
    echo "  and place/extract it as fMRI/data_demo/" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "${BIDS_ROOT}/participants.tsv" ]; then
    echo "ERROR: missing ${BIDS_ROOT}/participants.tsv" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Demo subject CSV from participants.tsv ---
python3 - "${BIDS_ROOT}/participants.tsv" "${SUBJECT_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

participants = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
rows = []
with participants.open(encoding="utf-8") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        pid = (row.get("participant_id") or "").strip()
        if not pid:
            continue
        code = pid[4:] if pid.startswith("sub-") else pid
        lab = code[:2] if len(code) >= 2 else ""
        rows.append(
            {
                "sub_code": code,
                "modality": "fMRI",
                "Lab": lab,
                "DECODING_min_sf_so_uf_uo": "TRUE",
                "DECODING_min_sf_so": "TRUE",
                "DECODING_min_sl_sr_ul_ur": "TRUE",
                "DECODING_min_sl_sr": "TRUE",
                "ACTIVATION_min_sf_uf": "TRUE",
                "ACTIVATION_min_so_uo": "TRUE",
                "SYNCHRONY_min_seen": "TRUE",
                "BASELINE_min_seen_unseen": "TRUE",
            }
        )

fieldnames = [
    "sub_code",
    "modality",
    "Lab",
    "DECODING_min_sf_so_uf_uo",
    "DECODING_min_sf_so",
    "DECODING_min_sl_sr_ul_ur",
    "DECODING_min_sl_sr",
    "ACTIVATION_min_sf_uf",
    "ACTIVATION_min_so_uo",
    "SYNCHRONY_min_seen",
    "BASELINE_min_seen_unseen",
]
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
print(f">> wrote {out_csv} ({len(rows)} subjects)")
PY

# --- ses-2 -> ses-V2 filename aliases (relative symlinks) ---
demo_link_ses_tree() {
    local src_ses="$1"   # absolute path to .../ses-2
    local dst_ses="$2"   # absolute path to .../ses-V2
    local src_token="$3" # ses-2
    local dst_token="$4" # ses-V2

    [ -d "$src_ses" ] || return 0
    mkdir -p "$dst_ses"

    local src_abs dst_abs rel target_name link_path
    # Walk files under src_ses
    while IFS= read -r -d '' f; do
        src_abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
        # path relative to src_ses
        local rel_dir
        rel_dir="$(realpath --relative-to="$src_ses" "$(dirname "$f")" 2>/dev/null || python3 -c "import os.path; print(os.path.relpath(os.path.dirname('$f'), '$src_ses'))")"
        target_name="$(basename "$f")"
        target_name="${target_name//${src_token}/${dst_token}}"
        mkdir -p "${dst_ses}/${rel_dir}"
        link_path="${dst_ses}/${rel_dir}/${target_name}"
        # Relative target from link location to source file
        rel="$(python3 -c "import os.path; print(os.path.relpath('${src_abs}', '${dst_ses}/${rel_dir}'))")"
        if [ -L "$link_path" ] || [ -e "$link_path" ]; then
            rm -f "$link_path"
        fi
        ln -s "$rel" "$link_path"
    done < <(find "$src_ses" -type f -print0 2>/dev/null)
}

echo ">> linking ses-2 -> ses-V2 aliases under ${BIDS_ROOT}"
shopt -s nullglob
for sub_dir in "${BIDS_ROOT}"/sub-*; do
    [ -d "$sub_dir" ] || continue
    if [ -d "${sub_dir}/ses-2" ]; then
        demo_link_ses_tree "${sub_dir}/ses-2" "${sub_dir}/ses-V2" "ses-2" "ses-V2"
        echo "   linked $(basename "$sub_dir")/ses-V2"
    fi
done

# fMRIPrep derivatives
if [ -d "${BIDS_ROOT}/derivatives/fmriprep" ]; then
    for sub_dir in "${BIDS_ROOT}/derivatives/fmriprep"/sub-*; do
        [ -d "$sub_dir" ] || continue
        if [ -d "${sub_dir}/ses-2" ]; then
            demo_link_ses_tree "${sub_dir}/ses-2" "${sub_dir}/ses-V2" "ses-2" "ses-V2"
            echo "   linked fmriprep/$(basename "$sub_dir")/ses-V2"
        fi
    done
fi

# FreeSurfer: often no ses- folder; if ses-2 exists, alias similarly
if [ -d "${BIDS_ROOT}/derivatives/freesurfer" ]; then
    for sub_dir in "${BIDS_ROOT}/derivatives/freesurfer"/sub-*; do
        [ -d "$sub_dir" ] || continue
        if [ -d "${sub_dir}/ses-2" ]; then
            demo_link_ses_tree "${sub_dir}/ses-2" "${sub_dir}/ses-V2" "ses-2" "ses-V2"
            echo "   linked freesurfer/$(basename "$sub_dir")/ses-V2"
        fi
    done
fi
shopt -u nullglob

echo ""
echo "Demo mode active:"
echo "  DEMO=${DEMO}"
echo "  BIDS_ROOT=${BIDS_ROOT}"
echo "  CODE_PATH=${CODE_PATH}"
echo "  SUBJECT_CSV=${SUBJECT_CSV}"
echo ""
echo "Next steps:"
echo "  1. Run fMRIPrep on \${BIDS_ROOT} (see README Sample data and demo)."
echo "  2. Re-run: source ${CODE_PATH}/demo_setup.sh   # link derivative ses-V2 aliases"
echo "  3. python3 ${CODE_PATH}/demo_make_ev_from_bids.py"
echo "  4. Follow README pipeline sections §3–§7 (skip RAW logfile extraction)."
echo "  Decoding (§8) is not part of this demo."
