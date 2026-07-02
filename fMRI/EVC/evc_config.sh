# Shared paths and naming for the EVC localizer / ROI pipeline.
# Source from other EVC scripts:  source "$(dirname "${BASH_SOURCE[0]}")/evc_config.sh"

BIDS_ROOT="${BIDS_ROOT:-/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids}"
CODE_PATH="${CODE_PATH:-${BIDS_ROOT}/code}"
SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri.csv}"

# Subjects with TRUE in SYNCHRONY_min_seen (same cohort as gPPI synchrony analyses).
SUBJECT_CSV_COLUMNS="${SUBJECT_CSV_COLUMNS:-SYNCHRONY_min_seen}"
# Optional: set SUBJECT_CSV_COLUMN alone to filter a single column instead of SUBJECT_CSV_COLUMNS.
SUBJECT_CSV_COLUMN="${SUBJECT_CSV_COLUMN:-}"

EVC_CODE_ROOT="${CODE_PATH}/EVC"
EVENTS_JSON_TEMPLATE="${EVC_CODE_ROOT}/events-json_file-templates/events-json_file-template_ses-V2_task-EVCLoc.json"
FSF_TEMPLATE_DIR="${EVC_CODE_ROOT}/fsf_templates"
FSF_FILES_ROOT="${EVC_CODE_ROOT}/fsf_files"
FMRIPREP_ROOT="${BIDS_ROOT}/derivatives/fmriprep"
FREESURFER_ROOT="${BIDS_ROOT}/derivatives/freesurfer"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
REGRESSOR_EVENT_ROOT="${REGRESSOR_EVENT_ROOT:-${BIDS_ROOT}/derivatives/regressoreventfiles}"
EVC_ROIS_ROOT="${EVC_ROIS_ROOT:-${BIDS_ROOT}/derivatives/evc_rois}"

# Reject legacy derivatives/regressoreventfiles (without exclude/new).
evc_validate_regressor_root() {
    case "${REGRESSOR_EVENT_ROOT}" in
        */derivatives/regressoreventfiles) return 0 ;;
        *)
            echo "ERROR: REGRESSOR_EVENT_ROOT must be under derivatives/regressoreventfiles" >&2
            echo "  got: ${REGRESSOR_EVENT_ROOT}" >&2
            return 1
            ;;
    esac
}

evc_assert_no_legacy_regressor_paths() {
    local file="$1"
    local line
    [[ -f "$file" ]] || return 0
    while IFS= read -r line; do
        if [[ "$line" == *"/derivatives/regressoreventfiles/"* ]] \
            && [[ "$line" != *"/derivatives/regressoreventfiles/"* ]]; then
            echo "ERROR: ${file} uses legacy derivatives/regressoreventfiles path" >&2
            echo "  use: ${REGRESSOR_EVENT_ROOT}/" >&2
            echo "  line: ${line}" >&2
            return 1
        fi
    done < <(grep -E '/derivatives/regressoreventfiles/' "$file" 2>/dev/null || true)
    return 0
}

evc_validate_regressor_root || exit 1

EVC_SESSION="ses-V2"
EVC_TASK="task-EVCLoc_run-1"
FEAT_SUFFIX="analysis-1stROI_space-T1w"
FEAT_BASENAME="${EVC_SESSION}_${EVC_TASK}_${FEAT_SUFFIX}"

# Binary EVC ROI: top-N from zstat4 (TLBR>TRBL), then fill to EVC_ROI_TOTAL_VOXELS from zstat5 (TRBL>TLBR), no overlap.
EVC_ROI_TOP_N_PER_CONTRAST="${EVC_ROI_TOP_N_PER_CONTRAST:-150}"
EVC_ROI_TOTAL_VOXELS="${EVC_ROI_TOTAL_VOXELS:-300}"

# Populate SUBJECTS (sub_code values) from SUBJECT_CSV and inclusion column(s).
# Returns 1 on error; sets SUBJECTS bash array and EVC_SUBJECT_FILTER_DESC.
evc_load_subjects() {
    local _tmp
    _tmp="$(mktemp)"
    SUBJECT_CSV="$SUBJECT_CSV" \
    SUBJECT_CSV_COLUMNS="$SUBJECT_CSV_COLUMNS" \
    SUBJECT_CSV_COLUMN="$SUBJECT_CSV_COLUMN" \
    python3 <<'PY' >"$_tmp"
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
single = os.environ.get("SUBJECT_CSV_COLUMN", "").strip()
columns = os.environ.get("SUBJECT_CSV_COLUMNS", "").strip()

subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

if "sub_code" not in subj_df.columns:
    print("ERROR: missing sub_code column", file=sys.stderr)
    sys.exit(1)

if single:
    cols = [single]
    desc = single
else:
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    desc = " OR ".join(cols) if cols else ""

if not cols:
    print("ERROR: set SUBJECT_CSV_COLUMNS or SUBJECT_CSV_COLUMN", file=sys.stderr)
    sys.exit(1)

mask = pd.Series(False, index=subj_df.index)
for col in cols:
    if col not in subj_df.columns:
        print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
        sys.exit(1)
    mask = mask | subj_df[col].astype(str).str.upper().eq("TRUE")

subj_df = subj_df.loc[mask]
print(f"# filter={desc}", file=sys.stderr)
for sub in subj_df["sub_code"].values:
    print(sub)
PY
    local _st=$?
    if [ "$_st" -ne 0 ]; then
        rm -f "$_tmp"
        return "$_st"
    fi
    SUBJECTS=()
    mapfile -t SUBJECTS < "$_tmp"
    rm -f "$_tmp"
    if [ ${#SUBJECTS[@]} -eq 0 ]; then
        echo "ERROR: no subjects after inclusion filter in ${SUBJECT_CSV}" >&2
        return 1
    fi
    if [ -n "$SUBJECT_CSV_COLUMN" ]; then
        EVC_SUBJECT_FILTER_DESC="${SUBJECT_CSV_COLUMN} == TRUE"
    else
        _csv_cols_or="${SUBJECT_CSV_COLUMNS//,/ OR }"
        EVC_SUBJECT_FILTER_DESC="(${_csv_cols_or}) == TRUE"
    fi
    return 0
}
