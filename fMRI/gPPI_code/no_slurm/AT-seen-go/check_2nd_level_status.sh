#!/usr/bin/env bash
# Summarise 2nd-level gPPI FEAT jobs from .run markers written by
# 07_run_PPI_2nd_level_Replay.sh / job_templates/PPI_2nd_lvl.sh.
#
# Expected jobs: SYNCHRONY_min_seen subjects × PPI_LOC,PPI_FFA.
#
# Usage:
#   bash check_2nd_level_status.sh
#
# If some jobs are FEAT_OK with start/finish timestamps, prints an ETA for
# QUEUED + RUNNING using median duration and GPPI_NJOBS (default 8).
#
# Writes job_files/run_PPI_2nd_level/_2nd_level_status.tsv

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"

GPPI_CODE_ROOT="${_SCRIPT_DIR}"
JOB_FILES_ROOT="${GPPI_CODE_ROOT}/job_files/run_PPI_2nd_level"
OUT_TSV="${JOB_FILES_ROOT}/_2nd_level_status.tsv"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat/gPPI_AT-seen-go"
GPPI_NJOBS="${GPPI_NJOBS:-8}"

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

_sublist_tmp="$(mktemp)"
SUBJECT_CSV="$SUBJECT_CSV" python3 <<'PY' >"$_sublist_tmp"
import os
import sys
import pandas as pd

path = os.environ["SUBJECT_CSV"]
subj_df = pd.read_csv(path, sep=None, engine="python")
if "sub_code" not in subj_df.columns and len(subj_df.columns) == 1 and ";" in subj_df.columns[0]:
    subj_df = pd.read_csv(path, sep=";")

col = "SYNCHRONY_min_seen"
if col not in subj_df.columns:
    print(f"ERROR: missing column {col} in {path}", file=sys.stderr)
    sys.exit(1)
subj_df = subj_df.loc[subj_df[col].astype(str).str.upper().eq("TRUE")]

for sub in subj_df["sub_code"].values:
    print(sub)
PY
_py_st=$?
if [ "$_py_st" -ne 0 ]; then
    rm -f "$_sublist_tmp"
    exit "$_py_st"
fi
mapfile -t SUBJECTS < "$_sublist_tmp"
rm -f "$_sublist_tmp"

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: no subjects after loading/filtering $SUBJECT_CSV" >&2
    exit 1
fi

declare -a ANALYSES=(PPI_LOC PPI_FFA)

last_status() {
    local f="$1"
    if [ ! -f "$f" ]; then
        return 0
    fi
    grep -E '^status=' "$f" 2>/dev/null | tail -n 1 | sed 's/^status=//' || true
}

feat_dir_for() {
    local subj_id="$1"
    local analysis="$2"
    local stem="${subj_id}_ses-V2_task-Replay_analysis-2ndGLM_${analysis}-MNI152NLin2009cAsym"
    local base="${FSLFEAT_ROOT}/${subj_id}"
    local d
    for d in "${base}/${stem}.gfeat" "${base}/${stem}.feat"; do
        if [ -d "$d" ]; then
            printf '%s' "$d"
            return 0
        fi
    done
    printf '%s' "${base}/${stem}.gfeat"
}

# Prints: bucket<TAB>note
classify() {
    local marker="$1"
    local feat_dir="$2"
    if [ ! -f "$marker" ]; then
        printf '%s\t%s\n' "NOT_QUEUED" "no .run marker"
        return 0
    fi
    local st started finished
    st="$(last_status "$marker")"
    started=0
    finished=0
    grep -q '========== job start' "$marker" 2>/dev/null && started=1 || true
    grep -q '========== job finished' "$marker" 2>/dev/null && finished=1 || true

    case "$st" in
        FEAT_OK)
            if [ -d "$feat_dir" ]; then
                printf '%s\t%s\n' "OK" "FEAT_OK"
            else
                printf '%s\t%s\n' "OK" "FEAT_OK (feat dir missing)"
            fi
            ;;
        FEAT_FAILED)
            printf '%s\t%s\n' "ERROR" "FEAT_FAILED"
            ;;
        PREFLIGHT_FAILED)
            printf '%s\t%s\n' "ERROR" "PREFLIGHT_FAILED"
            ;;
        QUEUED|"")
            if [ "$started" -eq 1 ] && [ "$finished" -eq 0 ]; then
                printf '%s\t%s\n' "RUNNING" "job started, no finish yet"
            else
                printf '%s\t%s\n' "QUEUED" "${st:-QUEUED}"
            fi
            ;;
        *)
            if [ "$started" -eq 1 ] && [ "$finished" -eq 0 ]; then
                printf '%s\t%s\n' "RUNNING" "status=${st}"
            else
                printf '%s\t%s\n' "ERROR" "status=${st}"
            fi
            ;;
    esac
}

mkdir -p "$JOB_FILES_ROOT"
{
    printf '%s\n' $'bucket\tsub\tanalysis\tstatus\tnote\tmarker\tfeat_dir'
} >"$OUT_TSV"

n_ok=0
n_err=0
n_run=0
n_queue=0
n_none=0
n_ok_missing_feat=0

echo ">> subjects=${#SUBJECTS[@]}  jobs_expected=$(( ${#SUBJECTS[@]} * ${#ANALYSES[@]} ))  (2 seeds)"
echo ">> markers: $JOB_FILES_ROOT"
echo ">> feat:    $FSLFEAT_ROOT"
echo

for subj in "${SUBJECTS[@]}"; do
    subj_id="sub-${subj}"
    job_dir="${JOB_FILES_ROOT}/${subj}"
    for analysis in "${ANALYSES[@]}"; do
        marker="${job_dir}/${subj_id}_ses-V2_task-Replay_analysis-2ndGLM_${analysis}-MNI152NLin2009cAsym.run"
        feat_dir="$(feat_dir_for "$subj_id" "$analysis")"
        IFS=$'\t' read -r bucket note < <(classify "$marker" "$feat_dir")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$bucket" "$subj" "$analysis" \
            "$(last_status "$marker")" \
            "$note" "$marker" "$feat_dir" >>"$OUT_TSV"
        case "$bucket" in
            OK)
                n_ok=$((n_ok + 1))
                if [[ "$note" == *"feat dir missing"* ]]; then
                    n_ok_missing_feat=$((n_ok_missing_feat + 1))
                fi
                ;;
            ERROR) n_err=$((n_err + 1)) ;;
            RUNNING) n_run=$((n_run + 1)) ;;
            QUEUED) n_queue=$((n_queue + 1)) ;;
            NOT_QUEUED) n_none=$((n_none + 1)) ;;
        esac
    done
done

n_total=$((n_ok + n_err + n_run + n_queue + n_none))

echo "========== 2nd-level status =========="
echo "OK (completed correctly):  ${n_ok}"
echo "ERROR (failed):            ${n_err}"
echo "RUNNING (started, not done): ${n_run}"
echo "QUEUED (waiting for slot): ${n_queue}"
echo "NOT_QUEUED (no marker):    ${n_none}"
echo "total expected:            ${n_total}"
if [ "$n_ok_missing_feat" -gt 0 ]; then
    echo "WARN: ${n_ok_missing_feat} FEAT_OK marker(s) have no .gfeat/.feat directory (check BIDS_ROOT)"
fi
echo "======================================"
echo ">> table: $OUT_TSV"

print_bucket() {
    local label="$1"
    local bucket="$2"
    local n
    n="$(awk -F '\t' -v b="$bucket" 'NR>1 && $1==b {c++} END {print c+0}' "$OUT_TSV")"
    if [ "$n" -eq 0 ]; then
        return
    fi
    echo
    echo "---- ${label} (${n}) ----"
    awk -F '\t' -v b="$bucket" 'NR>1 && $1==b {printf "  %s  %s  %s\n", $2, $3, $5}' "$OUT_TSV"
}

print_bucket "errors" "ERROR"
print_bucket "running" "RUNNING"
print_bucket "queued" "QUEUED"
if [ "$n_none" -gt 0 ] && [ "$n_none" -le 40 ]; then
    print_bucket "not queued" "NOT_QUEUED"
elif [ "$n_none" -gt 40 ]; then
    echo
    echo "---- not queued (${n_none}) — omitted (see TSV) ----"
fi

OUT_TSV="$OUT_TSV" GPPI_NJOBS="$GPPI_NJOBS" python3 <<'PY'
import datetime as dt
import os
import re
import statistics
import sys
from pathlib import Path

tsv = Path(os.environ["OUT_TSV"])
nslot = max(1, int(os.environ.get("GPPI_NJOBS", "8")))
start_re = re.compile(r"job start (\S+)")
fin_re = re.compile(r"job finished (\S+)")
now = dt.datetime.now().astimezone()

durs = []
running_elapsed = []
n_queue = 0
n_running = 0

if not tsv.is_file():
    sys.exit(0)

with tsv.open(encoding="utf-8", errors="replace") as fh:
    next(fh, None)
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        bucket, marker = parts[0], Path(parts[5])
        if bucket == "QUEUED":
            n_queue += 1
            continue
        if not marker.is_file():
            continue
        text = marker.read_text(encoding="utf-8", errors="replace")
        sm, fm = start_re.search(text), fin_re.search(text)
        if bucket == "OK" and sm and fm:
            try:
                t0 = dt.datetime.fromisoformat(sm.group(1))
                t1 = dt.datetime.fromisoformat(fm.group(1))
            except ValueError:
                continue
            sec = (t1 - t0).total_seconds()
            if sec > 0:
                durs.append(sec)
        elif bucket == "RUNNING":
            n_running += 1
            if sm:
                try:
                    t0 = dt.datetime.fromisoformat(sm.group(1))
                except ValueError:
                    continue
                running_elapsed.append(max(0.0, (now - t0).total_seconds()))

def fmt(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s"

print()
if not durs:
    print(">> ETA: no finished jobs with start/finish timestamps yet")
    sys.exit(0)

med = statistics.median(durs)
slots = max(1, min(nslot, n_queue + n_running) if (n_queue + n_running) else nslot)
work = n_queue * med + sum(max(0.0, med - e) for e in running_elapsed)
eta = work / slots
finish_at = now + dt.timedelta(seconds=eta)
print(
    f">> ETA from {len(durs)} FEAT_OK job(s): "
    f"median={fmt(med)}  min={fmt(min(durs))}  max={fmt(max(durs))}"
)
print(
    f">> remaining: queued={n_queue}  running={n_running}  "
    f"slots={slots} (GPPI_NJOBS={nslot})"
)
if n_queue + n_running == 0:
    print(">> nothing left in queue/running")
else:
    print(f">> estimated remaining ≈ {fmt(eta)}  (around {finish_at.isoformat(timespec='minutes')})")
    if len(durs) < 3:
        print(">> (rough: fewer than 3 finished jobs)")
PY

