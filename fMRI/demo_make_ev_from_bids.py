#!/usr/bin/env python3
"""
Build FSL 3-column EV files for demo subjects from BIDS events.tsv.

Demo data ships events.tsv but not RAW behavioral logs, so logfile extraction
is skipped. This script writes derivatives/regressoreventfiles/ using the same
event naming as logfiles_and_checks/exp2_fMRI_logfile_extraction_and_checks.py.

Requires DEMO paths (source demo_setup.sh first) and ses-V2 aliases.
Also writes EVCLoc TLBR/TRBL/buttonPress EVs when EVCLoc events exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on path for demo_paths / logfile helpers
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "logfiles_and_checks"))

from demo_paths import BIDS_ROOT, SUBJECT_CSV, SESSION, is_demo  # noqa: E402

# Reuse EV builders from the logfile pipeline
import exp2_fMRI_logfile_extraction_and_checks as logmod  # noqa: E402

STIMULUS_DURATION = logmod.stimulusDuration
RELEVANT_LOCATIONS = logmod.relevantLocations
VG_RUNS = range(1, 9)
REPLAY_RUNS = range(1, 5)


def load_demo_subjects() -> list[str]:
    path = SUBJECT_CSV
    df = pd.read_csv(path, sep=None, engine="python")
    if "sub_code" not in df.columns and len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return [str(s) for s in df["sub_code"].tolist()]


def events_path(sub: str, task_run: str) -> Path:
    return (
        BIDS_ROOT
        / f"sub-{sub}"
        / SESSION
        / "func"
        / f"sub-{sub}_{SESSION}_{task_run}_events.tsv"
    )


def output_pattern() -> str:
    return str(
        BIDS_ROOT
        / "derivatives"
        / "regressoreventfiles"
        / "sub-%(sub)s"
        / SESSION
        / f"sub-%(sub)s_{SESSION}_%(runType)s_%(eventType)s_EV.txt"
    )


def write_vg_replay_evs(sub: str, run_type: str, events_tsv: Path) -> None:
    log = pd.read_csv(events_tsv, sep="\t")
    # Ensure expected columns exist
    for col in ("onset", "duration", "type", "stimulusType", "stimulusLocation", "response"):
        if col not in log.columns:
            raise ValueError(f"{events_tsv}: missing column {col}")
    pattern = output_pattern()
    logmod.createEvFile_nullAndEndOfRunEV(log, pattern, sub, run_type)
    logmod.createEvFile_categoryOnly(log, STIMULUS_DURATION, pattern, sub, run_type)
    logmod.createEvFile_categorySide(log, RELEVANT_LOCATIONS, STIMULUS_DURATION, pattern, sub, run_type)
    logmod.createEvFile_seenCategorySide(
        log, RELEVANT_LOCATIONS, STIMULUS_DURATION, pattern, sub, run_type
    )


def write_evcloc_evs(sub: str, events_tsv: Path) -> None:
    """Minimal EVCLoc EVs compatible with EVC/02 naming (TLBR, TRBL, buttonPress)."""
    log = pd.read_csv(events_tsv, sep="\t")
    out_dir = BIDS_ROOT / "derivatives" / "regressoreventfiles" / f"sub-{sub}" / SESSION
    out_dir.mkdir(parents=True, exist_ok=True)
    run_type = "task-EVCLoc_run-1"
    # Map trial_type / type columns flexibly
    event_col = "trial_type" if "trial_type" in log.columns else "type"
    for event_type in ("TLBR", "TRBL", "buttonPress"):
        if event_col in log.columns:
            mask = log[event_col].astype(str) == event_type
        else:
            mask = pd.Series([False] * len(log))
        # Also try type column values containing the label
        if not mask.any() and "type" in log.columns:
            mask = log["type"].astype(str).str.contains(event_type, na=False)
        onsets = log.loc[mask, "onset"].to_numpy(dtype=float) if mask.any() else np.array([])
        if len(onsets) == 0:
            event = np.array([[0.0, 0.0, 0.0]])
        else:
            if "duration" in log.columns:
                durations = log.loc[mask, "duration"].to_numpy(dtype=float)
            else:
                durations = np.full(len(onsets), 15.25)
            parametric = np.ones(len(onsets))
            event = np.vstack((onsets, durations, parametric)).T
        fname = out_dir / f"sub-{sub}_{SESSION}_{run_type}_{event_type}_EV.txt"
        np.savetxt(fname, event, fmt="%f")


def main() -> int:
    if not is_demo() and not os.environ.get("BIDS_ROOT"):
        print(
            "WARNING: DEMO is not set. Using paths from demo_paths "
            f"(BIDS_ROOT={BIDS_ROOT}). Prefer: source demo_setup.sh",
            file=sys.stderr,
        )

    subjects = load_demo_subjects()
    if not subjects:
        print(f"ERROR: no subjects in {SUBJECT_CSV}", file=sys.stderr)
        return 1

    pattern = output_pattern()
    n_ok = 0
    for sub in subjects:
        sub_dir = BIDS_ROOT / f"sub-{sub}" / SESSION / "func"
        if not sub_dir.is_dir():
            print(f"[{sub}] SKIP — missing {sub_dir} (run demo_setup.sh to create ses-V2 aliases)")
            continue

        for run in VG_RUNS:
            run_type = f"task-VG_run-{run}"
            tsv = events_path(sub, run_type)
            if not tsv.is_file():
                print(f"[{sub}] missing {tsv.name}")
                continue
            write_vg_replay_evs(sub, run_type, tsv)
            print(f"[{sub}] wrote VG run-{run} EVs")
            n_ok += 1

        for run in REPLAY_RUNS:
            run_type = f"task-Replay_run-{run}"
            tsv = events_path(sub, run_type)
            if not tsv.is_file():
                continue
            write_vg_replay_evs(sub, run_type, tsv)
            print(f"[{sub}] wrote Replay run-{run} EVs")
            n_ok += 1

        evc_tsv = events_path(sub, "task-EVCLoc_run-1")
        if evc_tsv.is_file():
            write_evcloc_evs(sub, evc_tsv)
            print(f"[{sub}] wrote EVCLoc EVs")
            n_ok += 1

    print(f">> done ({n_ok} run files processed); pattern={pattern}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
