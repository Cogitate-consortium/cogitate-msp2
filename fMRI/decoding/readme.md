# fMRI Decoding Pipeline (Experiment 2)

This repository contains scripts for **trial-level fMRI decoding analyses** in **Experiment 2** of the COGITATE project.  
The pipeline covers:  

- Beta-series extraction  
- ROI-based decoding (category, location, Go/NoGo)  
- Searchlight decoding (category, location, subsampling, Go/NoGo, probe, seen/unseen)  
- Representational Similarity Analysis (RSA) for Replay (category and location)  
- Group-level statistical testing  
- Visualization of ROI and searchlight results  

---

## Contents

1. [Scripts Overview](#scripts-overview) – descriptions of all scripts  
2. [Execution Order](#execution-order) – what to run, who calls who, and examples  
3. [Dependencies](#dependencies)  
4. [Quick Start: Commands in Order](#quick-start-commands-in-order)

---

## Scripts Overview

Below is a full list of scripts in this repository, organized by function, with a short description of what each script does.

### Configuration & Utilities
- **`config.py`** — Global visualization configuration (fonts, colors, figure sizes).  
- **`plotters.py`** — Utility functions for plotting cortical surfaces.  

---

### Beta-series (nibetaseries)
- **`exp2_nibetaseries_eye_noSD193.sh`** — Run *nibetaseries* with eye-tracking regressors (SLURM wrapper). Excludes subject SD193.
- **`exp2_nibetaseries_no_eye.sh`** — Run *nibetaseries* for subjects **without eye-tracking data**.  

---

### ROI – Subject-level
- **`Exp2_roi_category.sh`** — SLURM wrapper for ROI **category** decoding.  
- **`Exp2_roi_category_generalization_forSlurm.py`** — ROI category decoding and VG↔Replay **generalization**.  
- **`Exp2_roi_category_subsample.sh`** — SLURM wrapper for ROI **category** decoding with **subsampling**.  
- **`Exp2_roi_category_subsample_forSlurm.py`** — ROI category decoding with **subsampling** (trial balancing).  
- **`Exp2_roi_location_generalization_forSlurm.py`** — ROI **location** decoding (Left vs Right) with VG↔Replay generalization.  
- **`Exp2_roi_location_subsample.sh`** — SLURM wrapper for ROI **location** decoding with subsampling.  
- **`Exp2_roi_location_subsample_forSlurm.py`** — ROI location decoding with **subsampling**.  

---

### ROI – Group-level
- **`Exp2_roi_generalization_group.py`** — Group-level stats for ROI decoding/generalization (permutation + FDR).  
- **`Exp2_roi_subsample_group.py`** — Group-level ROI stats for **subsampled** decoding.  
- **`Exp2_roi_subsampleDiff_group.py`** — Group-level ROI difference analysis (e.g., **seen vs unseen**).  

---

### ROI – Visualization
- **`Exp2_roi_plots.py`** — Plots ROI decoding results (category/location).  
- **`Exp2_roi_plots_subsample.py`** — Plots ROI decoding results for **subsampled** analyses.  

---

### Searchlight – Orchestrators
*(Run these when available instead of calling `.sh` manually — they submit jobs for all subjects.)*  
- **`Exp2_searchlight_category_callSlurm.py`** — Submit per-subject **category** searchlight decoding jobs.  
- **`Exp2_searchlight_category_subsample_callSlurm.py`** — Submit per-subject **category (subsampled)** searchlight jobs.  
- **`Exp2_searchlight_location_callSlurm.py`** — Submit per-subject **location** searchlight jobs.  
- **`Exp2_searchlight_location_subsample_callSlurm.py`** — Submit per-subject **location (subsampled)** searchlight jobs.  
- **`Exp2_searchlight_goNogo_callSlurm.py`** — Submit per-subject **Go/NoGo** searchlight decoding jobs.  
- **`Exp2_searchlight_probe_callSlurm.py`** — Submit per-subject **Probe vs No-Probe** searchlight decoding jobs.  
- **`Exp2_searchlight_probe_group_callSlurm.py`** — Submit **group-level Probe** searchlight jobs.  
- **`Exp2_searchlight_seenUnseen_callSlurm.py`** — Submit per-subject **Seen vs Unseen** searchlight jobs.  

---

### Searchlight – Subject-level
- **`Exp2_searchlight_category.sh`** — SLURM wrapper for **category** searchlight decoding (single subject).  
- **`Exp2_searchlight_category_forSlurm.py`** — Worker script for category decoding (subject-level).  
- **`Exp2_searchlight_category_subsample.sh`** — SLURM wrapper for **category (subsampled)** decoding.  
- **`Exp2_searchlight_category_subsample_forSlurm.py`** — Worker script for **category (subsampled)** decoding.  
- **`Exp2_searchlight_location.sh`** — SLURM wrapper for **location** searchlight decoding.  
- **`Exp2_searchlight_location_forSlurm.py`** — Worker script for location decoding.  
- **`Exp2_searchlight_location_subsample_forSlurm.py`** — Worker script for **location (subsampled)** decoding.  
- **`Exp2_searchlight_goNogo.sh`** — SLURM wrapper for **Go/NoGo** searchlight decoding.  
- **`Exp2_searchlight_goNogo_forSlurm.py`** — Worker script for Go/NoGo decoding.  
- **`Exp2_searchlight_probe.sh`** — SLURM wrapper for **Probe vs No-Probe** decoding.  
- **`Exp2_searchlight_probe_forSlurm.py`** — Worker script for Probe decoding.  
- **`Exp2_searchlight_seenUnseen.sh`** — SLURM wrapper for **Seen vs Unseen** decoding.  
- **`Exp2_searchlight_seenUnseen_forSlurm.py`** — Worker script for Seen vs Unseen decoding.  
- **`Exp2_searchlight_seen-unseen.sh`** — SLURM wrapper for **Seen–Unseen (subsampled)** decoding.  
- **`Exp2_searchlight_seen-unseen_forSlurm.py`** — Worker script for Seen–Unseen (subsampled) decoding.  

---

### Searchlight – Group-level
- **`Exp2_searchlight_group.sh`** — SLURM wrapper for **group-level** searchlight inference (category/location).  
- **`Exp2_searchlight_group_forSlurm.py`** — Group-level non-parametric/cluster-based inference.  
- **`Exp2_searchlight_subsample_group.sh`** — SLURM wrapper for **group-level subsampled** inference.  
- **`Exp2_searchlight_subsample_group_forSlurm.py`** — Group-level inference for subsampled maps.  
- **`Exp2_searchlight_goNogo_group.sh`** — SLURM wrapper for **group-level Go/NoGo** inference.  
- **`Exp2_searchlight_goNogo_group_forSlurm.py`** — Group-level Go/NoGo inference.  
- **`Exp2_searchlight_probe_group.sh`** — SLURM wrapper for **group-level Probe** inference.  
- **`Exp2_searchlight_probe_group_forSlurm.py`** — Group-level Probe inference.  
- **`Exp2_searchlight_seenUnseen_group.sh`** — SLURM wrapper for **group-level Seen vs Unseen** inference.  
- **`Exp2_searchlight_seenUnseen_group_forSlurm.py`** — Group-level Seen vs Unseen inference.  
- **`Exp2_searchlight_seen-unseen_group.sh`** — SLURM wrapper for **group-level Seen–Unseen (subsampled)** inference.  
- **`Exp2_searchlight_seen-unseen_group_forSlurm.py`** — Group-level Seen–Unseen (subsampled) inference.  

---

### RSA (Replay)
- **`Exp2_searchlight_RSA_Replay_category.sh`** — SLURM wrapper for **Replay RSA (category)** subject-level.  
- **`Exp2_searchlight_RSA_Replay_category_callSlurm.py`** — Submit jobs across subjects for category RSA.  
- **`Exp2_searchlight_RSA_Replay_category_forSlurm.py`** — Worker script for category RSA (subject-level).  
- **`Exp2_searchlight_RSA_Replay_location.sh`** — SLURM wrapper for **Replay RSA (location)** subject-level.  
- **`Exp2_searchlight_RSA_Replay_location_callSlurm.py`** — Submit jobs across subjects for location RSA.  
- **`Exp2_searchlight_RSA_Replay_location_forSlurm.py`** — Worker script for location RSA (subject-level).  
- **`Exp2_searchlight_RSA_Replay_group.sh`** — SLURM wrapper for **group-level Replay RSA**.  
- **`Exp2_searchlight_RSA_Replay_group_forSlurm.py`** — Group-level Replay RSA inference.  

---

### Searchlight – Visualization
- **`Exp2_searchlight_plots.py`** — Plots searchlight maps on cortical surfaces.

---

## Execution Order

| **Stage** | **What to Run** | **Who Calls Who** | **Example Command** |
|-----------|-----------------|-------------------|----------------------|
| **1. Beta-series extraction** | Run the appropriate SLURM script depending on subject/eye-tracking | SLURM script → `nibetaseries` inside Singularity | With eye-tracking:<br>`sbatch exp2_nibetaseries_eye.sh`<br>Excluding SD193:<br>`sbatch exp2_nibetaseries_eye_noSD193.sh`<br>SD193 only:<br>`sbatch exp2_nibetaseries_eye_SD193.sh`<br>No eye-tracking:<br>`sbatch exp2_nibetaseries_no_eye.sh` |
| **2. ROI decoding (subject-level)** | SLURM wrapper | `Exp2_roi_category.sh` → `Exp2_roi_category_generalization_forSlurm.py`<br>`Exp2_roi_category_subsample.sh` → `Exp2_roi_category_subsample_forSlurm.py`<br>`Exp2_roi_location_subsample.sh` → `Exp2_roi_location_subsample_forSlurm.py` | Category:<br>`sbatch Exp2_roi_category.sh --condition=seen-seen --vg_or_replay=VG-Replay`<br>Location:<br>`sbatch Exp2_roi_location_subsample.sh --condition=seen` |
| **3. ROI group-level analysis** | Run Python script directly | Standalone Python (no SLURM) | `python Exp2_roi_generalization_group.py`<br>`python Exp2_roi_subsample_group.py`<br>`python Exp2_roi_subsampleDiff_group.py` |
| **4. Searchlight decoding (subject-level)** | If a `*_callSlurm.py` exists → run it.<br>If not, run the `.sh` script directly. | Example:<br>`Exp2_searchlight_category_callSlurm.py` → `Exp2_searchlight_category.sh` → `Exp2_searchlight_category_forSlurm.py`<br>`Exp2_searchlight_seen-unseen.sh` → `Exp2_searchlight_seen-unseen_forSlurm.py` | With callSlurm:<br>`python Exp2_searchlight_category_callSlurm.py --condition seen --vg_or_replay VG`<br>No callSlurm:<br>`sbatch Exp2_searchlight_seen-unseen.sh --decoding_problem=category` |
| **5. Searchlight group-level analysis** | Run the SLURM group script (unless a `*_callSlurm.py` exists) | Example:<br>`Exp2_searchlight_group.sh` → `Exp2_searchlight_group_forSlurm.py`<br>`Exp2_searchlight_probe_group.sh` → `Exp2_searchlight_probe_group_forSlurm.py` | General:<br>`sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen --vg_or_replay=Replay`<br>Subsampled:<br>`sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=location --condition=unseen` |
| **6. RSA (Replay)** | Subject-level: run the `*_callSlurm.py` orchestrators.<br>Group-level: run the `.sh` wrapper. | Example:<br>`Exp2_searchlight_RSA_Replay_category_callSlurm.py` → `Exp2_searchlight_RSA_Replay_category.sh` → `Exp2_searchlight_RSA_Replay_category_forSlurm.py`<br>`Exp2_searchlight_RSA_Replay_group.sh` → `Exp2_searchlight_RSA_Replay_group_forSlurm.py` | Subject-level:<br>`python Exp2_searchlight_RSA_Replay_category_callSlurm.py`<br>Group-level:<br>`sbatch Exp2_searchlight_RSA_Replay_group.sh --decoding_problem=location` |
| **7. Visualization** | Run Python script directly | Standalone Python (uses `plotters.py` + `config.py`) | `python Exp2_roi_plots.py`<br>`python Exp2_roi_plots_subsample.py`<br>`python Exp2_searchlight_plots.py` |

---

## Dependencies

- **HPC environment**:  
  - SLURM workload manager  
  - Singularity  
  - Anaconda / Conda  

- **Neuroimaging tools**:  
  - fMRIPrep derivatives  
  - nibetaseries ≥ 0.6.0  ([https://nibetaseries.readthedocs.io/en/stable/](https://nibetaseries.readthedocs.io/en/stable/))
  - Nilearn  
  - MNE-Python  

- **Python libraries**:  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `scipy`  
  - `matplotlib`  
  - `argparse`  
  - `warnings`  

---

## Quick Start: Commands in Order

Here is a **full pipeline example** with the commands to run in sequence.  

```bash
# ----------------------
# 1) Beta-series extraction
# ----------------------
sbatch exp2_nibetaseries_eye_noSD193.sh          # with eye-tracking, excluding SD193
sbatch exp2_nibetaseries_no_eye.sh               # only for subjects without eye-tracking (SD176, SD201, SD156)

# ----------------------
# 2) ROI decoding (subject-level)
# ----------------------
sbatch Exp2_roi_location.sh --condition=seen-seen --vg_or_replay=VG-Replay
sbatch Exp2_roi_location.sh --condition=seen-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_location.sh --condition=seen-seen_go --vg_or_replay=VG-Replay
sbatch Exp2_roi_location.sh --condition=seen_go-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_location.sh --condition=seen-seen_nogo --vg_or_replay=VG-Replay
sbatch Exp2_roi_location.sh --condition=seen_nogo-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_category.sh --condition=seen-seen --vg_or_replay=VG-Replay
sbatch Exp2_roi_category.sh --condition=seen-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_category.sh --condition=seen-seen_go --vg_or_replay=VG-Replay
sbatch Exp2_roi_category.sh --condition=seen_go-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_category.sh --condition=seen-seen_nogo --vg_or_replay=VG-Replay
sbatch Exp2_roi_category.sh --condition=seen_nogo-seen --vg_or_replay=Replay-VG
sbatch Exp2_roi_location_subsample.sh --condition=seen
sbatch Exp2_roi_location_subsample.sh --condition=unseen
sbatch Exp2_roi_category_subsample.sh --condition=seen
sbatch Exp2_roi_category_subsample.sh --condition=unseen

# ----------------------
# 3) ROI group-level (set conditions and vg/replay flags within the Python scripts)
# ----------------------
python Exp2_roi_generalization_group.py
python Exp2_roi_subsample_group.py
python Exp2_roi_subsampleDiff_group.py

# ----------------------
# 4) Searchlight decoding (subject-level)
# ----------------------
python Exp2_searchlight_location_callSlurm.py --condition seen --vg_or_replay Replay
python Exp2_searchlight_category_callSlurm.py --condition seen --vg_or_replay Replay
python Exp2_searchlight_location_callSlurm.py --condition seen_go --vg_or_replay Replay
python Exp2_searchlight_category_callSlurm.py --condition seen_go --vg_or_replay Replay
python Exp2_searchlight_location_callSlurm.py --condition seen_nogo --vg_or_replay Replay
python Exp2_searchlight_category_callSlurm.py --condition seen_nogo --vg_or_replay Replay
python Exp2_searchlight_location_subsample_callSlurm.py --condition seen
python Exp2_searchlight_location_subsample_callSlurm.py --condition unseen
python Exp2_searchlight_category_subsample_callSlurm.py --condition seen
python Exp2_searchlight_category_subsample_callSlurm.py --condition unseen
python Exp2_searchlight_location_callSlurm.py --condition seen-seen --vg_or_replay Replay-VG
python Exp2_searchlight_location_callSlurm.py --condition seen-seen --vg_or_replay VG-Replay
python Exp2_searchlight_category_callSlurm.py --condition seen-seen --vg_or_replay Replay-VG
python Exp2_searchlight_category_callSlurm.py --condition seen-seen --vg_or_replay VG-Replay
python Exp2_searchlight_location_callSlurm.py --condition seen_go-seen --vg_or_replay Replay-VG
python Exp2_searchlight_location_callSlurm.py --condition seen-seen_go --vg_or_replay VG-Replay
python Exp2_searchlight_category_callSlurm.py --condition seen_go-seen --vg_or_replay Replay-VG
python Exp2_searchlight_category_callSlurm.py --condition seen-seen_go --vg_or_replay VG-Replay
python Exp2_searchlight_location_callSlurm.py --condition seen_nogo-seen --vg_or_replay Replay-VG
python Exp2_searchlight_location_callSlurm.py --condition seen-seen_nogo --vg_or_replay VG-Replay
python Exp2_searchlight_category_callSlurm.py --condition seen_nogo-seen --vg_or_replay Replay-VG
python Exp2_searchlight_category_callSlurm.py --condition seen-seen_nogo --vg_or_replay VG-Replay
python Exp2_searchlight_goNogo_callSlurm.py
python Exp2_searchlight_probe_callSlurm.py --condition seen
python Exp2_searchlight_seenUnseen_callSlurm.py
sbatch Exp2_searchlight_seen-unseen.sh --decoding_problem=category   # directly run the slurm script
sbatch Exp2_searchlight_seen-unseen.sh --decoding_problem=location   # directly run the slurm script

# ----------------------
# 5) Searchlight group-level
# ----------------------
sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=location --condition=seen --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen_go --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=location --condition=seen_go --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen_nogo --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=location --condition=seen_nogo --vg_or_replay=Replay
sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen_nogo --vg_or_replay=Replay
sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=location --condition=seen
sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=location --condition=unseen
sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=category --condition=seen
sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=category --condition=unseen
sbatch Exp2_searchlight_seen-unseen_group.sh --decoding_problem=location
sbatch Exp2_searchlight_seen-unseen_group.sh --decoding_problem=category
python Exp2_searchlight_group.py --decoding_problem=location --condition seen-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=location --condition seen-seen --vg_or_replay VG-Replay
python Exp2_searchlight_group.py --decoding_problem=category --condition seen-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=category --condition seen-seen --vg_or_replay VG-Replay
python Exp2_searchlight_group.py --decoding_problem=location --condition seen_go-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=location --condition seen-seen_go --vg_or_replay VG-Replay
python Exp2_searchlight_group.py --decoding_problem=category --condition seen_go-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=category --condition seen-seen_go --vg_or_replay VG-Replay
python Exp2_searchlight_group.py --decoding_problem=location --condition seen_nogo-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=location --condition seen-seen_nogo --vg_or_replay VG-Replay
python Exp2_searchlight_group.py --decoding_problem=category --condition seen_nogo-seen --vg_or_replay Replay-VG
python Exp2_searchlight_group.py --decoding_problem=category --condition seen-seen_nogo --vg_or_replay VG-Replay
sbatch Exp2_searchlight_seenUnseen_group.sh
sbatch Exp2_searchlight_goNogo_group.sh
sbatch Exp2_searchlight_probe_group.sh --condition=seen

# ----------------------
# 6) RSA (Replay)
# ----------------------
python Exp2_searchlight_RSA_Replay_category_callSlurm.py
python Exp2_searchlight_RSA_Replay_location_callSlurm.py
sbatch Exp2_searchlight_RSA_Replay_group.sh --decoding_problem=category
sbatch Exp2_searchlight_RSA_Replay_group.sh --decoding_problem=location

# ----------------------
# 7) Visualization
# ----------------------
python Exp2_roi_plots.py
python Exp2_roi_plots_subsample.py
python Exp2_searchlight_plots.py

