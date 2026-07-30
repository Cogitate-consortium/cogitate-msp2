# fMRI Experiment 2 — Analysis Pipeline

Code for COGITATE fMRI Experiment 2 (ses-V2): logfile processing, GLM/FEAT analyses (activation, gPPI, FIR), EVC localizer ROIs, and decoding.

Run pipelines **in the order below**. Later steps depend on outputs from earlier ones.

---

## General requirements

### Compute environment

- **Linux cluster** with **SLURM** (`sbatch`, job arrays)
- **FSL 6.x** (`feat`, `fslmeants`, `fslstats`, `fslmaths`, …) — load before running FEAT steps, e.g. `module load FSL`
- **Python 3.7+** with `pandas`, `numpy` (most scripts); gPPI Bayes Factor steps also need `nibabel`, `pingouin`; decoding additionally needs scikit-learn, nilearn, MNE, etc. (see [decoding/readme.md](decoding/readme.md))



### Upstream data (not produced by this repo)


| Resource                   | Default path                                            |
| -------------------------- | ------------------------------------------------------- |
| BIDS dataset               | `/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids` |
| fMRIPrep derivatives       | `{BIDS_ROOT}/derivatives/fmriprep`                      |
| FreeSurfer derivatives     | `{BIDS_ROOT}/derivatives/freesurfer`                    |
| Raw behavioural log files  | `{RAW_DIR}` (see logfiles script)                       |
| Subject inclusion CSV      | `{CODE_PATH}/ses-v2-analysis-subs-fmri.csv`             |
| FFA/LOC seed masks (Exp 1) | `{BIDS_ROOT}/derivatives/gppi_seeds/sub-{code}/`        |
| Theory ROI masks (gPPI BF) | `{BIDS_ROOT}/derivatives/masks/ICBM2009c_asym_nlin` (override with `$MASKS_PATH`) |




### Path variables

Most scripts accept overrides via environment:

```bash
export BIDS_ROOT=/path/to/bids
export CODE_PATH="${BIDS_ROOT}/code"
```

Clone or sync this repository to `{CODE_PATH}` on your cluster.

### Shared outputs


| Location                                       | Contents                                              |
| ---------------------------------------------- | ----------------------------------------------------- |
| `{BIDS_ROOT}/derivatives/regressoreventfiles/` | FSL 3-column event files and confounds                |
| `{BIDS_ROOT}/derivatives/fslFeat/`             | FEAT outputs (`activation/`, `gPPI/`, `FIR/`, `EVC/`) |
| `{BIDS_ROOT}/derivatives/bf/`                  | gPPI group Bayes Factor maps and ROI summary CSVs     |
| `{CODE_PATH}/<pipeline>/fsf_files/`            | Generated FSF design files                            |
| `{CODE_PATH}/<pipeline>/job_files/`            | SLURM submission markers and logs                     |




### QC utilities (optional, after FEAT)

```bash
cd glm
bash submit_check_FSLFeat.sh              # check all FEAT folders
bash submit_check_FSLFeat.sh --level 1st  # 1st level only
```

From the repo root:

```bash
python generate_final_report.py
```

Aggregates 3rd-level inclusion/failure reports for activation, FIR, and gPPI.

---



## 1. Logfiles and checks (`logfiles_and_checks/`)

**Purpose:** Extract BIDS event TSVs and FSL event files from behavioural logs; run log QC.

**Extra dependencies:** Raw Exp 2 log files under `RAW_DIR`; subject list in `ses-v2-analysis-subs-fmri.csv`.

```bash
cd "${CODE_PATH}/logfiles_and_checks"

# All subjects (SLURM): input = raw logs, SUBJECT_CSV
# Output = BIDS events, regressoreventfiles/*_EV.txt, logfilechecks QC CSVs
bash 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh

# Single subject
python3 exp2_fMRI_logfile_extraction_and_checks.py --sub-code SC108
```

**Main outputs:**

- `{BIDS_ROOT}/sub-{code}/ses-V2/func/sub-{code}_ses-V2_task-{VG|VGR}_run-{N}_events.tsv`
- `{BIDS_ROOT}/derivatives/regressoreventfiles/sub-{code}/ses-V2/*_EV.txt`
- `{BIDS_ROOT}/derivatives/logfilechecks/ses-V2/sub-{code}_ses-mri02_errorFlags.csv`

---



## 2. EVC localizer (`EVC/`)

**Purpose:** 1st-level FEAT on the EVC localizer; build subject-specific V1/V2 (300-voxel) ROIs in MNI space.

**Extra dependencies:** EVC localizer log files; fMRIPrep BOLD/T1w; FreeSurfer labels. Subjects filtered on `SYNCHRONY_min_seen == TRUE` (`evc_config.sh`).

```bash
cd "${CODE_PATH}/EVC"

# 1. Input: EVC localizer logs → Output: BIDS events TSV per subject
python3 01_create_events_tsv.py

# 2. Input: events TSV, fMRIPrep confounds → Output: FSL EV txt + confounds
python3 02_create_regressor_txt_files.py

# 3. Input: templates, regressors → Output: EVC/fsf_files/1st_level/*.fsf
bash 03_make_fsf_files.sh

# 4. Input: FSF, fMRIPrep BOLD → Output: fslFeat/EVC/sub-{code}/*.feat
bash 04_run_EVC_1st_level.sh

# 5. Input: FreeSurfer V1/V2 labels, T1w → Output: evc_rois/sub-{code}/anat/*_V1V2.nii.gz
#    (can run in parallel with step 4)
bash 05_make_anat_masks.sh

# 6. Input: 1st-level FEAT + anat masks → Output: evc_rois/sub-{code}/*_evc_300_V1V2.nii.gz
bash 06_make_evc_roi.sh

# 7. Input: above outputs → Output: EVC/inclusion report/evc_roi_inclusion_report.txt
bash 07_review_EVC.sh
```

---



## 3. GLM — shared 1st level (`glm/`)

**Purpose:** Run session-V2 VG **1st-level GLM** FEAT (MNI space, registration off). These runs feed the **activation** 2nd-level pipeline.

**Extra dependencies:** Event files from §1; fMRIPrep preprocessed BOLD.

```bash
cd "${CODE_PATH}/glm"

# 1. Input: fMRIPrep *_desc-confounds_timeseries.tsv
#    Output: regressoreventfiles/.../confound_event_files/*_confounds.txt
python3 01_create_confound_regressor_ev_file.py

# 2. Input: glm/fsf_templates/1st_level/, BOLD, EV + confound files
#    Output: glm/fsf_files/1st_level/*.fsf
#            fslFeat/activation/sub-{code}/..._analysis-1stGLM_space-MNI152NLin2009cAsym.feat
module load FSL
python3 02_run_fsf_feat_analyses.py
```

Set `submit_jobs = False` in `02_run_fsf_feat_analyses.py` to generate FSFs only.

**Shared helper:** `glm/identity_reg_for_2nd_lvl.sh` — creates dummy `reg/` in 1st-level `.feat` dirs (used by activation, gPPI, and FIR before 2nd level).

---



## 4. Activation (`activation/`)

**Purpose:** 2nd- and 3rd-level FEAT for **seen face vs unseen face** and **seen object vs unseen object** contrasts.

**Extra dependencies:** Completed 1st-level GLM from §3; subject flags `ACTIVATION_min_sf_uf` / `ACTIVATION_min_so_uo` in `ses-v2-analysis-subs-fmri.csv`.

```bash
cd "${CODE_PATH}/activation"

# 1. Input: 1st-level .feat under fslFeat/activation/ → Output: dummy reg/ per run
bash 02_create_identity_reg_for_2nd_lvl.sh

# 2. Input: 1st-level FEAT → Output: activation/fsf_files/2nd_level/*.fsf
bash 03_make_fsf_files.sh

# 3. Input: 2nd-level FSFs → Output: fslFeat/activation/sub-{code}/*_analysis-2ndGLM_*.gfeat
bash 04_run_2nd_level.sh

# 4. Input: 2nd-level .gfeat (cope1=face, cope2=object)
#    Output: activation/fsf_files/group/*.fsf, inclusion report
bash 06_make_fsf_files_3rdlevel.sh

# 5. Input: 3rd-level FSFs → Output: fslFeat/activation/group/seen_vs_unseen_{F,O}.gfeat
bash 07_run_3rd_level.sh

# 6. Input: group threshold maps → Output: activation/results/overlap_O_and_F
bash 08_overlap.sh

# Optional: 3rd-level inclusion report → activation/3rd level report/
bash make_3rd_level_inclusion_report.sh
```

`05_create_identity_reg_for_3rd_lvl.sh` is only needed if reverting to legacy parent-`.gfeat` 3rd-level inputs.

---



## 5. gPPI (`gPPI_code/`)

**Purpose:** Generalized psychophysiological interaction (gPPI) with **FFA** and **LOC** seeds; group-level maps for `PPI_FFA` (cope 3) and `PPI_LOC` (cope 4).

**Extra dependencies:** Event files from §1; seed masks in `gppi_seeds/` (Experiment 1); fMRIPrep MNI BOLD; EVC ROIs from §2 (optional follow-up only). Subjects: `SYNCHRONY_min_seen == TRUE`.

```bash
cd "${CODE_PATH}/gPPI_code"

# 1. Input: seen face/object EV files → Output: *_prefer_Face.txt, *_prefer_Object.txt
bash 01_create_ev_files.sh

# 2. Input: seed masks, fMRIPrep BOLD → Output: gppi_timecourse/sub-{code}/*_timecourse_{FFA,LOC}.txt
bash 02_make_timecourse_files.sh

# 3. Input: templates, EVs, timecourses, confounds → Output: gPPI_code/fsf_files/1st_level/*.fsf
bash 03_make_fsf_files.sh

# 4. Input: 1st-level FSFs → Output: fslFeat/gPPI/sub-{code}/*_analysis-1stGLM_PPI_{FFA,LOC}*.feat
bash 04_run_PPI_1st_level.sh

# 5. Input: 1st-level PPI .feat → Output: dummy reg/ per run
bash 05_create_identity_reg_for_2nd_lvl.sh

# 6. Input: 1st-level FEAT → Output: gPPI_code/fsf_files/2nd_level/*.fsf
bash 06_make_fsf_files_2ndlevel.sh

# 7. Input: 2nd-level FSFs → Output: fslFeat/gPPI/sub-{code}/*_analysis-2ndGLM_PPI_*.gfeat
bash 07_run_PPI_2nd_level_VG.sh

# 8. Input: 2nd-level .gfeat → Output: dummy reg/ at 2nd-level roots
bash 08_create_identity_reg_for_3rd_lvl.sh

# 9. Input: 2nd-level copes 3 (FFA) / 4 (LOC) → Output: gPPI_code/fsf_files/group/*.fsf
bash 09_make_fsf_files_3rdlevel.sh

# 10. Input: 3rd-level FSFs → Output: fslFeat/gPPI/group/*.gfeat
bash 10_run_PPI_3rd_level.sh

# 11. ROI means in V1/V2 → derivatives/gppi/gppi_evc_roi_means_*.csv
bash run_average_gppi_in_evc.sh

# 12. Voxel-wise Bayes Factor maps from 3rd-level gPPI t-stats
#     Input: fslFeat/gPPI/group/N*_..._PPI_{FFA,LOC}_..._desc-cope{3,4}.gfeat
#            (tstat1, dof, mask under cope1.feat/)
#     Output: derivatives/bf/{FFA,LOC}/{bf10,log10_bf10,bf01,log10_bf01}.nii.gz
python3 12_gppi_group_analysis_BF.py

# 13. ROI summary tables from BF maps (needs roi_definitions.py + theory masks)
#     Input: derivatives/bf/{FFA,LOC}/*.nii.gz
#            derivatives/masks/ICBM2009c_asym_nlin/*bh_{roi}_space-*.nii.gz
#     Output: derivatives/bf/{FFA,LOC}_{bf10,bf01}_*_roi_summary.csv
python3 13_gppi_group_level_tables_BF.py

# Optional: 3rd-level inclusion report → gPPI_code/3rd level report/
bash make_3rd_level_inclusion_report.sh
```

Steps **12–13** convert group FEAT one-sample t-maps to JZS Bayes Factors (Pingouin) and summarize `% voxels with BF > 3` per theory ROI (GNW / IIT). Override paths with `$BIDS_ROOT` / `$MASKS_PATH` if needed.

**Optional EVC follow-up:**

```bash
cd "${CODE_PATH}/gPPI_code"

# QC 2nd-level maps before/at 3rd level → gPPI_code/3rd level report/inspect_*
bash inspect_2nd_level_for_3rd.sh

# Copy 2nd-level stat maps → derivatives/gppi/2nd_lvl/
bash run_copy_gppi_2nd_lvl_for_evc.sh
```

---



## 6. FIR (`FIR/`)

**Purpose:** Finite impulse response (FIR) models for seen vs unseen probes; 2nd/3rd-level FEAT and ROI averaging in FFA/LOC.

**Extra dependencies:** Event files from §1; FFA/LOC seed masks (`gppi_seeds/`). Subjects: `BASELINE_min_seen_unseen == TRUE`.

```bash
cd "${CODE_PATH}/FIR"

# 1. Input: per-trial EV files → Output: FIR/*_probed{Seen,Unseen}_Shifted.txt
bash 01_create_ev_files.sh

# 2. Input: FIR templates, shifted EVs → Output: FIR/fsf_files/1st_level/*.fsf
bash 02_make_fsf_files.sh

# 3. Input: 1st-level FSFs → Output: fslFeat/FIR/sub-{code}/*_analysis-1stGLM_FIR*.feat
bash 03_run_FIR_1st_level.sh

# 4. Input: 1st-level .feat → Output: dummy reg/ per run
bash 04_create_identity_reg_for_2nd_lvl.sh

# 5. Input: 1st-level FEAT → Output: FIR/fsf_files/2nd_level/*.fsf
bash 05_make_fsf_files_2nd_lvl.sh

# 6. Input: 2nd-level FSFs → Output: fslFeat/FIR/sub-{code}/*_analysis-2ndGLM_FIR.gfeat
bash 06_run_FIR_2nd_level_VG.sh

# 7. Input: 2nd-level .gfeat → Output: dummy reg/ at 2nd-level roots
bash 07_create_identity_reg_for_3rd_lvl.sh

# 8. Input: 2nd-level copes 1–42 → Output: FIR/fsf_files/group/*.fsf
bash 08_make_fsf_files_3rdlevel.sh

# 9. Input: 3rd-level FSFs → Output: fslFeat/FIR/group/*.gfeat
bash 09_run_FIR_3rd_level.sh

# 10. Input: 2nd-level .gfeat, seed masks → Output: derivatives/fir/*.csv
#     (can run after step 6; 3rd level not required)
bash 10_average_FIR_in_FFA_and_LOC.sh
```

**Optional reports**

```bash
bash make_fir_roi_inclusion_report.sh
bash make_fir_3rd_level_inclusion_report.sh
```

---



## 7. Decoding (`decoding/`)

**Purpose:** Trial-level beta-series extraction (nibetaseries), ROI and searchlight decoding, group inference, RSA, and plotting.

**Extra dependencies:** fMRIPrep derivatives; nibetaseries (≥ 0.6), Nilearn, MNE-Python, scikit-learn, Singularity; BIDS events from §1; ROI definitions.

```bash
cd "${CODE_PATH}/decoding"

# 1. Beta-series extraction
#    Input: fMRIPrep BOLD, events → Output: derivatives/nibetaseries/
sbatch exp2_nibetaseries_eye_noSD193.sh   # with eye-tracking (excl. SD193)
sbatch exp2_nibetaseries_no_eye.sh        # subjects without eye-tracking

# 2. ROI decoding (subject-level; examples)
sbatch Exp2_roi_category.sh --condition=seen-seen --vg_or_replay=VG-Replay
sbatch Exp2_roi_location_subsample.sh --condition=seen

# 3. ROI group-level stats
python Exp2_roi_generalization_group.py
python Exp2_roi_subsample_group.py
python Exp2_roi_subsampleDiff_group.py

# 4. Searchlight decoding (subject-level; examples)
python Exp2_searchlight_category_callSlurm.py --condition seen --vg_or_replay Replay
python Exp2_searchlight_location_subsample_callSlurm.py --condition seen

# 5. Searchlight group-level (examples)
sbatch Exp2_searchlight_group.sh --decoding_problem=category --condition=seen --vg_or_replay=Replay
sbatch Exp2_searchlight_subsample_group.sh --decoding_problem=location --condition=unseen

# 6. RSA — Replay (examples)
python Exp2_searchlight_RSA_Replay_category_callSlurm.py
sbatch Exp2_searchlight_RSA_Replay_group.sh --decoding_problem=category

# 7. Visualization
python Exp2_roi_plots.py
python Exp2_roi_plots_subsample.py
python Exp2_searchlight_plots.py
```

**Full script list, parameters, and additional example commands:** see [decoding/readme.md](decoding/readme.md).

---



## Repository layout

```
fMRI_exp2/
├── logfiles_and_checks/   # Behavioural log extraction and QC
├── EVC/                   # EVC localizer → V1/V2 ROIs
├── glm/                   # Shared VG 1st-level GLM + FEAT QC
├── activation/            # Seen vs unseen face/object GLM
├── gPPI_code/             # FFA/LOC psychophysiological interaction
├── FIR/                   # FIR timecourse models
├── decoding/              # nibetaseries decoding (see decoding/readme.md)
├── MNI_standard_space/    # Reference templates
├── generate_final_report.py
├── delete_identity_reg.py
└── delete_fslfeat_derivatives.py
```

---



## Notes

- Scripts use **absolute cluster paths** by default; override with `BIDS_ROOT` and `CODE_PATH` when running elsewhere.
- SLURM submission scripts skip work when a `.run` marker already exists; use `--force` where supported to resubmit.
- Subject inclusion differs by analysis (`SYNCHRONY_min_seen`, `BASELINE_min_seen_unseen`, `ACTIVATION_min_*`, etc.) — see `ses-v2-analysis-subs-fmri.csv` and each pipeline’s filter column.

