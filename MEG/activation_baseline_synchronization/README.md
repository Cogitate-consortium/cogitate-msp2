# MEG Source-Level Activation, Baseline & Synchronization

MEG source-level **activation**, **baseline**, and **synchronization** analyses
(COGITATE Experiment 2). This folder is one part of the larger project and covers
only these MEG analyses:

- **Activation** — source-level evoked (ERF) and oscillatory (TFR) responses;
- **Baseline** — contralateral-alpha power;
- **Synchronization** — PPC / DFC connectivity;

together with the **vertex-selection** (IIT VertROI) and **GED spatial-filter**
steps that these analyses build on.

> Statistical methods used throughout: cluster-based permutation tests (CBPT,
> Maris & Oostenveld 2007) and Bayesian paired *t*-tests (BF₁₀ / BF₀₁, JZS prior
> r = 0.707; `bayes_factor_fun.bayes_ttest`). 
---

## 0. Paths & setup (read this first)

Only **one** path must be set by hand — everything else is derived from it or from
the repository's own location, so the code runs wherever you clone it.

### The one knob you set: `bids_root`
Edit `py_code/EXP1_config.py` and point `bids_root` at your local copy of the
COGITATE MEG BIDS dataset (external raw data, **not** shipped with this repo):

```python
bids_root = r"/path/to/your/COGITATE/MEG/phase_2/processed/bids"
```

Until you set it, `bids_root = None` and importing the config raises a clear
`ValueError`, so the pipelines fail fast instead of silently running on a wrong path.

### Two roots, one manual — the rest is derived

| Name | Defined in | Value | Meaning |
|------|-----------|-------|---------|
| **`bids_root`** | `EXP1_config.py` | *(you set it)* | root of the external COGITATE BIDS dataset — raw data, QC tables, and all `derivatives/` |
| **`Project_Dir`** | `_help_functions.py` | auto (`<this file>/../..`) | **this repo's** root (holds `py_code/` + `config_files/`); found from the file location |
| `path_allana` | `_help_functions.py` | `bids_root/derivatives/ana` | where **all pipeline outputs** are written & read (one subfolder per `pipeline_*`) |
| `path_group_result` | `_help_functions.py` | `bids_root/derivatives/ana_group` | group-level result store |
| QC csvs | `_help_functions.py` | `bids_root/derivatives/qcs/…` | behavioural / MEG trial-count tables used to build subject lists |
| config paths | every script | `Project_Dir/config_files/<pipeline>/<cfg>.json` | per-analysis JSON configs, resolved relative to the repo |

In short: **`bids_root` → data** (external, you set it); **`Project_Dir` → code + configs**
(this repo, automatic).

### Output layout under `path_allana` (= `bids_root/derivatives/ana`)

```
derivatives/ana/
├── pipeline_<name>/<epoch_info>/    # per-pipeline outputs, keyed by epoch preset
│   └── sub-<ID>_…_<compare>.pkl     # one cached .pkl per subject
├── pipeline_GED/<ged_subfolder>/    # GED spatial filters (e.g. noft_nobc_deci1_nocut)
└── …
```

GED filters are addressed by **`ged_subfolder`** — just the run-folder name (e.g.
`"noft_nobc_deci1_nocut"`) stored in `general_config.json` →
`GED_params_dict[<epochtype>][<GED_type>]["ged_subfolder"]`. The loader rebuilds the
full path as `path_allana/pipeline_GED/<ged_subfolder>`, so it too follows `bids_root`
automatically (no absolute paths in the JSON).

---

## 1. Repository layout

```
activation_baseline_synchronization/
├── py_code/                 # all analysis scripts + helper libraries
│   ├── pipeline_*.py        # the analysis pipelines (see §5)
│   ├── general_config.json  # master parameter dictionary  -> general_param (§3)
│   ├── _help_functions.py   # core pipeline utilities + general_param loader
│   ├── _help_module.py      # plotting + trial/vertex reduction utilities
│   ├── EXP1_help_functions.py  # low-level MEG/BIDS/GED helpers (from Experiment 1)
│   ├── EXP1_config.py        # study parameter/config module (from Experiment 1)
│   └── bayes_factor_fun.py   # Bayesian t-test / Bayes-factor functions (from Experiment 1)
└── config_files/            # per-analysis JSON configs, one folder per pipeline (§3)
```

---

## 2. How the pipelines are organized

Almost every analysis is a **3-stage chain**:

| Stage | Suffix | Runs over | What it does |
|-------|--------|-----------|--------------|
| Individual | *(none)* | one **subject** | compute & save per-subject result (`.pkl`) |
| Group | `_group…` | all subjects | aggregate + run statistics (CBPT / Bayesian) |
| Plot | `…_plot` | all subjects | reload results and render figures (jpg / pdf) |

**Run entry points** (command-line, via `argparse`):

- **Individual scripts** take:
  ```bash
  python pipeline_<name>.py --subject SA108 \
      --configfile /…/config_files/pipeline_<name>/<config>.json
  ```
- **Group / plot scripts** take an *individual* + a *group* config pair:
  ```bash
  python pipeline_<name>_group.py \
      --individual_ana pipeline_<name> \
      --individual_configfolder pipeline_<name> \
      --individual_configfile   /…/config_files/pipeline_<name>/<indiv>.json \
      --group_ana        pipeline_<name>_group \
      --group_configfolder pipeline_<name>_group_<window> \
      --group_configfile   /…/config_files/pipeline_<name>_group_<window>/<group>.json
  ```
- A few scripts run **directly / in batch** (no argparse): they hardcode a config
  path or loop over a config folder — noted individually in §5
  (`pipeline_IIT_vertices_morph_group.py`,
  `pipeline_IIT_vertices_plot.py`,
  `pipeline_syn_group_plot_…logscale.py`).

Each pipeline caches its output and **skips** work if the result file already
exists, so runs are resumable.

---

## 3. Configuration

### `general_config.json` → `general_param`
The single master dictionary, loaded once by `_help_functions.py` into the global
`general_param`. Every pipeline reads its shared settings from here. Top-level keys:

| Key | Contents |
|-----|----------|
| `sub_no_T1` | subjects without a T1 (use fsaverage / identity morph) |
| `source_method_setting` | inverse-solution settings (method, SNR, etc.) |
| `epoch_setting_dict` | epoch preprocessing presets (filter / baseline / crop / decimation) |
| `epoch_data_dict` | task/session/data descriptors (`exp_id`, `task_id`, …) |
| `roi_params_dict` | anatomical ROIs (parc, labels_list, rois_list) |
| `VertROI_params_dict` | IIT vertex-ROIs (category-selective / V1–V2 vertices) |
| `GED_params_dict` | GED contrasts + saved-filter locations per epoch/GED type |
| `timecourse_data_dict` | ROI time-course extraction settings |
| `band_map` | frequency-band definitions (e.g. alpha 8–13 Hz) + Morlet settings |
| `tfr_map_dict` | multitaper TFR settings per method / freq range (1–30, 30–100 Hz) |
| `cbpt_params_dict` | cluster-based permutation-test parameters |
| `Nsample` | subsampling count for trial balancing |
| `MI_method_dict` | Gaussian-Copula MI settings for DFC |

### Per-analysis configs: `config_files/<pipeline_name>/*.json`
Each pipeline has its own config folder. **Individual** configs pick the epoch
setting, data, ROI, conditions, TFR method, etc.; **group** configs pick the time
windows, tail, statistic parameters, and which subjects/files to include.
Naming convention encodes preprocessing, e.g.
`dAT_probe_stim_noft_nobc_deci5_nocut_PFC_30_100.json`
= dAT probe / stimulus trials / no filter / no baseline-correct / decimate 5 /
no cut / ROI PFC / 30–100 Hz. The exact config files are listed in each script's
head docstring.

---

## 4. Helper / library modules

| File | Purpose |
|------|---------|
| **`_help_functions.py`** | Core pipeline utilities and the `general_param` loader. Epoch prep (`get_eps_use`, `apply_*_to_epochs`, `get_eps_use_info`), source modeling (`get_fw_inv`, `get_stcs_allsrc`, `get_stcs_of_label`), labels/vertices (`read_labels_exp2`, `get_label_vertidx`), subject lists (`get_sublist`, `group_get_sublist`), trial selection (`get_trl_indices`), config (`read_configfile`), stats/CI helpers, `roi_longname_dict`, `turn_string_to_num`. |
| **`_help_module.py`** | Higher-level plotting and reduction. ERF/diff plotting (`plot_erf_erfdiff`, `plot_erf_reduce_and_save`, `subplot_erf_and_diff_reduce_and_save`), trial-distribution plots, `SourceROIEpochs` + `compute_all_combinations(_three)` (trial/vertex reducers → `t_dict`), significant-window merging, `get_trl_indices`. |
| **`EXP1_help_functions.py`** | (from Experiment 1) Low-level MEG/BIDS/GED helpers carried over from Experiment 1: `BidsPath`, `read_epochs`, `read_fwd`, `create_inverse`, covariance/GED routines (`get_ged`, `ged_get_time_course`, `ged_compute`, …), `crop_stcs`, `lowpass_filter`. |
| **`EXP1_config.py`** | (from Experiment 1) Study parameter/config module (paths and study-level constants). |
| **`bayes_factor_fun.py`** | (from Experiment 1) Bayesian statistics. Main entry `bayes_ttest` (paired/one-sample JZS Bayesian *t*-test → BF₁₀, with tail); plus Bayesian binomial/decoding and Kendall-τ Bayes factors. |

---

## 5. Pipelines by analysis stage

### Stage 0 — Preparation (per subject)
| Script | Role |
|--------|------|
| `pipeline_presave_sourceinfo.py` | Pre-save forward/inverse solution (inverse operator, covariances/rank, example stc) so downstream source analyses reuse it. Config: `pipeline_presave_sourceinfo/`. |
| `pipeline_presaved_power.py` | Pre-save source-level band power (Morlet, per band) as `pow_tlvt` (trial × vertex × time) + per-ROI vertex indices. Config: `pipeline_presaved_power/`. |

### Stage 1 — IIT vertex selection (VertROI)
Select the "qualified" vertices per ROI, then bring them to a common brain.
| Script | Role |
|--------|------|
| `pipeline_IIT_vertices_select.py` | **Individual.** CBPT on source band power to select vertices: *within-trial* (active vs baseline → responsive vertices) or *between-trial* (face vs object → content-selective vertices). Saves a VertROI label. Config: `pipeline_IIT_vertices_select/`. |
| `pipeline_IIT_vertices_morph.py` | **Individual.** Morph a subject's selected vertices to fsaverage (continuous + binary presence map). Config: `pipeline_IIT_vertices_morph/`. |
| `pipeline_IIT_vertices_morph_group.py` | **Group.** Average morphed maps across subjects for common-space display. *Runs directly* (hardcoded config, e.g. `…/pipeline_IIT_vertices_morph/dAT.json`). |
| `pipeline_IIT_vertices_plot.py` | **Group.** VertROI summary table (which/how many subjects have vertices) + group-mean log-power time course (cond1 vs cond2) over significant vertices. *Iterates over* config folder `pipeline_IIT_vertices_plot/`. |

### Stage 2 — GED spatial filters (GNW nodes)
| Script | Role |
|--------|------|
| `pipeline_GED.py` | **Individual.** Build a GED (Generalized Eigenvalue Decomposition) spatial filter per ROI/GED_type (signal = stim-active vs reference = no-stim), save component time courses. Used later for synchronization (GNW nodes FF, PFC). Config: `pipeline_GED/`. |
| `pipeline_GED_group_plot.py` | **Group plot.** For category-selective FF filters, plot preferred vs non-preferred (face vs object) time courses; trials balanced, reduced by RMS (filter sign is arbitrary). Config folder: `pipeline_GED_group_plot/`. |

### Stage 3 — Activation: ERF (evoked responses)
Seen vs unseen (and vs blank) evoked responses, tested with CBPT + Bayesian.
Four parallel families share the same 3-stage structure:

| Family | Individual → Group → Plot | Signal / ROI |
|--------|---------------------------|--------------|
| **source** | `pipeline_erf_source` · `…_group` · `…_group_plot` | anatomical-ROI source ERF (GNW: PFC / POS). Group windows: `pipeline_erf_source_group_250_500/`, `…_GNW_baseline/`. |
| **source (leakage)** | `pipeline_erf_source_group_checkleackage.py` | leakage control comparing PFC vs POS (magnitude test + jackknife onset-latency test, Miller et al. 1998). |
| **source 3-cond** | `pipeline_erf_source_3conditions` · `…_group` · `…_group_plot` | seen / unseen / blank; repeated-measures one-way ANOVA (cluster-forming F) + FDR post-hoc; reported BF₀₁ from the group step. Group: `pipeline_erf_source_3conditions_group_0_1500/`. |
| **VertROI** | `pipeline_erf_VertROI` · `…_group` · `…_group_plot` | IIT vertex-ROI source ERF. Group: `pipeline_erf_VertROI_group_IIT_baseline/`. |
| **sensor** | `pipeline_erf_sensor` · `…_group` · `…_group_plot` | sensor-space ERF. Group: `pipeline_erf_sensor_group_0_1000/`. |

### Stage 4 — Activation: TFR (oscillatory power)
Log-power difference between conditions, 1–100 Hz multitaper (1–30 Hz: 1 taper,
n_cycles=f/2; 30–100 Hz: 3 tapers, n_cycles=f/4). Two ROI families, each with a
**band** analysis and a **freq×time (ft)** analysis:

| Family | Individual | Band (alpha/gamma) | Freq×time |
|--------|-----------|--------------------|-----------|
| **source** | `pipeline_tfr_source` | `…_group_band` (+`_plot`) | `…_group_ft` (+`_plot`) |
| **VertROI** | `pipeline_tfr_VertROI` | `…_group_band` (+`_plot`) | `…_group_ft` (+`_plot`) |

- **band**: average power over one band's frequencies + over vertices → per-label
  1-D time course; cluster over `labels × time`. Reported values via Bayesian
  BF₀₁ + CBPT.
- **ft**: keep the full frequency axis → per-label 2-D `(freq × time)` map; cluster
  over `labels × freqs × times`. Mainly the time-frequency figure.
- Configs: `pipeline_tfr_source_group_band_250_500/`, `…_band_baseline/`,
  `…_ft_250_500/`; `pipeline_tfr_VertROI_group_band_baseline/`, `…_ft_baseline/`.

### Stage 5 — Baseline: contralateral alpha
| Script | Role |
|--------|------|
| `pipeline_contralateral_alpha.py` | **Individual.** Re-compute alpha-band Morlet power (needs long windows) for **all** trials in the ROI, keeping left/right hemisphere vertex indices. Config: `pipeline_contralateral_alpha/`. |
| `pipeline_contralateral_alpha_group.py` | **Group.** Split trials by hemifield (LVF/RVF) × response, average power over contralateral vs ipsilateral vertices, pool locations → **contralateral power** (`contra_power`). Bayesian + CBPT on seen vs unseen. Config: `pipeline_contralateral_alpha_group_baseline/contra_power_baseline.json`. |
| `pipeline_contralateral_alpha_group_plot.py` | **Group plot.** Contralateral-power time courses + significant clusters. |

### Stage 6 — Synchronization (GNW: GED nodes · IIT: vertex nodes)
Extract ROI time courses → compute connectivity per subject (with trial
subsampling) → group statistics → plot.

| Step | GNW (GED ROIs: FF vs PFC) | IIT (vertex ROIs: FF vs V1/V2) |
|------|---------------------------|-------------------------------|
| Time courses (individual) | `pipeline_syn_timecourse_GED_ROI.py` (project pre-computed GED filters → component time course) | `pipeline_syn_timecourse_PCA_VertROI.py` (PCA-flip vertex time course) |
| Connectivity (individual, subsampled) | `pipeline_syn_subsampling_ppc_GED_ROI.py` (PPC) · `pipeline_syn_subsampling_dfc_GED_ROI.py` (DFC, Gaussian-Copula MI) | `pipeline_syn_subsampling_ppc_PCA_VertROI.py` (PPC) |
| Group stats | `pipeline_syn_ppc_GED_ROI_group.py` · `pipeline_syn_power_dfc_GED_ROI_group.py` | `pipeline_syn_ppc_PCA_VertROI_group.py` |
| Group plot | `pipeline_syn_group_plot_mannual_loop_sameclim_pcolormesh_logscale.py` — freq×time matrices, shared color limits, log-y; *manual loop*, no argparse | ← same script covers both GNW & IIT |

- **PPC** = pairwise phase consistency (Vinck et al. 2010; MNE `spectral_connectivity_epochs`).
- **DFC** = dynamic functional connectivity (Gaussian-Copula MI over sliding windows; `frites`).
- Individual configs: `pipeline_syn_ppc_GED_ROI/`, `pipeline_syn_power_dfc_GED_ROI/`,
  `pipeline_syn_ppc_PCA_VertROI/`, `pipeline_syn_timecourse_*`.
  Group configs: `…_group_250_500/`, `…_group_100_600/`, `…_group_baseline/`.

---

## 6. Typical run order

```
0. presave_sourceinfo → presaved_power              (per subject, once)
1. IIT_vertices_select → _morph → _morph_group / _plot   (IIT vertex ROIs)
2. GED                                               (GNW spatial filters)
3. erf_* individual → _group → _group_plot           (ERF activation)
4. tfr_* individual → _group_band/_ft → _plot        (TFR activation)
5. contralateral_alpha → _group → _group_plot        (alpha baseline)
6. syn_timecourse_* → syn_subsampling_* → _group → group_plot   (synchrony)
```

Each script's head docstring lists its exact inputs, outputs, config files, and
which script to run before it.

---

## 7. Notes / conventions

- **Reducers**: `mean` vs `rms` over vertices/trials; RMS is used where a signal's
  sign is arbitrary (e.g. GED filters, source vertices) so a signed mean would cancel.
- **Trial balancing**: unbalanced conditions are equalized by subsampling
  (`Nsample` draws, seed derived from the subject code).
- **Reported statistics** favor Bayesian BF₀₁ (evidence for the null) alongside CBPT,
  matching the preregistration.
- Python 3.12+ is assumed (some f-strings use nested same-quote interpolation).
