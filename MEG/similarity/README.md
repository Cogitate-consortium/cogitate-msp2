# MEG representational similarity analysis

This directory contains the code used for the MEG representational similarity analyses reported in COGITATE Experiment 2 (ses-V2): main analysis, temporal-split control, gamma-based vertex-selection control, optimized vertex-selection control, group statistics, and plotting.

The analyses quantify time-resoled similarity between MEG source-level representations of stimulus **category** and **location** across experimental conditions and tasks.

---

## General requirements

### Compute environment

* Python 3
* MNE-Python
* NumPy
* SciPy
* scikit-learn
* matplotlib
* FreeSurfer reconstructions for source-space analyses

### Upstream data

The analyses require outputs from the MEG preprocessing and source-reconstruction pipelines:

| Resource                         | Description                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------ |
| Preprocessed MEG epochs          | Epochs for the dAT, AT and resting-state recordings                            |
| Forward solutions                | Participant-specific MEG forward models                                        |
| FreeSurfer derivatives           | Participant-specific cortical reconstructions                                  |
| Stimulus–blank power differences | Alpha- and gamma-power differences used for ROI selection in the main analysis |

### Paths

Input and output paths are defined in `cfg.py`. Edit these paths to point to the local MEG derivatives and output directories before running the analyses.

The participant to be analysed is specified by its index in `cfg.subjects`.

---

## 1. Main representational similarity analysis (`rsa_compute_core.py`)

Computes time-resolved source-pattern similarity for stimulus category and location across the three task/response partitions:

* `vg / seen`
* `replay / seen-go`
* `replay / seen-no-go`

Run separately for each participant:

```bash
python rsa_compute_core.py <subject_index> main
```

Output:

```text
similarity_mnn_<subject>_core.pkl
```

---

## 2. Temporal-split control (`rsa_compute_core.py`)

Control analysis for temporal proximity between trials by dividing each task/response partition into two halves according to trial progression and repeating the similarity analysis.

Run separately for each participant:

```bash
python rsa_compute_core.py <subject_index> temporal_splits
```

Output:

```text
similarity_mnn_<subject>_temporal_splits.pkl
```

---

## 3. Gamma-based vertex-selection control (`rsa_compute_gamma.py`)

Control analysis targeting vertices showing strong stimulus-related gamma activity.

Vertex selection is estimated from replay `seen-no-go` trials relative to blank trials during the 0.3–0.7 s post-stimulus window. Vertices are ranked according to the stimulus-to-blank gamma-power difference, and the top 10% and 5% are selected. Size-matched random selections are included as controls.

Run separately for each participant:

```bash
python rsa_compute_gamma.py <subject_index>
```

Output:
```text
rsa_gamma_<selection>_<subject>.pkl
```

---

## 4. Cross-validated vertex-selection control (`rsa_compute_optimal.py`)

Control analysis testing the similarity effect using a data-driven vertex-selection criterion.

For each cross-validation split, vertices are ranked on the training data by the ratio of task-related to content-related response differences during 0.3–0.7 s. The top 10% and 5% of vertices are selected, and RSA is computed on the held-out test data.

Run:

```bash
python rsa_compute_optimal.py
```

Output:
```text
rsa_optimal_p<percentile>_<subject>_cvi<iteration>.pkl
```

---

## 5. Group-level contrasts and statistics (`rsa_contrast.py`)

Aggregates participant-level similarity matrices and computes the group-level contrasts.

The main contrast compares:

**same-content similarity across different task/response partitions − different-content similarity within the same partition**

The contrast is computed:

* across all three task/response partitions;
* within the two replay partitions only;
* for the temporal-split control;
* for the gamma-based vertex selections;
* for the cross-validated vertex selections.

Statistical inference is performed using one-sample cluster-based permutation tests on both the full time-by-time similarity matrices and their temporal diagonals.

Bootstrap confidence intervals are estimated for the diagonal time courses.

Directional Bayes factors (`BF01`) are also computed for the null-category contrast using a half-Cauchy prior with scale `r = 0.707`.

Outputs:
Group-level contrast arrays and statistical results used by `rsa_plots.py`.

---

## 6. Plotting (`rsa_plots.py`)

Generates the group-level figures for the main analysis and control analyses.

The plotting code generates:

* main similarity contrast;
* replay-only similarity contrast;
* temporally close and far controls;
* gamma-based vertex-selection controls;
* cross-validated vertex-selection controls;
* comparisons across vertex-selection procedures;
* Bayes-factor plots.

Run:

```bash
python rsa_plots.py
```

Outputs:
Figure files for the main and control analyses in the configured output directory (`cfg.out_dir`).
