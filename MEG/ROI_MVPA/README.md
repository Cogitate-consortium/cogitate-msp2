# MEG MVPA decoding analysis

This directory contains the code used for the MEG MVPA decoding analyses reported in COGITATE Experiment 2 (ses-V2): subject level analysis for different analysis mthods, group level analysis for concatenate individual subject data, statistical analysis and plotting, as well as statistic for bayes factors.
---

## General requirements

### Compute environment

* Python 3
* MNE-Python
* NumPy
* SciPy
* scikit-learn
* scikit-image
* joblib
* matplotlib
* FreeSurfer reconstructions for source-space analyses

### Upstream data

The analyses require outputs from the MEG preprocessing and source-reconstruction pipelines:

| Resource                         | Description                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------ |
| Preprocessed MEG epochs          | Epochs for the dAT, AT and resting-state recordings                            |
| Forward solutions                | Participant-specific MEG forward models                                        |
| FreeSurfer derivatives           | Participant-specific cortical reconstructions                                  |


### Usage



1.0 Subject level analysis: Use D0X_AAA_MVPA_XX.py to run subject level analysis, each code represent a analysis method. AAA could be ROI/sensor/ET which means decoding in ROI based MEG source signal, sensor signal or eyemovement signal. To run the Face vs Object Category decoding analysis for Subject SA001 for experiment 2 in dAT task in source space, simply use the parameter. "analysis_name" used in each code will be the index for group level analysis, e.g for D01_ROI_MVPA_dAT.py, analysis_name='dAT'

python D01_ROI_MVPA_dAT.py --sub SA001 

2.0 Group level anaylsis: D99_group_data_xx.py is used for concatenate individual subject data to one file of group data. To concatenate decoding analysis for dAT condition, simply use the parameter:

python D99_group_data_pkl_E2.py --analysis dAT

3.0 Group level statistical analysis and plotting D98_group_stat_sROI_xx.py is used for generate final results figure with the data that generated from D99_group_data_xx.py code. To generate the main Figure of Category decoding analysis, simply use the parameter:

Python D96_group_stat_sROI_plot_E2.py  --analysis dAT

config file contain the parameter used for MEG analysis
D_MEG_function_E2.py contain the function used on ROI_MVPA analysis
D98_group_stat_bayes_factors.py used for Bayes factors analysis
sublist_exp2.py is subject list for ROI_MVPA analysis