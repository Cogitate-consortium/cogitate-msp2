# Eye Tracking Quality Checks for Experiment 2 Data

This repository contains codes for ET QC of experiment 2 data. 
**Note**: the quality checks assume that subject folers are already in an XNAT-compatible structure. Meaning, that each subject folder (e.g. "SZ104") contains 2 sub-folders: /BEH, /ET (e.g., "SZ104/BEH"). 
Each data type (BEH/ET) contains the structure of the resource as it should've been uploaded to XNAT (e.g., "SZ104/BEH/SZ104/1"). 
See more here:  https://twcf-arc.slab.com/posts/2-subject-visit-data-structure-and-naming-conventions-gkibcour

## What Does it Do?
In the following explanation:
- Conditions = Information types to divide trials by. 
- Sub conditions = the specific possible values for each condition

In the table below, each column header is a condition, and the column values are its sub conditions:

| Category      | Location      | Game World  | Replay World  | Visibility     |
| ------------- |:-------------:|:-----------:|:-------------:| --------------:|
| Face          | Top Right     |   World 1   |    World A    | True Positive  |
| Object        | Top Left      |   World 2   |    World B    | False Positive |
|               | Bottom Right  |   World 3   |               | True Negative  |
|               | Bottom Left   |   World 4   |               | False Negative |

- Time-Windows: in ms, the durations of specific events as they are defined in the `ET_param_manager.py` file. There are 3 types of time-windows:
	- Pre stimulus: the amount of milliseconds to take before stimulus appears
	- Stimulus Duration: the amount of milliseconds during which the stimulus appeard
	- Epoch: an amount of time prior to stimulus start (Epoch start), and after stimulus start (Epoch end), s.t Epoch start + Epoch end = Epoch. 

For every subject, you get the following information about the ET data:
- A QC_results_....csv table: This is a summary table which is outputted **for** **every** **time-window** **separately** and contains 4 rows (4 types of data):
	- % fixation on Stimulus
	- Mean fixation distance from Stimulus
	- % fixation on Center
	- Mean fixation distance from Center
	The columns are "overall" (across all trials), and then per sub-condition.

- Mean Euclidian Distance From Center/Stimulus: these are 2 plots (and their accompanying csv files) which are calculated over Epochs, and outputted both overall and for each Condition separately (in the condition folder). It depicts the mean distance in time (entire Epoch), with vertical lines where stimulus presentation starts and ends.  

- Fixation Density: these are 3 plots (one per time-window) (and their accompanying csv files) outputted both overall and for each Condition separately (in the condition folder). This is a heatmap of gaze distribution. 

## How Does it Work?
The quality checks are ran by the `ET_qc_manager.py` file. Call *analyze_ET* , giving it *sub_dir* and *save_path* as inputs, and it will take care of the rest.
*sub_dir* is a directory of subject folders to be checked, while the checks themselves are performed one subject at a time. *save_path* is the folder in which all the outputs will be saved, divided to subject folders. 
`ET_qc_manager.py` goes subject by subject and calls the `QualityChecker.py` module to perform the checks. 
The parameters on which all the data extraction, parsing and checks are based can be found in the `ET_param_manager.py` file. 
The module which actually checks the data is the `QualityChecker.py`. It calls other modules to help with parsing and calculations, and it produces all the data and plots for the QC of a single subject's ET data. 


## Authors
- **Abdo Sharaf** [AbdoSharaf98](https://github.com/AbdoSharaf98)
- **Rony Hirschhorn** [RonyHirsch](https://github.com/RonyHirsch)

