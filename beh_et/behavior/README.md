# Checks and Analyses Codes for Exp.2
This folder is a collection of data quality checks and behavioral analyses scripts that can be done on any subject folder/s of exp.2 (regardless of modality).
NOTE: if you ran the Exp.2 analyzer in advance to output a "___Analysis.csv" files for each subject (separatly, so that the "___Analysis.csv" file only contains a single subject), the program assumes that this file is located within the SESSION folder of a specific subject.
Meaning, if you have analysis output files (per subject awaaion), make sure the relevant subject session folder (whether it's pre-screening data or a full run data folder, e.g., "SA103/A", "SA103/1") has a "StimulusAnalysis.csv" file right under it (e.g. "SA103/1/StimulusAnalysis.csv"), and not elsewhere.
IMPORTANT: the analyzer output is NOT necessary for any of the quality checks, or the behavioral analyses. 

## Quick Navigation
### I want one simple file where I can see all the options for analyses and decide for myself what I will run
The [behav_analysis_runner](#behav_analysis_runner) module. Only 1 function in it, where you choose what analyses you want to perform (QC, general, specific), if you want to filter general/specific analyses according to quality checks, and that's it - the rest will be done for you. 

### I want to perform quality-checks
You need the [quality_checks](#quality_checks) module. Within it, run the "check_data" function, with its required parameters. 
What you'll get is a BIDS-compatible folder hierarchy under the data_path folder you entered, and within it the quality check plot and csv file under: "...folder_name_you_gave_as_root_for_BIDS\ANALYSIS_DATA\derivatives\behavioral\population_analysis\quality_checks". 
See more at the [Data Quality Checks](#quality_checks) section below.

### I want to perform general analyses 
You need the [general_analysis](#general_analysis) module. Within it, run the "analyze" function, with its required parameters. Note that the first parameter is "check", and if you want to perform quality checks on your data prior to the general analysis, you can mark it as "True". This will both perform the quality checks (and save them etc), AND will go on to analyze only the subjects who passed the checks.
What you'll get is a BIDS-compatible folder hierarchy under the data_path folder you entered and within it the general analyses' csv files and plots under: "...folder_name_you_gave_as_root_for_BIDS\ANALYSIS_DATA\derivatives\behavioral\population_analysis\general_analysis". 
See more at the [General Analyses](#general_analysis) section below.

### I want to perform specific analyses of response types
You need the [specific_analysis](#specific_analysis) module. Within it, run the "analyze" function, with its required parameters.
What you'll get is a BIDS-compatible folder hierarchy under the data_path folder you entered and within it the specific analyses' csv files and plots under: "...folder_name_you_gave_as_root_for_BIDS\ANALYSIS_DATA\derivatives\behavioral\population_analysis\specific_analysis". 
See more at the [Specific Analyses](#specific_analysis) section below.


## Data Upload and Organization
#### data_reader
This module uploads relevant information from the subject data folders, and organizes it in classes for future checks and analyses.
Conceptually, each subject is a class (class Subject) that contains basic subject information (path, lab code, subject id) and all the sessions of exp.2 this subject had (pre-screening session, full run, or both).
Each session is also a class (class Session), which contains basic session information (path, name) and all the important raw data from that session that will be used for quality checks and analyses. 
This includes the Details files (class SessDetails) and the analyzer output file ("StimulusAnalysis.csv" - if it exists in the expected place in the subject's folder hierarchy), which are saved as dataframes.
The module includes all the said classes, and two methods:
- read_sub_data : which, given a single subject, reads and organizes all the subject information.
- read_data : which, given a folder containing multiple subject, goes over all of them to perform the said actions.

#### data_saver
This module includes all the actions related to data saving. The data and figures are saved in a BIDS-compatible structure. 
The basic method which creates the BIDS-compatible file hierarchy is create_bids.py. 
Whenever any of the check/analysis modules saves data, it is performed using methods from the data_saver module.
When you give a "save_path" parameter to an analysis/qc function, you give the path to the HIGHEST folder in the BIDS-compatible folder hierarchy. If such folder does not exist, the data_saver module creates it and then creates the required folder structure underneath it, in order to save all files in the correct directories.


## Data Quality Checks
#### quality_checks
This module checks whether behavioral subject data from exp.2 meets the requirements: both for screening and for full-game data.  
It loads data using the [data_reader](#data_reader) module, parses the relevant data and calculates the parameters to be sent to the [quality_checks_criteria](#quality_checks_criteria) module to be checked. 
Then, it creates a table of all checked subjects, with all the checked parameters, and summary columns which indicate whether or not the test passed or failed. 
For subjects who have behavioral screening data, there's an option to plot their true positive (TP) and false alarm (FA) ratios from their screening session (by calling the [boxplotter](#boxplotter) module). 

To perform the check, call the "check_data" method, entering the data_path (where all the subject folders are) and the save_path (the path to the existing/to be created folder which is the highest folder in the BIDS-compatible structure).

#### quality_checks_criteria
This module contains all the exclusion criteria for exp.2 subject data. Conceptually, there are 2 types of exclusions:
- Subjects who had a behavioral screening session: those subjects will be summoned to participate in the full experiment if and only if their behavioral data meets the requirements.
- Subjects who participated in exp.2 (full experimental paradigm): their data will be analyzed if and only if their behavioral data meets the requirements.
All of the requirements, including the functions that check them, are in this module. Given a certain ratio/amount, each function checks a different requirement.


## Data Analyses
### Behavioral Analyses
These are analyses of the behavioral data of exp.2, meaning all that has to do with subject's tasks (whether it's playing the video game, or indicating whether they saw a face/object).
#### General Analyses
##### general_analysis
This module coordinates and runs everything that has to do with analyses of video-game difficulty, performance, and probe response-types across the game. 
After extracting the relevant data for these analyses, this module calls the analysis modules:
- [response_type_analysis](#response_type_analysis)
- [diff_perf_analysis](#diff_perf_analysis)

##### response_type_analysis
This module performs analysis of response type rates. It calculates the mean/median performance/difficulty per response type. 
It plots boxplots of this data (using the [boxplotter](#boxplotter) module), as well as line plots (using the [lineplotter](#lineplotter) module) depicting moving average windows of specific response type rates across the entire game (1 plot per response type). 
All the data generating the plots is saved as csv files under the appropriate BIDS-compatible folder. 

##### diff_perf_analysis
This module performs analysis of difficulty and performance across the game.
It produces line plots (using the [lineplotter](#lineplotter) module) depicting moving average windows of difficulty and performance (separatly) across the entire game, and saves the csv files where this data is derived from.


#### Specific Analyses
##### specific_analysis
This module performs everything that has to do with specific analyses of subjects' responses across the entire exp.2 duration. It splits response type data:
- Per level
- Per World
- Per stimulus location
- Per stimulus type
And for each type (and for each section of exp.2: game or replay) it outputs raincloud plots with response-type rate information (using the [boxplotter](#boxplotter) module) and saves the appropriate data as csv files. 


## Data Visualisation
### boxplotter
This module has 2 plotting capabilities:
- boxplot: with or without scatter points and lines connecting dots belonging to the same subject.
- raincloudplot: a plot containing a half-violin plot, boxplot and scatter of each data section. 

### lineplotter
This module can plot either 1 or 2 datasets on the same plot, both of which depict average (or moving average) and can contain (optional) standard-error margins encompasing the lines. 

## The Key to Everything
### behav_analysis_runner
This module can run all the existing behavioral analyses:
- [Quality checks](quality_checks)
- [General analysis](general_analysis)
- [Specific analysis](specific_analysis)
Choose which of them you want to run (could be one, could be all). For general and specific, choose if you want to analyze only subjects who pass the quality checks (True), or you want to analyze all subjects (False). 

## Authors
- **Abdo Sharaf** [AbdoSharaf98](https://github.com/AbdoSharaf98)
- **Rony Hirschhorn** [RonyHirsch](https://github.com/RonyHirsch)