#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 1st-level FEAT analyses using FSL.

For each analysis in analysis_definitions, creates per-subject, per-run FSF
files from templates under glm/fsf_templates/1st_level/, then optionally
submits FEAT jobs via sbatch.

Assumed:
    1. BIDS dataset preprocessed with fMRIPrep (default paths and filenames)
    2. FSF templates with placeholders:
       - PXX (subject ID)
       - XXX_NTPS (number of volumes; filled from fMRIPrep preproc BOLD)
       - task/run placeholder (e.g. VGXX for task-VG)
       - space-SXX (analysis space, e.g. MNI152NLin2009cAsym)

Requires FSL in PATH before running:
    module load FSL

Created on Fri Mar 29 15:11:06 2021

@author: David Richter, Yamil Vidal
Modified by Yamil Vidal 08/05/2026

"""

import os
import time

import pandas as pd


# %% Paths and Parameters

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
fsl_output_dir = bids_dir + '/derivatives/fslFeat'

# Whether to submit feat jobs using sbatch. If False only fsf files are created
submit_jobs = True

# Optional wait time between submitting successive jobs
wait_between_jobs_in_sec = 0


# %% Define analyses to be performed
"""
Dict per analysis, specifying analysis label, suffix and number of runs.
Analysis labels must match the fsf file template names!

desc: (optional) description of analysis
label: session and task key-value pairs plus run- prefix (e.g. ses-V2_task-VG_run-)
suffix: analysis type and space (e.g. analysis-1stGLM_space-MNI152NLin2009cAsym)
fsf_file: path to fsf templates relative to bids_dir
runs: number of runs
walltime: hours required to process run (submitted as job)
memory: gb memory required to process run (submitted as job)
"""

analysis_definitions = {
    'GLM_VG_MNI': {
        'desc': '1st level GLM for Exp2 VG',
        'label': 'ses-V2_task-VG_run-',
        'suffix': 'analysis-1stGLM_space-MNI152NLin2009cAsym',
        'runs': 8,
        'fsf_file': '/code/glm/fsf_templates/1st_level',
        'walltime': 8,
        'memory': 8,
    },
}


# %% Functions returning analysis names/types and space and paths
def get_fmriprep_processed_nifti_file(bids_dir, sub, session_label, analysis, space):
    """
    get nifti file name, given path + pattern (defined here), and input parameters
    bids_dir = bids directory
    sub: subject ID (e.g. sub-SC101)
    session_label: session label ("key-value" pair; e.g. ses-V2)
    analysis: analysis label (e.g. ses-V2_task-VG_run-1)
    space: analysis space (e.g. space-MNI152NLin2009cAsym)
    Returns nifti file path+filename
    """
    nifti_path_pattern = (
        bids_dir + os.sep + 'derivatives' + os.sep + 'fmriprep' + os.sep
        + '%(sub)s' + os.sep + '%(ses)s' + os.sep + 'func' + os.sep
    )
    nifti_file_pattern = '%(sub)s_%(analysis)s_%(space)s_desc-preproc_bold.nii.gz'
    nifti_file = nifti_path_pattern % {'sub': sub, 'ses': session_label}
    nifti_file += nifti_file_pattern % {'sub': sub, 'analysis': analysis, 'space': space}
    return nifti_file


def get_analysis_labels(analysis_dict):
    """
    Combine analysis label, suffix and run numbers into per-run analysis labels.
    Returns analyses, analyses_suffix, fsf_templates_dir.
    """
    analyses = []
    analyses_suffix = analysis_dict['suffix']
    fsf_templates_dir = bids_dir + analysis_dict['fsf_file']
    for run in range(analysis_dict['runs']):
        analyses.append(analysis_dict['label'] + str(run + 1))
    return analyses, analyses_suffix, fsf_templates_dir


def write_error_log(bids_dir, error_df):
    """Write log file listing possible errors encountered during job submission."""
    timestr = time.strftime("date-%Y%m%d_time-%H%M%S")
    file_name = os.sep + 'log_job_submission_1st-level_' + timestr + '.csv'
    output_dir = (
        bids_dir + os.sep + 'derivatives' + os.sep + 'exclude' + os.sep + 'new'
        + os.sep + 'fMRI_exp2' + os.sep + 'glm' + os.sep + 'fslFeat'
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fname = output_dir + file_name
    error_df.to_csv(fname, sep=',', index=False, na_rep='Null')
    print('Error saved to ' + output_dir)


# %% functions to modify fsf templates
def replace_text(fname, string_to_replace, replacement_string):
    """Replace text in input file (fname)."""
    import fileinput
    with fileinput.FileInput(fname, inplace=True) as file:
        for line in file:
            print(line.replace(string_to_replace, replacement_string), end='')


def get_n_volumes(nifti_file):
    """Get number of volumes in input nifti file."""
    import subprocess
    full_cmd = ['fslhd', nifti_file]
    result = subprocess.run(full_cmd, stdout=subprocess.PIPE, check=False)
    fslHd_output = result.stdout.decode('utf-8')
    dim4_target_string = '\ndim4\t\t'
    n_vols = fslHd_output[
        fslHd_output.find(dim4_target_string) + len(dim4_target_string):
        fslHd_output.find('\ndim5')
    ]
    return n_vols


def make_this_fsf_template(fsf_file, sub, analysis, nifti_file, place_holder, space):
    """
    Take fsf template and replace subject, analysis and number of volumes
    placeholders. Subject ID placeholder is PXX; volumes XXX_NTPS;
    analysis placeholder is task key value + XX (e.g. VGXX).
    """
    print(' . Creating fsf files for analysis: ' + analysis)

    replace_text(fsf_file, place_holder, analysis)
    replace_text(fsf_file, 'PXX', sub[4:])
    n_vols = get_n_volumes(nifti_file)
    replace_text(fsf_file, 'XXX_NTPS', n_vols)
    replace_text(fsf_file, 'space-SXX', space)


def check_if_feat_output_exists(fsf_file):
    """
    Checks if feat output specified in fsf file already exists.
    Returns feat_dir_exists: bool; true if output already exists.
    """
    with open(fsf_file, 'r') as file:
        fsf_file_text = file.read()
    target_string = 'set fmri(outputdir) "'
    feat_dir = fsf_file_text[
        fsf_file_text.find(target_string) + len(target_string):
        fsf_file_text.find('# TR(s)') - 3
    ]
    return os.path.isdir(feat_dir)


# %% functions for submitting feat jobs
def write_tmp_file_with_sbatch_cmd(fsf_file):
    """Create file with command to run feat on fsf file to be submitted as job."""
    tmp_file = (
        os.path.split(fsf_file)[0] + os.sep + 'tmpscript_'
        + fsf_file[fsf_file.find('sub-'):fsf_file.find('.fsf')]
    )
    with open(tmp_file, 'w') as file:
        file.write(
            '#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --partition=octopus\nfeat '
            + fsf_file + '\n'
        )
    return tmp_file


def submit_feat_job(fsf_file, walltime, memory):
    """
    Submit feat jobs using fsf files, unless outputs already exist.
    Returns error_msg: list of [error_type, run_no] entries.
    """
    output_dir_str = 'set fmri(outputdir) "'
    with open(fsf_file, 'r') as file:
        fsf_file_text = file.read()
    feat_dir = fsf_file_text[
        fsf_file_text.find(output_dir_str) + len(output_dir_str):
        fsf_file_text.find('# TR(s)') - 3
    ]
    if os.path.isdir(feat_dir):
        print(' . ! FEAT dir ALREADY EXISTS: ' + os.path.split(feat_dir)[1] + ' ! Skipping !')
        return [['_ Analysis output already existed - Skipped _', '_']]

    tmp_file = write_tmp_file_with_sbatch_cmd(fsf_file)
    submit_cmd = (
        f"sbatch -N1 -n1 -t {walltime}:00:00 --mem={memory}gb {tmp_file}"
    )
    print(' . . Submitting FEAT job --> ' + tmp_file[tmp_file.find('tmpscript'):])
    os.system(submit_cmd)
    time.sleep(wait_between_jobs_in_sec)
    return []


# %% main function to loop over subjects & analyses
def run_fsf_creation_and_submit_feat_job(
    bids_dir, subjects, analyses, analyses_suffix,
    fsf_templates_dir, fsf_output_dir, error_df, walltime, memory,
):
    """
    Run fsf file creation for all subjects and runs for current analysis specification.
    """
    from shutil import copy as copyFile

    for sub in subjects:
        print('Subject: ' + sub + ' | processing...')
        for analysis in analyses:
            error_msg = []
            session_label = analysis[analysis.find('ses-'):analysis.find('task-') - 1]
            space = analyses_suffix[analyses_suffix.find('space-')::]
            nifti_file = get_fmriprep_processed_nifti_file(
                bids_dir, sub, session_label, analysis, space
            )
            place_holder = analysis[analysis.find('task-') + 5:analysis.find('_run-')] + 'XX'
            place_holder_analysis_label = analyses_suffix[:analyses_suffix.find('_space')]

            if os.path.isfile(nifti_file):
                input_file = (
                    fsf_templates_dir + os.sep + 'sub-PXX_' + place_holder + '_'
                    + place_holder_analysis_label + '_space-SXX.fsf'
                )
                fsf_file = (
                    fsf_output_dir + os.sep + sub + '_' + analysis + '_'
                    + analyses_suffix + '.fsf'
                )
                copyFile(input_file, fsf_file)
                if check_if_feat_output_exists(fsf_file):
                    print(
                        ' ! Subject: ' + sub + ' | Analysis: ' + analysis
                        + ' in ' + space + ' space. FEAT output already exists ! Skipping ! '
                    )
                    continue
                make_this_fsf_template(fsf_file, sub, analysis, nifti_file, place_holder, space)
                if submit_jobs:
                    error_msg = submit_feat_job(fsf_file, walltime, memory)
                if any(error_msg):
                    for err_row in error_msg:
                        error_df.loc[len(error_df)] = [
                            sub, analysis, analyses_suffix, 'run-' + err_row[1], err_row[0],
                        ]
                else:
                    error_df.loc[len(error_df)] = [
                        sub, analysis, analyses_suffix, '-', '_ Job submitted _',
                    ]
            else:
                print(nifti_file)
                print(
                    ' ! Subject: ' + sub + ' | Analysis: ' + analysis
                    + ' in ' + space + ' space. Nifti FILE NOT FOUND !'
                )
                error_df.loc[len(error_df)] = [
                    sub, analysis, analyses_suffix, 'run-x', 'nifti file not found',
                ]


# %% run
if __name__ == '__main__':
    subject_list = projectRoot + '/bids/code/ses-v2-analysis-subs-fmri.csv'
    subj_df = pd.read_csv(subject_list, sep=None, engine='python')
    if 'sub_code' not in subj_df.columns and len(subj_df.columns) == 1 and ';' in subj_df.columns[0]:
        subj_df = pd.read_csv(subject_list, sep=';')

    required_cols = ['ACTIVATION_min_sf_uf', 'ACTIVATION_min_so_uo']
    missing_cols = [col for col in required_cols if col not in subj_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in subject CSV: {missing_cols}")
    flag_sf_uf = subj_df['ACTIVATION_min_sf_uf'].astype(str).str.upper().eq('TRUE')
    flag_so_uo = subj_df['ACTIVATION_min_so_uo'].astype(str).str.upper().eq('TRUE')
    subj_df = subj_df.loc[flag_sf_uf | flag_so_uo]

    subjects = [f'sub-{sub}' for sub in subj_df['sub_code'].values]

    flag_sf_uf = subj_df['ACTIVATION_min_sf_uf'].astype(str).str.upper().eq('TRUE')
    flag_so_uo = subj_df['ACTIVATION_min_so_uo'].astype(str).str.upper().eq('TRUE')
    only_sf_uf = flag_sf_uf & ~flag_so_uo
    only_so_uo = flag_so_uo & ~flag_sf_uf
    both_true = flag_sf_uf & flag_so_uo

    print(f"Number of subjects with only 'ACTIVATION_min_sf_uf' == TRUE: {only_sf_uf.sum()}")
    print(f"Number of subjects with only 'ACTIVATION_min_so_uo' == TRUE: {only_so_uo.sum()}")
    print(
        "Number of subjects with both 'ACTIVATION_min_sf_uf' and "
        f"'ACTIVATION_min_so_uo' == TRUE: {both_true.sum()}"
    )

    # subjects = ['sub-SC108']  # debug: single subject

    error_df = pd.DataFrame(columns=['sub', 'analysis', 'analyses_suffix', 'error_run_no', 'error_msg'])

    for key in analysis_definitions:
        print('')
        print('PROCESSING ANALYSIS: -> ' + key + ' <- ')

        analysis_dict = analysis_definitions[key]
        analyses, analyses_suffix, fsf_templates_dir = get_analysis_labels(analysis_dict)

        if submit_jobs:
            walltime = analysis_dict['walltime']
            memory = analysis_dict['memory']
        else:
            walltime = 0
            memory = 0

        fsf_output_subdir = analyses_suffix[
            analyses_suffix.find('analysis-') + len('analysis-'):
            analyses_suffix.find('analysis-') + len('analysis-') + 3,
        ]
        fsf_output_dir = (
            bids_dir + os.sep + 'derivatives' + os.sep + 'fslFeat' + os.sep
            + 'fsf_files' + os.sep + fsf_output_subdir + '_level'
        )
        if not os.path.isdir(fsf_output_dir):
            os.makedirs(fsf_output_dir)

        run_fsf_creation_and_submit_feat_job(
            bids_dir, subjects, analyses, analyses_suffix,
            fsf_templates_dir, fsf_output_dir, error_df, walltime, memory,
        )

    write_error_log(bids_dir, error_df)
