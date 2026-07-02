"""
Author: Zvi Roth
Date created: 14-08-2024
"""
import os
import operator

import nilearn.image
import pandas as pd
import numpy as np
from mansfield import get_searchlight_neighbours_matrix
import time
import argparse


num_iter = 10

tic=time.time()
# searchlight dir
# CHANGE FOR ACTUAL SUBJECTS!!
# csv_file_end = 'ses-v2-optimization-subs.csv'
csv_file_end = 'ses-v2-analysis-subs.csv'

csv_file = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/qcs/', csv_file_end)
csv_data = pd.read_csv(csv_file, sep=',')
subjects_phase2 = csv_data.sub_code
subject_list = subjects_phase2.tolist()

approach = 'within_condition'


def run_analysis():
    tic = time.time()
    excluded_subjects = list()
    missing_subjects = list()
    included_subjects = list()

    vg_or_replay = 'VG'
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Run group fMRI searchlight analysis for experiment2")
    parser.add_argument('--decoding_problem', type=str, default=None,
                        help="category or location")

    args = parser.parse_args()
    decoding_problem = args.decoding_problem

    i=0
    # Loop over subjects
    if decoding_problem == 'category':
        output_dir = os.path.join(
            '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
            'searchlight_decoding_categ_subsample/')
        analysis_check_data = csv_data.DECODING_min_sf_so_uf_uo
    else:
        output_dir = os.path.join(
            '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
            'searchlight_decoding_loc_subsample/')
        analysis_check_data = csv_data.DECODING_min_sl_sr_ul_ur

    analysis_check_list = analysis_check_data.tolist()
    print('starting to loop over subjects')
    print(subject_list)
    # Loop over subjects
    for sub_code, analysis_check in zip(subject_list, analysis_check_list):
        if analysis_check:
            sub = 'sub-' + sub_code
            # Load searchlight nifti files
            if num_iter>0:
                seen_filename = 'searchlight_' + sub + '_' + vg_or_replay + '_' + decoding_problem + '_seen_' + approach + '_' + str(num_iter) + 'subsample.nii'
                unseen_filename = 'searchlight_' + sub + '_' + vg_or_replay + '_' + decoding_problem + '_unseen_' + approach + '_' + str(num_iter) + 'subsample.nii'
            else:
                seen_filename = 'searchlight_' + sub + '_' + vg_or_replay + '_' + decoding_problem + '_seen_' + approach + '.nii'
                unseen_filename = 'searchlight_' + sub + '_' + vg_or_replay + '_' + decoding_problem + '_unseen_' + approach + '.nii'
            seen_path = os.path.join(output_dir, seen_filename)
            unseen_path = os.path.join(output_dir, unseen_filename)
            if (os.path.exists(seen_path)) & (os.path.exists(unseen_path)):
                print('adding maps for sub: ' + sub)
                included_subjects.append(sub_code)
                seen_img = nilearn.image.load_img(seen_path)
                unseen_img = nilearn.image.load_img(unseen_path)
                # subtract unseen from seen
                result_img = nilearn.image.math_img("img1 - img2", img1=seen_img, img2=unseen_img)
                if num_iter > 0:
                    output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen_' + str(
                        num_iter) + 'subsample.nii'
                else:
                    output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen.nii'

                result_img.to_filename(os.path.join(output_dir, output_filename))
                i = i + 1
                print(i)

            else:
                print('MISSING SUB: ' + sub)
                missing_subjects.append(sub_code)
        else:
            excluded_subjects.append(sub_code)


if __name__ == "__main__":
    run_analysis()

toc = time.time()
print('total runtime: ' + str(toc-tic))