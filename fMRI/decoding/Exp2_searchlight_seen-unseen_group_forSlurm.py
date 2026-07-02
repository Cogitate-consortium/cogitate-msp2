
# Runs cluster-based permutation testing on the searchlight accuracy maps against the chance level
"""
Author: Zvi Roth
Date created: 14-08-2024
"""

import os
import pandas as pd
import nilearn
import nilearn.image
import argparse
import time

num_iter = 10
#num_iter = 0

#csv_file_end = 'ses-v2-analysis-subs.csv'
csv_file_end = 'ses-v2-decodingOnly-top30.csv'

csv_file = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/qcs/', csv_file_end)
csv_data = pd.read_csv(csv_file, sep=',')
subjects_phase2 = csv_data.sub_code
subject_list = subjects_phase2.tolist()


def second_level_analysis(second_level_input, chance_level, to_flip):
    # Performs second level analysis
    searchlight_images = list()
    # Threshold accuracy maps before running second level GLM
    for index in second_level_input:
        if to_flip:
            searchlight_images.append(nilearn.image.math_img(str(chance_level) + '-a', a=index))
        else:
            searchlight_images.append(nilearn.image.math_img('a-' + str(chance_level), a=index))
        #searchlight_images.append(nilearn.image.math_img('a-' + str(chance_level), a=index))
        #searchlight_images.append(index)
    # Get mean accuracy map across subjects
    accuracy_map = nilearn.image.mean_img(searchlight_images)
    # Create Design matrix
    design_matrix = pd.DataFrame([1] * len(searchlight_images), columns=['intercept'])
    # Run second level analysis
    from nilearn.glm.second_level import non_parametric_inference

    out_dict = non_parametric_inference(
        searchlight_images,
        design_matrix=design_matrix,
        n_perm=5000,
        two_sided_test=False,
        n_jobs=6,
        threshold=0.001,
    )
    logp_map = out_dict["logp_max_size"]
    return logp_map, accuracy_map

def second_level_display(logp_map, accuracy_map, chance_level, output_filename):

    # Displays group-level accuracy map on axial brain slices
    from nilearn import plotting
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    # Define figure parameters
    fig = plt.figure(figsize=(5, 7))
    fig.set_facecolor((0, 0, 0))
    plt.subplots_adjust(hspace=0.001)
    plt.subplots_adjust(wspace=0.001)
    c_map = mpl.cm.hot
    slices = range(-56, 96, 8)

    # MNI template to display group-level accuracy
    file_name = 'mni_icbm152_t1_tal_nlin_asym_09c.nii'
    # Cluster correction threshold
    threshold= -np.log10(0.05)
    # Apply threshold and get the corrected group-level accuracy map
    accuracy_mask = nilearn.image.math_img(f'img > {threshold}', img=logp_map)
    accuracy_map_thresholded = nilearn.image.math_img('a*b', a= accuracy_mask, b=accuracy_map)
    # Save group-level accuracy map
    accuracy_map_thresholded.to_filename(output_filename + '.nii')
    accuracy_map.to_filename(output_filename + '_unthresholded.nii')

    # Plot group accuracy map on axial brain slices
    slice_index = 0
    for index in range(18):
        ax = plt.subplot(5, 4, index + 1)
        plotting.plot_img(accuracy_map_thresholded, cut_coords=range(slices[slice_index], slices[slice_index + 1], 8), bg_img=file_name,
                          axes=ax, annotate=False, threshold=chance_level, display_mode="z",
                          cmap=c_map, colorbar=False, vmin=chance_level, vmax=0.8)

        slice_index = slice_index + 1

    ax = plt.subplot(5, 4, 20)
    fig.subplots_adjust(bottom=0.12, top=0.7, left=0.43, right=0.63)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=c_map),
                      cax=ax, orientation='vertical')

    cb.ax.set_yticklabels([chance_level, 0.65, 0.8], fontsize=10, weight='bold')
    cb.set_label('Accuracy', rotation=90, color='white', fontsize=12, weight='bold')
    cb.ax.tick_params(colors='white')
    cb.ax.tick_params(size=0)
    # Save group accuracy map as a figure
    plt.savefig(output_filename + '.png')
    plotting.show()


searchlight_filename = list()

def run_analysis():
    tic = time.time()
    chance_level = 0.0
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
                output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen_' + str(num_iter) + 'subsample.nii'
            else:
                output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen.nii'
            searchlight_path = os.path.join(output_dir, output_filename)

            if os.path.exists(searchlight_path):
                searchlight_filename.append(searchlight_path)
                print('adding map for sub: ' + sub)
                included_subjects.append(sub_code)

            else:
                print('MISSING SUB: ' + sub)
                missing_subjects.append(sub_code)
        else:
            excluded_subjects.append(sub_code)

    # Perform second level analysis
    stats_map, accuracy_map = second_level_analysis(searchlight_filename, chance_level, False)
    stats_map_flip, accuracy_map_flip = second_level_analysis(searchlight_filename, chance_level, True)

    # Display group level accuracy map on axial brain slices
    if num_iter > 0:
        output_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen_' + str(num_iter) + 'subsample'
    else:
        output_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen'

    # Display group level accuracy map on axial brain slices
    second_level_display(stats_map, accuracy_map, chance_level, os.path.join(output_dir, output_filename))
    second_level_display(stats_map_flip, accuracy_map_flip, chance_level, os.path.join(output_dir, output_filename) + 'flip')

    second_level_display(stats_map, accuracy_map, chance_level, os.path.join(output_dir, output_filename))


    print('total subjects: ' + str(len(included_subjects)))
    print('total missing subjects: ' + str(len(missing_subjects)))
    toc = time.time()
    print('total runtime: ' + str(toc - tic))

if __name__ == "__main__":
    run_analysis()

