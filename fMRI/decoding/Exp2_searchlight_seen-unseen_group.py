
# Runs cluster-based permutation testing on the searchlight accuracy maps against the chance level
"""
Author: Aya Khalaf
Date created: 10-10-2022
Date modified: 10-19-2023 by Zvi
"""

import os
import pandas as pd
import nilearn
import nilearn.image
# num_iter = 10
num_iter = 0
# decoding_problem = 'category'
decoding_problem = 'location'
chance_level=0
searchlight_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_decoding/')

def second_level_analysis(second_level_input,chance_level):
    # Performs second level analysis
    searchlight_images = list()
    # Threshold accuracy maps before running second level GLM
    for index in second_level_input:
        # searchlight_images.append(nilearn.image.math_img('a-' + str(chance_level), a=index))
        searchlight_images.append(index)
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

def second_level_display(logp_map, accuracy_map, chance_level, searchlight_dir):

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
    # accuracy_map_thresholded.to_filename(os.path.join(searchlight_dir, 'searchlight_group_accuracy_map_nonparametric.nii'))
    if num_iter>0:
        output_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen_' + str(num_iter) + 'subsample.nii'
    else:
        output_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen.nii'
    accuracy_map_thresholded.to_filename(os.path.join(searchlight_dir, output_filename))

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
    if num_iter>0:
        png_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen_' + str(num_iter) + 'subsample.png'
    else:
        png_filename = 'searchlight_group_' + decoding_problem + '_seen-unseen.png'
    plt.savefig(os.path.join(searchlight_dir, png_filename))
    plotting.show()


# Get phase3 subjects
# tsv_file = os.path.join(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv')
tsv_file = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Exp2/Zvi/','participants_for_exp2_decoding.tsv')
tsv_data= pd.read_csv(tsv_file, sep='\t')
subjects_phase2 = tsv_data.participant_id
subject_list=subjects_phase2.tolist()

searchlight_filename = list()

# Loop over subjects
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_decoding/')
for sub in subject_list:
    if num_iter > 0:
        output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen_' + str(num_iter) + 'subsample.nii'
    else:
        output_filename = 'searchlight_' + sub + '_' + decoding_problem + '_seen-unseen.nii'
    searchlight_filename.append(os.path.join(output_dir, output_filename))
    print('adding map for sub: ' + sub)

# Perform second level analysis
stats_map, accuracy_map = second_level_analysis(searchlight_filename, chance_level)
# Display group level accuracy map on axial brain slices
second_level_display(stats_map, accuracy_map, chance_level, searchlight_dir)
