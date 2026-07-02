"""
Author: Zvi Roth
Date created: 06-05-2024
Purpose: correlation analysis, test whether correlations between categories within experiments are smaller than correlations between experiments within categories.
2 categories, 3 experiments (AT seen go, AT seen nogo, dAT seen)
Approach: For each condition save the sphere vector at each voxel. Then compute correlations between conditions.
"""
import os
import operator
import pandas as pd
import numpy as np
from mansfield import get_searchlight_neighbours_matrix
import time
from nilearn.masking import intersect_masks
from nilearn.input_data import NiftiMasker
import itertools
from nilearn import image
import argparse

tic=time.time()

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_RSA_Replay_location/')

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'

stimulus_categories = ['Face', 'Object']
stimulus_location = ['TopLeft', 'BottomLeft', 'TopRight', 'BottomRight']
location_labels = ['Left', 'Right']

seen_response = 'TruePositive'  # there was a face/object and the subject saw it
unseen_response = 'FalseNegative'  # there was a face/object but the subject didn't see it
Replay_seen_nogo_response = 'TrueNegative'  # correct no-go response
Replay_seen_go_response = 'TruePositive'  # correct go response

# Radius of the searchlight sphere
searchlight_radius = 4
searchlight_size = 29


def nilearn_correlation(img1, img2): # converts nilearn images to ndarrays, performs correlation, and converts back to nilearn
    from nilearn import image

    data1 = image.get_data(img1)
    data2 = image.get_data(img2)
    # Calculate the mean along the 4th dimension
    mean_data1 = data1.mean(axis=-1)
    mean_data2 = data2.mean(axis=-1)
    # Compute the cross-correlation
    corrs = (data1 * data2).mean(axis=-1) - mean_data1 * mean_data2
    # Compute the standard deviation along the 4th dimension
    std_prod = np.sqrt(np.var(data1, axis=-1) * np.var(data2, axis=-1))
    # Compute the final correlation
    out = corrs / std_prod
    # Replace NaN values with 0
    out = np.where(np.isnan(out), 0, out)
    # The resulting 3D array 'out' contains the correlations
    new_img = image.new_img_like(img1, out)
    return new_img

def prepare_nibetaseries_data(vg_or_replay, nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs):
    # Extracts nibetaseries trial estimates
    from nilearn.image import index_img
    import nibabel as nb
    # Get number of runs
    unique_runs = range(number_of_runs)
    # Define lists to be filled with relevant trial estimates and the corresponding labels
    beta_maps_list = list()
    trial_labels = list()
    run_labels = list()
    # Loop over sessions to extract relevant trial estimates
    for run in unique_runs:
        i = 0
        # Get stimuli representing the categories in the decoding problem of interest
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')
        stim_all = behavioral
        location_1_boolean = (stim_all.stimulusLocation == stimulus_location[0]) | (
                    stim_all.stimulusLocation == stimulus_location[1])
        location_2_boolean = (stim_all.stimulusLocation == stimulus_location[2]) | (
                    stim_all.stimulusLocation == stimulus_location[3])
        if (vg_or_replay == 'Replay') & (condition == 'seen_go'):
            selected_stim_1 = (stim_all.response == Replay_seen_go_response) & location_1_boolean  # GO SEEN
            selected_stim_2 = (stim_all.response == Replay_seen_go_response) & location_2_boolean  # GO SEEN
        elif (vg_or_replay == 'Replay') & (condition == 'seen_nogo'):
            selected_stim_1 = (stim_all.response == Replay_seen_nogo_response) & location_1_boolean  # NO-GO SEEN
            selected_stim_2 = (stim_all.response == Replay_seen_nogo_response) & location_2_boolean  # NO-GO SEEN
        elif (vg_or_replay == 'Replay') & (condition == 'seen'):
            selected_stim_1 = ((stim_all.response == Replay_seen_nogo_response) | (
                    stim_all.response == Replay_seen_go_response)) & location_1_boolean  # SEEN left
            selected_stim_2 = ((stim_all.response == Replay_seen_nogo_response) | (
                    stim_all.response == Replay_seen_go_response)) & location_2_boolean  # SEEN right
        elif condition == 'seen':  # VG, not replay
            selected_stim_1 = (stim_all.response == seen_response) & location_1_boolean  # SEEN left
            selected_stim_2 = (stim_all.response == seen_response) & location_2_boolean  # SEEN right
        elif condition == 'unseen':
            selected_stim_1 = (stim_all.response == unseen_response) & location_1_boolean  # UNSEEN left
            selected_stim_2 = (stim_all.response == unseen_response) & location_2_boolean  # UNSEEN right
        else:
            raise Exception("unrecognized condition, should be 'unseen', or 'seen'. ")

        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('ses-V1') == -1:  # ignore older V1 files that may exist in these folders.
                # for Face trials, add betas corresponding to Left locations, and to Right locations
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                    selected_stim_1_Face = (selected_stim_1[behavioral.trial_type == stimulus_categories[0]])
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1_Face)
                    beta_maps_list.append(nibetaseries_stim_1_files)
                    selected_stim_2_Face = (selected_stim_2[behavioral.trial_type == stimulus_categories[0]])
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2_Face)
                    beta_maps_list.append(nibetaseries_stim_2_files)

                # for Object trials, add betas corresponding to Left locations, and to Right locations
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                    selected_stim_1_Object = (selected_stim_1[behavioral.trial_type == stimulus_categories[1]])
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1_Object)
                    beta_maps_list.append(nibetaseries_stim_1_files)
                    selected_stim_2_Object = (selected_stim_2[behavioral.trial_type == stimulus_categories[1]])
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2_Object)
                    beta_maps_list.append(nibetaseries_stim_2_files)
        # Create labels corresponding to the selected trial estimates, first for Face trials and then for Object trials
        trial_labels = trial_labels + [location_labels[0]] * sum(selected_stim_1_Face) + [location_labels[1]] * sum(
            selected_stim_2_Face)
        run_labels = run_labels + [run] * len(
            [location_labels[0]] * sum(selected_stim_1_Face) + [location_labels[1]] * sum(selected_stim_2_Face))
        trial_labels = trial_labels + [location_labels[0]] * sum(selected_stim_1_Object) + [location_labels[1]] * sum(
            selected_stim_2_Object)
        run_labels = run_labels + [run] * len(
            [location_labels[0]] * sum(selected_stim_1_Object) + [location_labels[1]] * sum(selected_stim_2_Object))

    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels

def searchlight_vectors_category(beta_maps, trial_labels, searchlight_size, searchlight_neighbours_matrix, masker):
    print("Running searchlight")

    data = masker.fit_transform(beta_maps)
    voxels = range(data.shape[1])

    Left_searchlight = np.full([searchlight_size, data.shape[1]], np.nan)
    Right_searchlight = np.full([searchlight_size, data.shape[1]], np.nan)

    # Loop over voxels and get the corresponding vectors
    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        # average over trials for each category:
        Left_vector = category_vector(data[:, sphere_indices], 'Left', trial_labels)
        Right_vector = category_vector(data[:, sphere_indices], 'Right', trial_labels)
        Left_searchlight[0:Left_vector.size, center_voxel] = Left_vector
        Right_searchlight[0:Right_vector.size, center_voxel] = Right_vector
    Left_img = masker.inverse_transform(Left_searchlight)
    Right_img = masker.inverse_transform(Right_searchlight)

    print("returning the searchlight vectors")
    return Left_img, Right_img

def category_vector(data, cond_name, trial_labels):
    #return single average vector for given category
    import numpy as np
    cond_trials = [i for i, item in enumerate(trial_labels) if item == cond_name]  # list of trials of this condition
    cond_data = data[cond_trials, :]
    cond_pattern = np.ndarray.mean(cond_data, axis=0)
    return cond_pattern



conditions_list = ['seen_go', 'seen_nogo']



os.makedirs(output_dir, exist_ok=True)

def single_subject_analysis():
    vg_or_replay = 'Replay'
    number_of_runs=4
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight RSA analysis for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                        help="Name of the subject")
    args = parser.parse_args()
    sub_code = args.subject

    print('running location RSA: ' + sub_code)

    sub = 'sub-' + sub_code
    print("Running " + sub)
    # Define lists to be filled with subject-specific data
    mask_filename = list()
    left_imgs = list()
    right_imgs = list()
    location_corr = list()
    left_corr = list()
    right_corr = list()
    for root, sess_dirs, session_files in os.walk(os.path.join(nibetaseries_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            if (sess_directory.find('ses-V2') != -1):

                # Loop over functional mask files in fmriprep directory of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(preprocessed_dir, sub, sess_directory, 'func'))):
                    if filename.find('MNI152NLin2009cAsym_desc-brain_mask.nii.gz') != -1:
                        mask_filename.append(os.path.join(preprocessed_dir, sub, sess_directory, 'func', filename))

                mask_filename.sort()

                # Create a mask so that the searchlight analysis is performed within the brain voxels only
                print("creating mask...", end="")
                func_mask = intersect_masks(mask_filename, threshold=1)
                # masker = NiftiMasker(mask_img=func_mask, standardize=True)
                masker = NiftiMasker(mask_img=func_mask, standardize=False)  # standardizing includes zscore, which causes Face to be negatively correlated to Object
                func_mask.to_filename('func_mask_' + sub_code + '.nii')
                print("done")

                # Get neighbours of each voxel in the brain
                print("getting neighbors...", end="")
                searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask_' + sub_code + '.nii',
                                                                                  radius=searchlight_radius)
                print("done")

                for condition in conditions_list:
                    # Loop over betas files of the subject
                    nibetaseries_filename = list()
                    for filecnt, filename in enumerate(
                            os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                        if (filename.find('Face') != -1) | (filename.find('Object') != -1):
                            nibetaseries_filename.append(
                                os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))
                    nibetaseries_filename.sort()
                    tsv_filename = list()
                    # Loop over event tsv files of the subject
                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay) != -1) and (
                                filename.find('_v2') == -1):
                            tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))
                    tsv_filename.sort()

                    beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(vg_or_replay, nibetaseries_filename,
                                                                                    tsv_filename, condition,
                                                                                    stimulus_categories, number_of_runs)

                    left_img, right_img = searchlight_vectors_category(beta_maps, trial_labels, searchlight_size, searchlight_neighbours_matrix, masker)
                    left_imgs.append(left_img)
                    right_imgs.append(right_img)
                    # compute correlations between categories
                    location_img = nilearn_correlation(left_img, right_img)
                    location_corr.append(location_img)

    #compute correlations between tasks
    all_pairs = list(itertools.combinations([0, 1], 2))
    for task1, task2 in all_pairs:
        left_corr.append(nilearn_correlation(left_imgs[task1], left_imgs[task2]))
        right_corr.append(nilearn_correlation(right_imgs[task1], right_imgs[task2]))

    #average correlations across tasks
    location_corr_img = image.concat_imgs(location_corr) # 2 values, 1 value per task (per voxel)
    mean_location_corr = image.mean_img(location_corr_img) # 1 value per voxel

    left_corr_img = image.concat_imgs(left_corr)
    mean_left_corr = image.mean_img(left_corr_img) # 1 value per voxel
    right_corr_img = image.concat_imgs(right_corr)
    mean_right_corr = image.mean_img(right_corr_img)  # 1 value per voxel

    corr_img = image.concat_imgs([mean_left_corr, mean_right_corr, mean_location_corr])
    # Save searchlight image as a nifti file
    output_filename = 'searchlight_RSA_Replay_location_' + sub + '.nii'
    corr_img.to_filename(os.path.join(output_dir, output_filename))

    mean_corr = image.mean_img(image.concat_imgs([mean_left_corr, mean_right_corr]))  # mean over both locations
    output_filename = 'searchlight_RSA_Replay_meanLocation_' + sub + '.nii'
    mean_corr.to_filename(os.path.join(output_dir, output_filename))
    diff_img = image.math_img("img1 - img2", img1=mean_corr, img2=mean_location_corr)
    output_filename = 'searchlight_RSA_Replay_location_diff_' + sub + '.nii'
    diff_img.to_filename(os.path.join(output_dir, output_filename))

    Left_diff_img = image.math_img("img1 - img2", img1=mean_left_corr, img2=mean_location_corr)
    output_filename = 'searchlight_RSA_Replay_location_Left_diff_' + sub + '.nii'
    Left_diff_img.to_filename(os.path.join(output_dir, output_filename))
    Right_diff_img = image.math_img("img1 - img2", img1=mean_right_corr, img2=mean_location_corr)
    output_filename = 'searchlight_RSA_Replay_location_Right_diff_' + sub + '.nii'
    Right_diff_img.to_filename(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
  single_subject_analysis()

toc = time.time()
print('subject runtime: ' + str(toc-tic))