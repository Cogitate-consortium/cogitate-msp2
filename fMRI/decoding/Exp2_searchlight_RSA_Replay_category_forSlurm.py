"""
Author: Zvi Roth
Date created: 06-05-2024
Purpose: correlation analysis, test whether correlations between categories within experiments are smaller than correlations between experiments within categories.
2 categories, 2 experiments (AT seen go, AT seen nogo)
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

import sys
import argparse

tic=time.time()

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_RSA_Replay_category/')

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
        # Get behavioral responses
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')

        stim_1_all = behavioral[behavioral.trial_type == stimulus_categories[0]]  # face trials
        stim_2_all = behavioral[behavioral.trial_type == stimulus_categories[1]]  # object trials

        if (vg_or_replay == 'Replay') & (condition == 'seen_go'):
            selected_stim_1 = stim_1_all.response == Replay_seen_go_response  # GO SEEN (correct)
            selected_stim_2 = stim_2_all.response == Replay_seen_go_response  # GO SEEN (correct)
        elif (vg_or_replay == 'Replay') & (condition == 'seen_nogo'):
            selected_stim_1 = stim_1_all.response == Replay_seen_nogo_response  # NO-GO SEEN (correct)
            selected_stim_2 = stim_2_all.response == Replay_seen_nogo_response  # NO-GO SEEN (correct)
        elif (vg_or_replay == 'Replay') & (condition == 'seen'):
            selected_stim_1 = (stim_1_all.response == Replay_seen_nogo_response) | (
                    stim_1_all.response == Replay_seen_go_response)  # CORRECT (go or no-go)
            selected_stim_2 = (stim_2_all.response == Replay_seen_nogo_response) | (
                    stim_2_all.response == Replay_seen_go_response)  # CORRECT (go or no-go)
        elif (vg_or_replay == 'VG') & (condition == 'seen'):  # VG, not Replay
            selected_stim_1 = stim_1_all.response == seen_response  # SEEN
            selected_stim_2 = stim_2_all.response == seen_response  # SEEN
        elif (vg_or_replay == 'VG') & (condition == 'unseen'):
            selected_stim_1 = stim_1_all.response == unseen_response  # UNSEEN
            selected_stim_2 = stim_2_all.response == unseen_response  # UNSEEN
        else:
            raise Exception("unrecognized condition, should be 'unseen', 'seen', or 'seen_nogo'. ")

        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('ses-V1') == -1:  # ignore older V1 files that may exist in these folders.
                # Face trials
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1)
                    beta_maps_list.append(nibetaseries_stim_1_files)
                # Object trials
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2)
                    beta_maps_list.append(nibetaseries_stim_2_files)
        # Create labels corresponding to the selected trial estimates
        trial_labels = trial_labels + [stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2)
        run_labels = run_labels + [run] * len([stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2))

    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels

def searchlight_vectors_category(beta_maps, trial_labels, searchlight_size, searchlight_neighbours_matrix, masker):
    print("Running searchlight")

    data = masker.fit_transform(beta_maps)
    voxels = range(data.shape[1])

    Face_searchlight = np.full([searchlight_size, data.shape[1]], np.nan) #at each voxel we save the sphere vector
    Object_searchlight = np.full([searchlight_size, data.shape[1]], np.nan)

    # Loop over voxels and get the corresponding vectors
    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        # average over trials for each category:
        Face_vector = category_vector(data[:, sphere_indices], 'Face', trial_labels)
        Object_vector = category_vector(data[:, sphere_indices], 'Object', trial_labels)
        # insert vectors into searchlight images
        Face_searchlight[0:Face_vector.size, center_voxel] = Face_vector
        Object_searchlight[0:Object_vector.size, center_voxel] = Object_vector
    Face_img = masker.inverse_transform(Face_searchlight)
    Object_img = masker.inverse_transform(Object_searchlight)

    print("returning the searchlight images")
    return Face_img, Object_img

def category_vector(data, cond_name, trial_labels):
    #return single average vector for given category
    import numpy as np

    cond_trials = [i for i, item in enumerate(trial_labels) if item == cond_name]  # list of trials of this condition
    cond_data = data[cond_trials, :]
    cond_pattern = np.ndarray.mean(cond_data, axis=0)  # average over trials

    return cond_pattern

#create output directory
os.makedirs(output_dir, exist_ok=True)

# we compare vectors across 2 tasks:
conditions_list = ['seen_go', 'seen_nogo']

def single_subject_analysis():
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight RSA analysis for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                        help="Name of the subject")
    args = parser.parse_args()
    sub_code = args.subject

    print('running category RSA on Replay only: ' + sub_code)

    sub = 'sub-' + sub_code
    print("Running " + sub)
    # Define lists to be filled with subject-specific data
    mask_filename = list()
    face_imgs = list()
    object_imgs = list()
    categ_corr = list()
    face_corr = list()
    object_corr = list()
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
                #masker = NiftiMasker(mask_img=func_mask, standardize=True)
                masker = NiftiMasker(mask_img=func_mask, standardize=False) #standardizing includes zscore, which causes Face to be negatively correlated to Object
                func_mask.to_filename('func_mask_' + sub_code + '.nii')
                print("done")

                # Get neighbours of each voxel in the brain
                print("getting neighbors...", end="")
                searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask_' + sub_code + '.nii',
                                                                                  radius=searchlight_radius)
                print("done")

                #for vg_or_replay, condition, number_of_runs in zip(experiments_list,conditions_list,num_runs_list):
                vg_or_replay = 'Replay'
                number_of_runs = 4
                for condition in conditions_list:
                    # Loop over betas files of the subject
                    nibetaseries_filename = list()
                    for filecnt, filename in enumerate(os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                        if (filename.find('Face') != -1) | (filename.find('Object') != -1):
                            nibetaseries_filename.append(os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))
                    nibetaseries_filename.sort()
                    tsv_filename = list()
                    # Loop over event tsv files of the subject
                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay) != -1) and (filename.find('_v2') == -1):
                            # print('found a tsv file\n')
                            tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))
                    tsv_filename.sort()
                    beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(vg_or_replay, nibetaseries_filename,
                                                                                    tsv_filename, condition,
                                                                                    stimulus_categories, number_of_runs)

                    face_img, object_img = searchlight_vectors_category(beta_maps, trial_labels, searchlight_size, searchlight_neighbours_matrix, masker)

                    face_imgs.append(face_img)
                    object_imgs.append(object_img)
                    # compute correlations between categories
                    categ_img = nilearn_correlation(face_img, object_img)
                    categ_corr.append(categ_img)

    #compute correlations between tasks
    all_pairs = list(itertools.combinations([0, 1], 2))
    for task1, task2 in all_pairs:
        face_corr.append(nilearn_correlation(face_imgs[task1], face_imgs[task2]))
        object_corr.append(nilearn_correlation(object_imgs[task1], object_imgs[task2]))

    #average correlations across experiments
    categ_corr_img = image.concat_imgs(categ_corr)  # 2 values, 1 value per experiment (per voxel)
    mean_categ_corr = image.mean_img(categ_corr_img)  # 1 value per voxel

    face_corr_img = image.concat_imgs(face_corr)
    mean_face_corr = image.mean_img(face_corr_img)  # 1 value per voxel
    object_corr_img = image.concat_imgs(object_corr)
    mean_object_corr = image.mean_img(object_corr_img)  # 1 value per voxel

    corr_img = image.concat_imgs([mean_face_corr, mean_object_corr, mean_categ_corr])

    # Save searchlight images as nifti files
    output_filename = 'searchlight_RSA_Replay_category_' + sub + '.nii'
    corr_img.to_filename(os.path.join(output_dir, output_filename))
    mean_corr = image.mean_img(image.concat_imgs([mean_face_corr, mean_object_corr]))#mean over both categories
    output_filename = 'searchlight_RSA_Replay_meanCateg_' + sub + '.nii'
    #mean_corr.to_filename(os.path.join(output_dir, output_filename))
    diff_img = image.math_img("img1 - img2", img1=mean_corr, img2=mean_categ_corr)
    output_filename = 'searchlight_RSA_Replay_category_diff_' + sub + '.nii'
    diff_img.to_filename(os.path.join(output_dir, output_filename))

    Face_diff_img = image.math_img("img1 - img2", img1=mean_face_corr, img2=mean_categ_corr)
    output_filename = 'searchlight_RSA_Replay_category_Face_diff_' + sub + '.nii'
    Face_diff_img.to_filename(os.path.join(output_dir, output_filename))
    Object_diff_img = image.math_img("img1 - img2", img1=mean_object_corr, img2=mean_categ_corr)
    output_filename = 'searchlight_RSA_Replay_category_Object_diff_' + sub + '.nii'
    Object_diff_img.to_filename(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    single_subject_analysis()

toc = time.time()
print('subject runtime: ' + str(toc - tic))