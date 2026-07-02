"""
Author: Zvi Roth
Date created: 06-05-2024
Purpose: Runs searchlight on single subjects, decoding category.
Designed to run on Seen or Unseen, in case they have unequal number of trials subsamples the condition with more trials.
incorporated code from MSP1: searchlight_category_decoding_subject_level.py
"""

# for decoding category WITHIN VG seen or unseen, if there are unequal trials for seen and unseen, we subsample so we use an equal number of trials.
import math
import os
import operator
import pandas as pd
import numpy as np
from mansfield import get_searchlight_neighbours_matrix
import time
from random import sample
from nilearn.image import index_img
import nilearn
import nilearn.image
from nilearn.input_data import NiftiMasker
import argparse
import warnings

# Suppress warning
warnings.filterwarnings(action='ignore', message='y_pred contains classes not in y_true')

num_iter = 10

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_decoding_categ_subsample/')

vg_or_replay = 'VG'

tic=time.time()
# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'

# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['Face', 'Object']
seen_response = 'TruePositive'  # there was a face/object and the subject saw it
unseen_response = 'FalseNegative'  # there was a face/object but the subject didn't see it
Replay_seen_response = 'TrueNegative'  # correct no-go response

# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'
# Radius of the searchlight sphere
searchlight_radius = 4

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
    unselected_trial_labels = list()
    unselected_run_labels = list()
    num_unselected = 0
    # Loop over sessions to extract relevant trial estimates
    for run in unique_runs:
        # Get stimuli representing the categories in the decoding problem of interest
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')

        stim_1_all = behavioral[behavioral.trial_type == stimulus_categories[0]]
        stim_2_all = behavioral[behavioral.trial_type == stimulus_categories[1]]

        if (vg_or_replay == 'Replay') & (condition == 'seen'):
            selected_stim_1 = stim_1_all.response == Replay_seen_response  # NO-GO SEEN
            selected_stim_2 = stim_2_all.response == Replay_seen_response  # NO-GO SEEN
            # For Replay we don't subsample:
            unselected_stim_1 = selected_stim_1
            unselected_stim_2 = selected_stim_2
        elif condition == 'seen':
            selected_stim_1 = stim_1_all.response == seen_response  # SEEN
            selected_stim_2 = stim_2_all.response == seen_response  # SEEN
            # collect trials for UNSEEN, to subsample
            unselected_stim_1 = stim_1_all.response == unseen_response  # UNSEEN
            unselected_stim_2 = stim_2_all.response == unseen_response  # UNSEEN
        elif condition == 'unseen':
            selected_stim_1 = stim_1_all.response == unseen_response  # UNSEEN
            selected_stim_2 = stim_2_all.response == unseen_response  # UNSEEN
            # collect trials for UNSEEN, to subsample
            unselected_stim_1 = stim_1_all.response == seen_response  # SEEN
            unselected_stim_2 = stim_2_all.response == seen_response  # SEEN
        elif condition == 'all':
            selected_stim_1 = stim_1_all.response != -1  # ALL
            selected_stim_2 = stim_2_all.response != -1  # ALL
            # For 'all' we don't subsample:
            unselected_stim_1 = selected_stim_1
            unselected_stim_2 = selected_stim_2
        else:
            raise Exception("unrecognized condition, should be 'unseen', 'seen', or 'all'. ")

        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('ses-V1') == -1: #older V1 files may exist in these folders
                if filename.find(vg_or_replay + '_run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1)
                    beta_maps_list.append(nibetaseries_stim_1_files)

                if filename.find(vg_or_replay + '_run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2)
                    beta_maps_list.append(nibetaseries_stim_2_files)

        # Create labels corresponding to the selected trial estimates
        trial_labels = trial_labels + [stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2)
        run_labels = run_labels + [run] * len([stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2))

        # Create labels corresponding to the UNSELECTED trial estimates
        # should reach a total of 20 trials per run.
        unselected_trial_labels = unselected_trial_labels + [stimulus_categories[0]] * sum(unselected_stim_1) + [stimulus_categories[1]] * sum(unselected_stim_2)
        unselected_run_labels = unselected_run_labels + [run] * len([stimulus_categories[0]] * sum(unselected_stim_1) + [stimulus_categories[1]] * sum(unselected_stim_2))

    num_selected = len(trial_labels)
    num_unselected = max(num_unselected, len(unselected_trial_labels))

    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels, num_selected, num_unselected

def searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, searchlight_neighbours_matrix, masker):
    import nilearn.decoding
    from sklearn.metrics import accuracy_score
    approach = 'within_condition'
    print("Running searchlight")

    data = masker.fit_transform(beta_maps)
    voxels = range(data.shape[1])
    searchlight_scores = np.zeros((1, data.shape[1]))
    # Loop over voxels and get the corresponding accuracies

    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        data_in_the_sphere = data[:, sphere_indices]
        scores = classification(data_in_the_sphere, trial_labels, run_labels, [], classifier, approach)
        searchlight_scores[:, center_voxel] = scores.mean()

    # Create a searchlight image with accuracies corresponding to each voxel
    searchlight_img = masker.inverse_transform(searchlight_scores)
    print("returning the scores")
    return searchlight_img

def searchlight_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, searchlight_radius, searchlight_neighbours_matrix, masker):
    import nilearn.decoding
    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score

    print("Running searchlight")
    data1 = masker.fit_transform(beta_maps1)
    data2 = masker.fit_transform(beta_maps2)
    voxels = range(data1.shape[1])
    searchlight_scores = np.zeros((1, data1.shape[1]))
    # Loop over voxels and get the corresponding accuracies
    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        data_in_the_sphere1 = data1[:, sphere_indices]
        data_in_the_sphere2 = data2[:, sphere_indices]
        predicted_labels = classification(data_in_the_sphere1, trial_labels1, [], data_in_the_sphere2, classifier, approach)
        searchlight_scores[:, center_voxel] = accuracy_score(predicted_labels, trial_labels2)
    # Create a searchlight image with accuracies corresponding to each voxel
    searchlight_img = masker.inverse_transform(searchlight_scores)

    return searchlight_img

def classification(data1, trial_labels1, run_labels1, data2, clf, approach):
    # Classifies trial estimates, depending on the selected approach, either using
    # 1) leave-one-run-out cross validation scheme or 2) specified training and testing sets
    from sklearn import svm
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import cross_val_score
    cv = LeaveOneGroupOut()
    # Classification Options
    if clf == 'SVM':
        classifier = svm.SVC(kernel='linear', class_weight='balanced')
    elif clf == 'LDA':
        classifier = LDA()
    elif clf == 'NB':
        classifier = GaussianNB()
    elif clf == 'LR':
        classifier = LogisticRegression()
    elif clf == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif clf == 'RF':
        classifier = RandomForestClassifier()

    if approach =='within_condition':
        scores = cross_val_score(classifier, data1, trial_labels1, cv=cv, groups=run_labels1, scoring='balanced_accuracy')
    elif approach == 'generalization':
        classifier.fit(data1, trial_labels1)
        scores = classifier.predict(data2)
    return scores

os.makedirs(output_dir, exist_ok=True)

def single_subject_analysis():
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight analysis for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                        help="Name of the subject")
    parser.add_argument('--condition', type=str, default=None,
                        help="seen-seen, seen-unseen, unseen, etc.")
    args = parser.parse_args()
    sub_code = args.subject
    condition = args.condition

    print('running location decoding (subsampled). ' + sub_code + '  ' + vg_or_replay + ' ' + condition + ' ' + str(num_iter) + ' iterations')

    # Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
    if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
        approach = 'within_condition'
    elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay'):
        approach = 'generalization'
    else:
        raise Exception("unrecognized vg_or_replay value, should be 'VG', 'Replay', 'Replay-VG', or 'VG-Replay'. ")
    # Number of runs
    if vg_or_replay == 'VG':
        number_of_runs = 8
    elif vg_or_replay == 'Replay':
        number_of_runs = 4
    elif vg_or_replay == 'Replay-VG':
        number_of_runs = [4, 8]
    elif vg_or_replay == 'VG-Replay':
        number_of_runs = [8, 4]

    sub = 'sub-' + sub_code
    # Define lists to be filled with subject-specific data
    mask_filename = list()
    nibetaseries_filename = list()
    if approach == 'within_condition':
        tsv_filename = list()
    if approach == 'generalization':
        tsv_filename1 = list()
        tsv_filename2 = list()

    for root, sess_dirs, session_files in os.walk(os.path.join(nibetaseries_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            #if ((sess_directory.find('ses-V2') != -1) &  operator.not_(os.path.isdir(os.path.join(output_dir, sub,'ses-V2')))):
            if (sess_directory.find('ses-V2') != -1):
                # Loop over functional mask files in fmriprep directory of the subject
                for filecnt, filename in enumerate(
                        os.listdir(os.path.join(preprocessed_dir, sub, sess_directory, 'func'))):
                    if filename.find('MNI152NLin2009cAsym_desc-brain_mask.nii.gz') != -1:
                        mask_filename.append(os.path.join(preprocessed_dir, sub, sess_directory, 'func', filename))
                # Loop over single trial estimates files in nibetaseries directory of the subject
                for filecnt, filename in enumerate( os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                    if ((filename.find('Face') != -1) | (filename.find('Object') != -1)):
                        nibetaseries_filename.append(os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))

                mask_filename.sort()
                nibetaseries_filename.sort()

                # Create a mask so that the searchlight analysis is performed within the brain voxels only
                print("creating mask...", end="")
                func_mask = nilearn.masking.intersect_masks(mask_filename, threshold=1)
                masker = NiftiMasker(mask_img=func_mask, standardize=True)
                func_mask.to_filename('func_mask_' + sub_code + '.nii')
                print("done")
                # Get neighbours of each voxel in the brain
                print("getting neighbors...", end="")
                searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask_' + sub_code + '.nii',
                                                                                  radius=searchlight_radius)
                print("done")

                if approach == 'generalization':
                    vg_or_replay_list = vg_or_replay.split('-')
                    vg_or_replay1 = vg_or_replay_list[0]
                    vg_or_replay2 = vg_or_replay_list[1]

                    # prepare betas and labels separately for VG and Replay:
                    # Loop over event tsv files of the subject
                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay_list[0]) != -1) and (filename.find('_v2') == -1):
                            tsv_filename1.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))
                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay_list[1]) != -1) and (filename.find('_v2') == -1):
                            tsv_filename2.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))
                    tsv_filename1.sort()
                    tsv_filename2.sort()

                    print("Running " + sub)
                    beta_maps1, trial_labels1, run_labels1, num_selected1, num_unselected1 = prepare_nibetaseries_data(vg_or_replay_list[0], nibetaseries_filename, tsv_filename1, condition, stimulus_categories, number_of_runs[0])
                    beta_maps2, trial_labels2, run_labels2, num_selected2, num_unselected2 = prepare_nibetaseries_data(vg_or_replay_list[1], nibetaseries_filename, tsv_filename2, condition, stimulus_categories, number_of_runs[1])

                    if num_unselected1+num_unselected2 < num_selected1+num_selected2:  # need to subsample
                        print('Subsampling')
                        subsample_img = list()
                        for i_iter in range(num_iter):
                            print('iter ' + str(i_iter))
                            if num_unselected1 < num_selected1:
                                sampled_ind1 = sample(range(num_selected1), num_unselected1)  # for training data
                                #print('sampled indices 1: ' + str(sampled_ind1))
                                sampled_beta_maps1 = index_img(beta_maps1, sampled_ind1)
                                sampled_trial_labels1 = list()
                                sampled_run_labels1 = list()
                                for x in sampled_ind1:
                                    sampled_trial_labels1.append(trial_labels1[x])
                                    sampled_run_labels1.append(run_labels1[x])
                            else:  # no need to subsample training data
                                sampled_trial_labels1 = trial_labels1
                                sampled_run_labels1 = run_labels1
                                sampled_beta_maps1 = beta_maps1
                            if num_unselected2 < num_selected2: #need to subsample test data
                                sampled_ind2 = sample(range(num_selected2), num_unselected2)  # for test data
                                #print('sampled indices 2: ' + str(sampled_ind2))
                                sampled_beta_maps2 = index_img(beta_maps2, sampled_ind2)
                                sampled_trial_labels2 = list()
                                sampled_run_labels2 = list()
                                for x in sampled_ind2:
                                    sampled_trial_labels2.append(trial_labels2[x])
                                    sampled_run_labels2.append(run_labels2[x])
                            else:  # no need to subsample test data
                                sampled_trial_labels2 = trial_labels2
                                sampled_run_labels2 = run_labels2
                                sampled_beta_maps2 = beta_maps2
                            subsample_img.append(searchlight_decoding_generalization(sampled_beta_maps1, sampled_trial_labels1, sampled_run_labels1,
                                                                                     sampled_beta_maps2, sampled_trial_labels2, sampled_run_labels2,
                                                                                    classifier, searchlight_radius, searchlight_neighbours_matrix, masker))
                        searchlight_img = nilearn.image.mean_img(subsample_img)  # mean over iterations
                    else:
                        searchlight_img = searchlight_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, searchlight_radius, searchlight_neighbours_matrix, masker)

                elif approach == 'within_condition':
                    # Loop over event tsv files of the subject
                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay) != -1) and (filename.find('_v2') == -1):
                            #print('found a tsv file\n')
                            tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))

                    tsv_filename.sort()
                    print("Running " + sub)
                    beta_maps, trial_labels, run_labels, num_selected, num_unselected = prepare_nibetaseries_data(vg_or_replay, nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs)
                    print('num_selected: ' + str(num_selected))
                    print('num_unselected: ' + str(num_unselected))
                    if num_unselected < num_selected:  # need to subsample
                        print('Subsampling')
                        subsample_img = list()
                        for i_iter in range(num_iter):
                            print('iter ' + str(i_iter))
                            sampled_ind = sample(range(num_selected), num_unselected)  # indices out of both categories
                            #print('sampled indices: ' + str(sampled_ind))
                            sampled_beta_maps = index_img(beta_maps, sampled_ind)
                            sampled_trial_labels = list()
                            sampled_run_labels = list()
                            for x in sampled_ind:
                                sampled_trial_labels.append(trial_labels[x])
                                sampled_run_labels.append(run_labels[x])

                            temp_img = (searchlight_decoding_crossval(sampled_beta_maps, sampled_trial_labels, sampled_run_labels,
                                                                                     classifier, searchlight_radius, searchlight_neighbours_matrix, masker))
                            subsample_img.append(temp_img)
                        searchlight_img_orig = nilearn.image.mean_img(subsample_img)  # mean over iterations

                    else:
                        searchlight_img_orig = searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, searchlight_neighbours_matrix, masker)

                    # make image 3D by removing singleton 4th dimension:
                    if len(searchlight_img_orig.shape) == 4 and searchlight_img_orig.shape[3] == 1:
                        print('squeezing')
                        # "squeeze" the image.
                        data = nilearn.image.get_data(searchlight_img_orig)
                        affine = searchlight_img_orig.affine
                        searchlight_img = nilearn.image.new_img_like(searchlight_img_orig, data[:, :, :, 0], affine)
                    else:
                        searchlight_img = searchlight_img_orig

                # Save searchlight image as a nifti file
                output_filename = 'searchlight_' + sub + '_' + vg_or_replay + '_category_' + condition + '_' + approach + '_' + str(num_iter) + 'subsample.nii'
                searchlight_img.to_filename(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
  single_subject_analysis()
toc = time.time()
print('subject runtime: ' + str(toc-tic))