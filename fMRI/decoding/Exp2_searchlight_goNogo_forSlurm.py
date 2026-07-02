"""
Author: Zvi Roth
Date created: 08-08-2023
"""
import os
import operator
import pandas as pd
import numpy as np
from mansfield import get_searchlight_neighbours_matrix
import time
import argparse
import warnings

# Suppress warning
warnings.filterwarnings(action='ignore', message='y_pred contains classes not in y_true')


tic=time.time()

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'searchlight_decoding_goNogo/')

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'


# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['Face', 'Object']
condition = 'seen'
go_labels = ['GO', 'NOGO']
# probe_labels = ['PROBE', 'NO_PROBE']
#seen_response = 'TruePositive'  # there was a face/object and the subject saw it
#unseen_response = 'FalseNegative'  # there was a face/object but the subject didn't see it
#seen_response = 'TruePositive'  # there was a face/object and the subject saw it
#unseen_response = 'FalseNegative'  # there was a face/object but the subject didn't see it
Replay_seen_nogo_response = 'TrueNegative'  # correct no-go response
Replay_seen_go_response = 'TruePositive'  # correct go response

vg_or_replay = 'Replay'

# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
approach = 'within_condition'
#approach = 'generalization'

# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'
# Radius of the searchlight sphere
searchlight_radius = 4
# Number of runs
number_of_runs = 4
# Scan repetition time
TR = 1.5


def prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs):
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

        # all trials, for category
        # stim_all = behavioral[behavioral.trial_type == stimulus_category]
        stim_all = behavioral
        #  probe_boolean = stim_all.type == 'GAME_PROBE'
        #  noprobe_boolean = stim_all.type == 'GAME_STIMULUS'

        #selected_stim_1 = (stim_all.response == seen_response) & (stim_all.type == 'GAME_PROBE')  # SEEN probe
        #selected_stim_2 = (stim_all.response == unseen_response) & (stim_all.type == 'GAME_PROBE')  # UNSEEN probe

        selected_stim_1 = (stim_all.response == Replay_seen_go_response)
        selected_stim_2 = (stim_all.response == Replay_seen_nogo_response)

        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('ses-V1')==-1:#older ses-V1 files exist in these folders!
                # for Face trials, add betas corresponding to seen trials, and to unseen trials
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                    selected_stim_1_Face = selected_stim_1[behavioral.trial_type == stimulus_categories[0]]
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1_Face)
                    beta_maps_list.append(nibetaseries_stim_1_files)
                    selected_stim_2_Face = selected_stim_2[behavioral.trial_type == stimulus_categories[0]]
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2_Face)
                    beta_maps_list.append(nibetaseries_stim_2_files)

                # for Object trials, add betas corresponding to seen trials, and to unseen trials
                if filename.find(vg_or_replay + '_run-' + str(run + 1) + '_space-MNI152NLin2009cAsym_desc-' +
                                 stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                    selected_stim_1_Object = selected_stim_1[behavioral.trial_type == stimulus_categories[1]]
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1_Object)
                    beta_maps_list.append(nibetaseries_stim_1_files)
                    selected_stim_2_Object = selected_stim_2[behavioral.trial_type == stimulus_categories[1]]
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2_Object)
                    beta_maps_list.append(nibetaseries_stim_2_files)

        # Create labels corresponding to the selected trial estimates, first for Face trials and then for Object trials
        trial_labels = trial_labels + [go_labels[0]] * sum(selected_stim_1_Face) + [go_labels[1]] * sum(selected_stim_2_Face)
        run_labels = run_labels + [run] * len([go_labels[0]] * sum(selected_stim_1_Face) + [go_labels[1]] * sum(selected_stim_2_Face))
        trial_labels = trial_labels + [go_labels[0]] * sum(selected_stim_1_Object) + [go_labels[1]] * sum(selected_stim_2_Object)
        run_labels = run_labels + [run] * len([go_labels[0]] * sum(selected_stim_1_Object) + [go_labels[1]] * sum(selected_stim_2_Object))

    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels

def searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, mask_filename, sub_code):
    import nilearn.decoding
    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score

    print("Running searchlight")
    # Create a mask so that the searchlight analysis is performed within the brain voxels only
    print("creating mask...", end="")
    func_mask = nilearn.masking.intersect_masks(mask_filename, threshold=1)
    masker = NiftiMasker(mask_img=func_mask, standardize=True)
    func_mask.to_filename('func_mask_' + sub_code + '.nii')
    print("done")
    # Get neighbours of each voxel in the brain
    print("getting neighbors...", end="")
    searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask_' + sub_code + '.nii', radius= searchlight_radius)
    print("done")
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


    scores = cross_val_score(classifier, data1, trial_labels1, cv=cv, groups=run_labels1, scoring='balanced_accuracy')

    return scores

#create output directory
os.makedirs(output_dir, exist_ok=True)


def single_subject_analysis():
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight analysis for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                        help="Name of the subject")

    args = parser.parse_args()
    sub_code = args.subject
    vg_or_replay = 'Replay'
    print('running go vs. nogo decoding: ' + sub_code)

    sub = 'sub-' + sub_code
    # Define lists to be filled with subject-specific data
    mask_filename = list()
    nibetaseries_filename = list()
    tsv_filename = list()

    for root, sess_dirs, session_files in os.walk(os.path.join(nibetaseries_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            if (sess_directory.find('ses-V2') != -1):
                # Loop over functional mask files in fmriprep directory of the subject
                for filecnt, filename in enumerate(
                        os.listdir(os.path.join(preprocessed_dir, sub, sess_directory, 'func'))):
                    if filename.find('MNI152NLin2009cAsym_desc-brain_mask.nii.gz') != -1:
                        mask_filename.append(os.path.join(preprocessed_dir, sub, sess_directory, 'func', filename))
                # Loop over single trial estimates files in nibetaseries directory of the subject
                for filecnt, filename in enumerate(
                        os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                    if ((filename.find('Face') != -1) | (filename.find('Object') != -1)):
                        nibetaseries_filename.append(
                            os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))
                # Loop over event tsv files of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                    if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay) != -1) and (
                            filename.find('_v2') == -1):
                        tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))

                mask_filename.sort()
                nibetaseries_filename.sort()
                tsv_filename.sort()
                print("Running " + sub)

                beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs)
                searchlight_img = searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, mask_filename, sub_code)

                output_filename = 'searchlight_' + sub + '_goNogo.nii'
                searchlight_img.to_filename(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
  single_subject_analysis()
toc = time.time()
print('subject runtime: ' + str(toc-tic))