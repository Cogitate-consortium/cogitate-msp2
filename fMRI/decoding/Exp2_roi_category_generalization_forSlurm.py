"""
Author: Zvi Roth
Date created: 10-19-2023
"""

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
import warnings
import argparse

use_evc_rois = 0

# Suppress warning
warnings.filterwarnings(action='ignore', message='y_pred contains classes not in y_true')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Number of voxels per ROI (ROI size)
n_voxels = 300

# Select functional ROIs that were created using the data of the selected decoding condition
roi_condition = 'rel'


tic=time.time()
# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'

# From Exp1: roi_category_decoding_subject_level.py
# V1 & V2 ROIs path
evc_rois_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/evc_rois'

# functional ROIs path
func_rois_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding_rois'
# GNW ROI list
GNW_roi_list = ['G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant', 'G_and_S_cingul-Mid-Post', 'G_front_inf-Opercular', 'G_front_inf-Orbital', 'G_front_inf-Triangul', 'G_front_middle', 'Lat_Fis-ant-Horizont', 'Lat_Fis-ant-Vertical', 'S_front_inf', 'S_front_middle', 'S_front_sup']
# IIT Basic ROI list
IIT_roi_list_1 = ['G_temporal_inf', 'Pole_temporal', 'G_cuneus', 'G_occipital_sup', 'G_oc-temp_med-Lingual', 'Pole_occipital', 'S_calcarine', 'G_and_S_occipital_inf', 'G_occipital_middle', 'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Parahip', 'S_intrapariet_and_P_trans', 'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_temporal_sup']
# IIT extended ROI list
IIT_roi_list_2 = ['G_precentral', 'G_temp_sup-Lateral', 'G_temp_sup-Plan_tempo', 'G_pariet_inf-Supramar', 'G_temporal_middle', 'S_temporal_inf', 'G_orbital', 'G_pariet_inf-Angular', 'S_interm_prim-Jensen', 'S_occipital_ant', 'S_oc-temp_lat', 'S_precentral-inf-part']
# IIT excluded ROI List
IIT_roi_list_3 = ['G_and_S_frontomargin', 'G_and_S_transv_frontopol', 'G_front_sup', 'G_rectus', 'G_subcallosal', 'S_orbital_lateral', 'S_orbital_med-olfact', 'S_orbital-H_Shaped', 'S_suborbital']
# IIT ROI list
IIT_roi_list = IIT_roi_list_1 + IIT_roi_list_2 + IIT_roi_list_3
# EVC list
EVC_list = ['evc_300_V1', 'evc_300_V1V2', 'evc_300_V2']

# Create IIT and GNW labels for each ROI
GNW_label = list()
IIT_label = list()
GNW_label = ['GNW'] * len(GNW_roi_list)
IIT_label = ['IIT'] * len(IIT_roi_list)
EVC_label = ['EVC'] * len(EVC_list)
roi_labels_list = GNW_label + IIT_label + EVC_label
selected_roi_list = GNW_roi_list+IIT_roi_list+EVC_list


# GNW/IIT ROIs combining all ROIs that represent GNW/IIT
combined_roi_list =['GNW',  'IIT_extended', 'IIT_excluded', 'IIT']
# Add combined GNW/IIT ROIs to the list and add corresponding labels
roi_labels_list = roi_labels_list + ['GNW', 'IIT', 'IIT', 'IIT']
selected_roi_list = selected_roi_list + combined_roi_list


# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['Face', 'Object']
seen_response = 'TruePositive'  # there was a face/object and the subject saw it
unseen_response = 'FalseNegative'  # there was a face/object but the subject didn't see it
Replay_seen_nogo_response = 'TrueNegative'  # correct no-go response
Replay_seen_go_response = 'TruePositive'  # correct go response

# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'

# Scan repetition time
TR = 1.5
# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/',
                          'roi_decoding/')

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
        # Get stimuli representing the categories in the decoding problem of interest
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')

        stim_1_all = behavioral[behavioral.trial_type == stimulus_categories[0]]
        stim_2_all = behavioral[behavioral.trial_type == stimulus_categories[1]]

        if (vg_or_replay == 'Replay') & (condition == 'seen_go'):
            selected_stim_1 = stim_1_all.response == Replay_seen_go_response  # GO SEEN
            selected_stim_2 = stim_2_all.response == Replay_seen_go_response  # GO SEEN
        elif (vg_or_replay == 'Replay') & (condition == 'seen_nogo'):
            selected_stim_1 = stim_1_all.response == Replay_seen_nogo_response  # NO-GO SEEN
            selected_stim_2 = stim_2_all.response == Replay_seen_nogo_response  # NO-GO SEEN
        elif (vg_or_replay == 'Replay') & (condition == 'seen'):
            selected_stim_1 = (stim_1_all.response == Replay_seen_nogo_response) | (
                        stim_1_all.response == Replay_seen_go_response)  # CORRECT (go or no-go)
            selected_stim_2 = (stim_2_all.response == Replay_seen_nogo_response) | (
                        stim_2_all.response == Replay_seen_go_response)  # CORRECT (go or no-go)
        elif (condition == 'seen') | (condition == 'seen_nogo'):  # VG, not Replay
            selected_stim_1 = stim_1_all.response == seen_response  # SEEN
            selected_stim_2 = stim_2_all.response == seen_response  # SEEN
        elif (vg_or_replay == 'VG') & (condition == 'unseen'):
            selected_stim_1 = stim_1_all.response == unseen_response  # UNSEEN
            selected_stim_2 = stim_2_all.response == unseen_response  # UNSEEN
        elif condition == 'all':
            selected_stim_1 = stim_1_all.response != -1  # ALL
            selected_stim_2 = stim_2_all.response != -1  # ALL
        else:
            raise Exception("unrecognized condition, should be 'unseen', 'seen', 'seen_nogo', or 'all'. ")

        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('ses-V1') == -1:  # ignore older V1 files that may exist in these folders.
                if filename.find(vg_or_replay + '_run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_1_files = index_img(filename, selected_stim_1)
                    beta_maps_list.append(nibetaseries_stim_1_files)

                if filename.find(vg_or_replay + '_run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                    nibetaseries_stim_2_files = index_img(filename, selected_stim_2)
                    beta_maps_list.append(nibetaseries_stim_2_files)

        # Create labels corresponding to the selected trial estimates
        trial_labels = trial_labels + [stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2)
        run_labels = run_labels + [run] * len([stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2))

    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels

def roi_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, func_roi_filename):
    #import nilearn.decoding
    #from sklearn.metrics import accuracy_score
    from nilearn.input_data import NiftiMasker
    approach = 'within_condition'
    #print("Running ROI decoding")
    # Define an empty list to be filled with accuracy scores
    roi_accuracy_score = list()
    # Loop over ROIs and get accuracy scores
    for ind, func_roi in enumerate(func_roi_filename):
        func_masker = NiftiMasker(mask_img=func_roi, standardize=True)
        data = func_masker.fit_transform(beta_maps)
        scores = classification(data, trial_labels, run_labels, [], classifier, approach)
        roi_accuracy_score.append(scores.mean())
    return roi_accuracy_score

def roi_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, func_roi_filename):
    # Decodes stimulus category within each ROI and gives the corresponding accuracy scores
    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import balanced_accuracy_score
    approach ='generalization'
    print("Running ROI Decoding")
    roi_accuracy_score = list()
    for j, func_roi in enumerate(func_roi_filename):
        func_masker = NiftiMasker(mask_img=func_roi, standardize=True)
        data1 = func_masker.fit_transform(beta_maps1)
        data2 = func_masker.fit_transform(beta_maps2)
        predicted_labels = classification(data1, trial_labels1, [], data2, classifier, approach)
        roi_accuracy_score.append(balanced_accuracy_score(predicted_labels, trial_labels2))

    return roi_accuracy_score


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

# Define lists to be filled with ROI information and subject-specific data
i = 0
included_subs = list()
excluded_subs = list()
missing_data = list()
missing_rois = list()
missing_evc = list()

# Create a data frame to save accuracy scores
df = pd.DataFrame(list())

# Get phase2 subjects
#csv_file_end = 'ses-v2-optimization-subs.csv'
csv_file_end = 'ses-v2-analysis-subs.csv'
csv_file = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/qcs/', csv_file_end)
csv_data= pd.read_csv(csv_file, sep=',')
subjects_phase2 = csv_data.sub_code
subject_list=subjects_phase2.tolist()

# use code for decoding location within condition:
analysis_check_data = csv_data.DECODING_min_sf_so
print('using list: DECODING_min_sf_so')
analysis_check_list = analysis_check_data.tolist()

def run_analysis():
    tic = time.time()
    i=0
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Run group fMRI ROI analysis for experiment2")
    parser.add_argument('--vg_or_replay', type=str, default=None,
                        help="VG-Replay, VG-VG, VG, etc.")
    parser.add_argument('--condition', type=str, default=None,
                        help="seen, unseen, seen-seen, unseen-seen, unseen-unseen, etc.")
    args = parser.parse_args()
    condition = args.condition
    vg_or_replay = args.vg_or_replay

    # Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
    if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
        approach = 'within_condition'
    elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay') | (vg_or_replay == 'VG-VG') | (
            vg_or_replay == 'Replay-Replay'):
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
    elif vg_or_replay == 'VG-VG':
        number_of_runs = [8, 8]
    elif vg_or_replay == 'Replay-Replay':
        number_of_runs = [4, 4]
    print('starting to loop over subjects')
    print(subject_list)
    # Loop over subjects
    for sub_code, analysis_check in zip(subject_list, analysis_check_list):
        sub = 'sub-' + sub_code
        if analysis_check:
            # Define lists to be filled with subject-specific data
            nibetaseries_filename = list()
            func_roi_filename = list()
            tsv_filename = list()
            roi_labels = list()
            theory_labels = list()
            mask_filename = list()

            if approach == 'within_condition':
                tsv_filename = list()
            if approach == 'generalization':
                tsv_filename1 = list()
                tsv_filename2 = list()

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
                        for filecnt, filename in enumerate( os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                            if ((filename.find('Face') != -1) | (filename.find('Object') != -1)):
                                nibetaseries_filename.append(os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))

                        mask_filename.sort()
                        nibetaseries_filename.sort()

                        # find EVC ROIs:
                        evcRoiPath = os.path.join(evc_rois_dir, sub)
                        if use_evc_rois:
                            if os.path.exists(evcRoiPath):
                                for filecnt, filename in enumerate(os.listdir(os.path.join(evc_rois_dir, sub))):
                                    if ((filename.find('.nii.gz') != -1) & (
                                            filename.find(stimulus_categories[1].lower()) != -1) & (
                                                filename.find('_V1') != -1) & filename.find(str(n_voxels) + '_') != -1) & (
                                    bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                                        func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                                        index = [selected_roi_list.index(x) for x in selected_roi_list if
                                                 x in filename]  # finds the index in selected_roi_list that correspond to filename
                                        roi_labels.append(selected_roi_list[index[0]]) 
                                        theory_labels.append(roi_labels_list[index[0]])                                       
                            else:
                                missing_evc.append(sub_code)
                                print('MISSING EVC ROIs: ' + sub)
                        roiPath = os.path.join(func_rois_dir, sub)
                        if os.path.exists(roiPath):
                            # Loop over functional ROIs of the subject
                            for filecnt, filename in enumerate(os.listdir(os.path.join(func_rois_dir, sub))):
                                # Select ROIs relevant to the decoding problem of interest
                                if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_categories[0].lower()) != -1) & (
                                        filename.find('GNW_S_front_inf') == -1) & ( #this is a composite ROI we don't analyze here
                                        filename.find(stimulus_categories[1].lower()) != -1) & (
                                        filename.find('_' + roi_condition) != -1) & (filename.find('leave_run') == -1) & (# NOTE: NOT INCLUDING LEAVE ONE OUT!
                                        filename.find('_' + str(n_voxels) + '_') != -1) & bool(
                                        [selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                                    func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                                    index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                                    roi_labels.append(selected_roi_list[index[0]])  
                                    theory_labels.append(roi_labels_list[index[0]])                                    
                            # exclude 3 subjects due to missing/corrupted files (SD131, SC140, SD193)
                            if (len(func_roi_filename) > 0) & (sub != 'sub-SD131') & (sub != 'sub-SC140') & (sub != 'sub-SD193'):
                                if approach == 'generalization':
                                    vg_or_replay_list = vg_or_replay.split('-')

                                    if (condition.find('-') != -1):
                                        condition_list = condition.split('-')
                                        condition1 = condition_list[0]
                                        condition2 = condition_list[1]
                                    else:
                                        condition1 = condition
                                        condition2 = condition

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

                                    func_roi_filename = [x for _, x in sorted(zip(roi_labels, func_roi_filename))]
                                    theory_labels = [x for _, x in sorted(zip(roi_labels, theory_labels))]
                                    roi_labels.sort()

                                    print("Running " + sub)
                                    beta_maps1, trial_labels1, run_labels1 = prepare_nibetaseries_data(vg_or_replay_list[0], nibetaseries_filename, tsv_filename1, condition1, stimulus_categories, number_of_runs[0])
                                    beta_maps2, trial_labels2, run_labels2 = prepare_nibetaseries_data(vg_or_replay_list[1], nibetaseries_filename, tsv_filename2, condition2, stimulus_categories, number_of_runs[1])

                                    roi_accuracy_scores = roi_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, func_roi_filename)

                                elif approach == 'within_condition':

                                    # Loop over event tsv files of the subject
                                    for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                                        if (filename.find('.tsv') != -1) and (filename.find(vg_or_replay) != -1) and (filename.find('_v2') == -1):
                                            #print('found a tsv file\n')
                                            tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))

                                    tsv_filename.sort()
                                    func_roi_filename = [x for _, x in sorted(zip(roi_labels, func_roi_filename))]

                                    theory_labels = [x for _, x in sorted(zip(roi_labels, theory_labels))]
                                    roi_labels.sort()


                                    print("Running " + sub)
                                    beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(vg_or_replay, nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs)
                                    roi_accuracy_scores = roi_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, func_roi_filename)

                                # Insert column names in the accuracy data frame
                                if i == 0:
                                    df.insert(0, 'ROI', roi_labels)
                                    df.insert(1, 'Theory', theory_labels)

                                # Increment row index and insert accuracy scores for the current subject
                                i = i + 1
                                df.insert(i + 1, sub, roi_accuracy_scores)
                                print('finished sub ' + str(i))
                                included_subs.append(sub_code)
                            else:
                                print('MISSING data: ' + sub)
                                missing_data.append(sub_code)
                        else:
                            print('MISSING ROIs: ' + sub)
                            missing_rois.append(sub_code)
        else:
            print('excluding subject: ' + sub_code)
            excluded_subs.append(sub_code)

    # Create a directory to save the output
    os.makedirs(output_dir, exist_ok=True)
    # Save accuracy scores in a csv file
    df.to_csv(os.path.join(output_dir, 'roi_' + vg_or_replay + '_category_' + condition + '_' + approach + '_' + str(n_voxels) + 'vox_' + roi_condition + '.csv'))
    #df.to_csv(os.path.join(output_dir, 'roi_' + vg_or_replay + '_category_' + condition + '_' + str(n_voxels) + 'vox_' + roi_condition + '.csv'))
    print('total subjects: ' + str(sum(analysis_check_list)))
    print('total included subjects: ' + str(len(included_subs)))
    print(included_subs)
    print('total missing subjects: ' + str(len(missing_data)))
    print(missing_data)
    print('total missing ROIs: ' + str(len(missing_rois)))
    print(missing_rois)
    if use_evc_rois:
        print('total missing EVC: ' + str(len(missing_evc)))
        print(missing_evc)
    toc = time.time()
    print('total runtime: ' + str(toc - tic))

if __name__ == "__main__":
    run_analysis()
