"""
Author: Zvi Roth
Date created: 12-08-2024
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import mne.stats

num_iter = 10
vg_or_replay = 'VG'
condition = 'seen-unseen'  # 'all' or 'seen' or 'unseen'
decoding_problem = 'location'  # 'location' or 'category'
#decoding_problem = 'category'

# Select functional ROIs that were created using the data of the selected decoding condition
roi_condition = 'rel' # 'irrel'

# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
# For orientation decoding, do not change 'stimulus_categories' here and change line 43 instead.
stimulus_categories = ['face', 'object']

# Number of voxels per ROI (ROI size)
n_voxels = 300
# Chance level (50% for category decoding (binary) and 33.33% for orientation decoding (3-class) )
chance_level = 0.0


# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
    approach = 'within_condition'
elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay'):
    approach = 'generalization'
else:
    raise Exception("unrecognized vg_or_replay value, should be 'VG', 'Replay', 'Replay-VG', or 'VG-Replay'. ")

csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/','roi_decoding')
condition = 'seen'
csv_file = 'roi_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + str(n_voxels) + 'vox_' + roi_condition + '_' + str(num_iter) + 'subsample.csv'
csv_filename_seen = os.path.join(csv_dir, csv_file)
condition = 'unseen'
csv_file = 'roi_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + str(n_voxels) + 'vox_' + roi_condition + '_' + str(num_iter) + 'subsample.csv'
csv_filename_unseen = os.path.join(csv_dir, csv_file)

# Load the accuracy values from the csv table:
data_df_seen = pd.read_csv(csv_filename_seen)
data_df_unseen = pd.read_csv(csv_filename_unseen)
accuracy_values_seen = data_df_seen.iloc[:,3:].to_numpy()
accuracy_values_unseen = data_df_unseen.iloc[:,3:].to_numpy()
accuracy_values_diff = accuracy_values_seen - accuracy_values_unseen

# Create a new dataframe to save the output values
df = pd.DataFrame(list())
df = data_df_seen.loc[:, ['ROI','Theory']]
df.insert(2, 'Average Accuracy', '')
df.insert(3, 'Corrected p-value', '')
df.insert(4, 'Significance', '')

df.loc[:,'Average Accuracy'] = accuracy_values_diff.mean(axis=1)

# Run one sample  permutation test
ROI_count= len(accuracy_values_diff[:,1])
p_values= np.zeros([ROI_count])
for i in range(ROI_count):
     statistic, p_value, H0 = mne.stats.permutation_t_test(np.reshape(accuracy_values_diff[i,:]-chance_level,(len(accuracy_values_diff[i, :]),1)), n_permutations=5000, tail=1)
     p_values[i]=p_value

# Correct for multiple comparisons
reject, p_values_corrected= mne.stats.fdr_correction(p_values, alpha=0.05, method='indep') #0.05
df.loc[:, 'Significance'] = np.multiply(p_values_corrected<0.05, 1) #0.05
df.loc[:, 'Corrected p-value'] = p_values_corrected

# Save average accuracy and significance for all ROIs in a csv file
condition = 'seen-unseen'
df.to_csv(os.path.join(csv_dir, 'roi_stats_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + str(n_voxels) + 'vox_' + roi_condition + '_' + str(num_iter) + 'subsample.csv'))
