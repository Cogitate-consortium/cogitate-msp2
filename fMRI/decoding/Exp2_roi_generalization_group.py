"""
Author: Zvi Roth
Date created: 10-14-2024
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import mne.stats

condition = 'seen_go-seen'
#condition = 'seen-seen_go'
#condition = 'seen-seen'
vg_or_replay = 'Replay-VG'  # generalization from Replay to VG
#vg_or_replay = 'VG-Replay'  # generalization from VG to Replay

#vg_or_replay = 'Replay'  # generalization from VG to Replay
#condition = 'seen_nogo'
#condition = 'seen_go'
#condition = 'seen'
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
chance_level = 0.5


# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
    approach = 'within_condition'
elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay') | (vg_or_replay == 'VG-VG') | (vg_or_replay == 'Replay-Replay'):
    approach = 'generalization'
else:
    raise Exception("unrecognized vg_or_replay value, should be 'VG', 'Replay', 'Replay-VG', or 'VG-Replay'. ")

csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/','roi_decoding')

csv_file = 'roi_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach + '_' + str(n_voxels) + 'vox_' + roi_condition + '.csv'

csv_filename = os.path.join(csv_dir, csv_file)

# Load the accuracy values from the csv table:
data_df = pd.read_csv(csv_filename)

# Create a new dataframe to save the output values
df = pd.DataFrame(list())
df = data_df.loc[:, ['ROI','Theory']]
df.insert(2, 'Average Accuracy', '')
df.insert(3, 'Corrected p-value', '')
df.insert(4, 'Significance', '')
accuracy_values = data_df.iloc[:,3:].to_numpy()
df.loc[:,'Average Accuracy'] = accuracy_values.mean(axis=1)

# Run one sample  permutation test
ROI_count= len(accuracy_values[:,1])
p_values= np.zeros([ROI_count])
for i in range(ROI_count):
     statistic, p_value, H0 = mne.stats.permutation_t_test(np.reshape(accuracy_values[i,:]-chance_level,(len(accuracy_values[i, :]),1)), n_permutations=5000, tail=1)
     p_values[i]=p_value

# Correct for multiple comparisons
reject, p_values_corrected= mne.stats.fdr_correction(p_values, alpha=0.05, method='indep') #0.05
df.loc[:, 'Significance'] = np.multiply(p_values_corrected<0.05, 1) #0.05
df.loc[:, 'Corrected p-value'] = p_values_corrected

# Save average accuracy and significance for all ROIs in a csv file
df.to_csv(os.path.join(csv_dir, 'roi_stats_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach + '_' + str(n_voxels) + 'vox_' + roi_condition + '.csv'))
