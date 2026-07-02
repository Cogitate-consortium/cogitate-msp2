"""
Plots ROI accuracy results on a brain surface

Author: Zvi Roth
Date created: 08-09-2024
"""
# mne.utils.set_config("SUBJECTS_DIR","/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/freesurfer", set_env=True)
# os.environ["SUBJECTS_DIR"] = "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/freesurfer"
import numpy as np
import matplotlib.pyplot as plt
from plotters import plot_time_series, plot_matrix, plot_rasters, plot_brain
import config
import os
import pandas as pd
# get the parameters dictionary
param = config.param

# =================================================================================
os.environ["SUBJECTS_DIR"] = "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/freesurfer"
#decoding_problem = 'location'
#decoding_problem = 'category'
#condition = 'seen-seen_nogo'  # 'all' or 'seen' or 'unseen', or 'seen-unseen'
#condition = 'seen_go-seen'  # 'all' or 'seen' or 'unseen', or 'seen-unseen'
#vg_or_replay = 'Replay-VG'  # generalization from Replay to VG
#vg_or_replay = 'VG-Replay'  # generalization from VG to Replay
#condition = 'seen_go'  # 'all' or 'seen' or 'unseen', or 'seen-unseen'
#vg_or_replay = 'Replay'
roi_condition = 'rel'

# Number of voxels per ROI (ROI size)
n_voxels = 300

# Plotting brain surface:
GNW_roi_list = ['G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant', 'G_and_S_cingul-Mid-Post', 'G_front_inf-Opercular', 'G_front_inf-Orbital', 'G_front_inf-Triangul', 'G_front_middle', 'Lat_Fis-ant-Horizont', 'Lat_Fis-ant-Vertical', 'S_front_inf', 'S_front_middle', 'S_front_sup']
# IIT Basic ROI list
IIT_roi_list_1 = ['G_temporal_inf', 'Pole_temporal', 'G_cuneus', 'G_occipital_sup', 'G_oc-temp_med-Lingual', 'Pole_occipital', 'S_calcarine', 'G_and_S_occipital_inf', 'G_occipital_middle', 'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Parahip', 'S_intrapariet_and_P_trans', 'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_temporal_sup']
roi_list = GNW_roi_list + IIT_roi_list_1

vg_or_replays = ('VG-Replay', 'Replay-VG', 'Replay')
vg_or_replays = ('Replay', 'VG-Replay', 'Replay-VG')

for vg_or_replay in vg_or_replays:
    # Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
    if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
        approach = 'within_condition'
    elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay'):
        approach = 'generalization'
    chance_level = 0.5
    if vg_or_replay == 'Replay':
        #conditions = ('seen', 'seen_go','seen_nogo') #replay location nogo is a problem
        conditions = ('seen', 'seen_go')
    elif vg_or_replay == 'VG-Replay':
        conditions = ('seen-seen', 'seen-seen_go', 'seen-seen_nogo')
    elif vg_or_replay == 'Replay-VG':
        conditions = ('seen-seen', 'seen_go-seen', 'seen_nogo-seen')
    decoding_problems = ('location', 'category')

    # Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
    if (vg_or_replay == 'Replay') | (vg_or_replay == 'VG'):
        approach = 'within_condition'
        approach_str = 'within_condition_'
    elif (vg_or_replay == 'Replay-VG') | (vg_or_replay == 'VG-Replay'):
        approach = 'generalization'
        approach_str = 'generalization_'
    else:
        raise Exception("unrecognized vg_or_replay value, should be 'VG', 'Replay', 'Replay-VG', or 'VG-Replay'. ")


    cmaps = 'Oranges'

    for decoding_problem in decoding_problems:
        for condition in conditions:

            csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/ses-V2/','roi_decoding')
            csv_filename= 'roi_stats_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach_str + str(n_voxels) + 'vox_' + roi_condition + '.csv'
            output_filename = 'roi_plots_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach_str + str(n_voxels) + 'vox_' + roi_condition + '.eps'

            csv_file = os.path.join(csv_dir, csv_filename)
            save_file= os.path.join(csv_dir, output_filename)
            data_df = pd.read_csv(csv_file)
            rois = (data_df['ROI']).tolist()
            accuracies= (data_df['Average Accuracy']).array
            Significance = (data_df['Significance']).array

            rois_dict = {}
            k=0
            for roi in rois:
                if roi in roi_list:
                    #rois_dict[roi] =accuracies[k]
                    if Significance[k]:
                        rois_dict[roi] = accuracies[k]
                k = k+1

            #df.to_csv(os.path.join(csv_dir, 'roi_stats_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach + '_' + str(n_voxels) + 'vox_' + roi_condition + '.csv'))
            #df.to_csv(os.path.join(csv_dir, 'roi_stats_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + str(n_voxels) + 'vox_' + roi_condition + '_' + str(num_iter) + 'subsample.csv'))

            max_value = max(rois_dict.values())
            chance_level = 0.5
            cmap_start = 0  # lowest color in the cmap which will correspond to vmin
            cmap_end = 1  # highest color in the cmap which will correspond to vmin
            vmin = chance_level  # min value of the cmap
            vmax = max_value  # max value of the cmap

            img_filename = vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach_str +  roi_condition
            plot_brain( roi_map=rois_dict, subject='fsaverage', surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s',
                                     views=['lateral',(90, -30, 0)],cmap_start=cmap_start, cmap_end=cmap_end,
                                     cmap=cmaps, colorbar=True, colorbar_title=None, vmin=vmin, vmax=vmax, outline_overlay=True, overlay_method='overlay',
                                     brain_cmap='Greys', brain_alpha=1, save_file=os.path.join(csv_dir, img_filename + '_lh_outline_30deg.png'))
            plot_brain( roi_map=rois_dict, subject='fsaverage', surface='inflated', hemi='rh', sulc_map='curv', parc='aparc.a2009s',
                                     views=['lateral',(-90, 30, 0)],cmap_start=cmap_start, cmap_end=cmap_end,
                                     cmap=cmaps, colorbar=True, colorbar_title=None, vmin=vmin, vmax=vmax, outline_overlay=True, overlay_method='overlay',
                                     brain_cmap='Greys', brain_alpha=1, save_file=os.path.join(csv_dir, img_filename + '_rh_outline_30deg.png'))
        plt.close('all')
#roi_map_edge_color = [0, 0, 0] to add borders around rois