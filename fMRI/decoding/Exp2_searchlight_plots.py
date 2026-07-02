"""
Plots Searclight results on a brain surface

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
import time
# import cog_plot as cp
import nilearn.image

tic = time.time()

# get the parameters dictionary
param = config.param

# =================================================================================
os.environ["SUBJECTS_DIR"] = "/wishData/cogitate-data/fMRI/bids/derivatives/freesurfer"

base_dir = os.path.join('/', 'wishData', 'cogitate-data', 'fMRI', 'bids', 'derivatives', 'decoding', 'nibetaseries', 'ses-V2')
# figures_dir = os.path.join(base_dir, 'searchlight_figs')
figures_dir = "/home/zviro/wish_outputs/searchlight_figs"
os.makedirs(figures_dir, exist_ok=True)


color_VG = '#8B2BE2'
color_Replay = '#D55E00'
color_IIT = '#0173B2'
color_GNW = '#029E73'
color_VG_seen = '#611E9E'
color_VG_unseen = '#AE6BEB'
color_Replay_seen = '#AA4B00'
color_Replay_unseen = '#E28E4D'
# color = color_Replay #default, orange. #
# color= cp.DEFAULT_COLORS['task relevant']
color = color_VG


def print_brain(img_filename, output_filename, chance_level, color, max_value=None):

    # color  = 'Oranges'

    overlay_list = list()
    overlay_list.append(img_filename)
    # img = nilearn.image.load_img(img_filename)

    # check whether nifti file exists
    if not os.path.exists(img_filename):
        print(f"[WARN] File not found, skipping: {img_filename}")
        return

    img_data = nilearn.image.get_data(img_filename)

    # 1) Guard against completely non-finite maps
    if not np.isfinite(img_data).any():
        print(f"[WARN] {img_filename} has only NaN/inf values – skipping.")
        return

    # 2) Guard against constant maps (all values identical)
    finite_data = img_data[np.isfinite(img_data)]
    if finite_data.size == 0:
        print(f"[WARN] {img_filename} has no finite values after masking – skipping.")
        return

    data_min = float(finite_data.min())
    data_max = float(finite_data.max())
    if data_min == data_max:
        print(f"[WARN] {img_filename} is constant ({data_min}) – skipping color plot.")
        return

    if max_value is None:
        max_value = data_max

    cmap_start = 0
    cmap_end = 1 #max_value
    vmax = max_value #max_value #if not set, it will be different for RH and LH
    vmin = chance_level #chance_level  # None #0.45 #0.5

    # to display decimals on colorbar. ZR 7/20/2025
    #vmax = int(100 * vmax)
    #vmin = int(100 * vmin)
    #cmap_end = int(100 * cmap_end)

    #vmax = round(100*vmax)
    #vmin = round(100*vmin)

    # print left hemisphere
    plot_brain(surface='inflated', cmap=color, cmap_start=cmap_start, cmap_end=cmap_end,
               overlays=overlay_list, hemi='lh', views=['lateral', (90, -30, 0)],
               overlay_threshold=0, vmin=vmin, vmax=vmax, outline_overlay=True,
               colorbar_title=None,
               save_file=os.path.join(figures_dir, output_filename + '_lh_outline_30deg.png'))
    # print right hemisphere
    plot_brain(surface='inflated', cmap=color, cmap_start=cmap_start, cmap_end=cmap_end,
               overlays=overlay_list, hemi='rh', views=['lateral', (-90, 30, 0)],
               overlay_threshold=0, vmin=vmin, vmax=vmax, outline_overlay=True,
               colorbar_title=None,
               save_file=os.path.join(figures_dir, output_filename + '_rh_outline_30deg.png'))
    ''' DEFAULT:
    plot_brain(subject='fsaverage', subjects_dir=None, surface='inflated', hemi='lh', sulc_map='curv',
               parc='aparc.a2009s', roi_map=None, roi_map_edge_color=None, roi_map_transparency=1.,
               views=['lateral', 'medial'],
               cmap='Oranges', colorbar=True, colorbar_title='ACC', colorbar_title_position='top', cmap_start=0.1,
               cmap_end=1.,
               vmin=None, vmax=None, overlay_method='overlay',
               overlays=None, overlay_threshold=None, outline_overlay=False,
               electrode_activations=None, vertex_distance=10., vertex_scaling='linear', vertex_scale=(1.0, 0.),
               vertex_method='max',
               scanner2tkr=(0, 0, 0), brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6),
               save_file=None, dpi=300)
               '''
# ===================
#       RSA
# ===================
#max_value = 0.41 # this is the max for category, for location it is 40. Makes uniform colorscale.
max_value = 0.4 #so uniform color scale for all plots.

chance_level = -max_value # 0.0

decoding_problem = 'category'
decoding_problems = ('category', 'location')
flip_strings = ('flip','')
replay_strings = ('Replay_','')
#color = cp.DEFAULT_COLORS['Irrelevant']
# color  = 'Oranges'
color = 'RdBu'
for decoding_problem in decoding_problems:
    if decoding_problem == 'category':
        suffixes = ['', 'Face_', 'Object_']
    else:
        suffixes = ['', 'Left_', 'Right_']
    suffix = suffixes[0]

    for replay_str in replay_strings:
        output_dir = os.path.join(base_dir, 'searchlight_RSA_' + replay_str + decoding_problem)
        for flip_str in flip_strings:
            output_filename = 'searchlight_RSA_' + replay_str + decoding_problem + '_' + suffix + flip_str
            img_filename = os.path.join(output_dir, output_filename + '.nii')
            if (replay_str=='Replay_') & (flip_str=='flip'):
                print_brain(img_filename, output_filename, chance_level, color, max_value)
            else:
                print_brain(img_filename, output_filename, chance_level, color)

toc = time.time()
print('RSA total runtime: ' + str(toc - tic))

'''

# =================================
#       Decoding - Replay Go vs. No-Go
# =================================
searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_goNogo')
searchlight_filename = 'searchlight_goNogo_group'
img_filename = os.path.join(searchlight_dir, searchlight_filename + '.nii')
#color = cp.DEFAULT_COLORS['task relevant']
color  = 'Oranges'
output_filename = searchlight_filename
chance_level = 0.5
print_brain(img_filename, output_filename, chance_level, color)


# =================================
#       Decoding - Seen vs. Unseen
# =================================
searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_seenUnseen')
searchlight_filename = 'searchlight_seenUnseen_group_VG'
img_filename = os.path.join(searchlight_dir, searchlight_filename + '.nii')
#color = cp.DEFAULT_COLORS['task relevant']
color  = 'Purples'
max_value = 0.55
output_filename = searchlight_filename
chance_level = 0.5
print_brain(img_filename, output_filename, chance_level, color, max_value)



# =================================
#       Decoding - NOT SUBSAMPLE
# =================================

color  = 'Oranges'
max_value = 0.55
#vg_or_replays = ('VG-Replay', 'Replay-VG', 'Replay')
vg_or_replays = ('Replay','VG-Replay', 'Replay-VG')
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

    for decoding_problem in decoding_problems:
        for condition in conditions:
            searchlight_filename = 'searchlight_group_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach

            if decoding_problem == 'category':
                searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_category')
            else:
                searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_location/')
            img_filename = os.path.join(searchlight_dir, searchlight_filename + '.nii')
            output_filename = searchlight_filename

            print_brain(img_filename, output_filename, chance_level, color, max_value)




# =================================
#       Decoding - subsample
# =================================

#decoding_problem = 'location'
vg_or_replay = 'VG'

#condition = 'unseen'

num_iter = 10
decoding_problems = ('location','category')

#color = cp.DEFAULT_COLORS['task relevant']
color  = 'Purples'
max_value = 0.55
conditions = ('seen-unseen','seen', 'unseen')
#conditions = ('seen', 'unseen')
for condition in conditions:
    for decoding_problem in decoding_problems:
        if condition == 'seen-unseen':
            approach = ''
            chance_level = 0.0
            searchlight_filename = 'searchlight_group_'  + decoding_problem + '_' + condition + '_' + str(num_iter) + 'subsample'
        else:
            approach = 'within_condition'
            chance_level = 0.5
            searchlight_filename = 'searchlight_group_' + vg_or_replay + '_' + decoding_problem + '_' + condition + '_' + approach + str(num_iter) + 'subsample'

        if decoding_problem == 'category':
            searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_categ_subsample')
        else:
            searchlight_dir = os.path.join(base_dir, 'searchlight_decoding_loc_subsample/')
        img_filename = os.path.join(searchlight_dir, searchlight_filename + '.nii')
        output_filename = searchlight_filename

        if (condition == 'seen-unseen'):
            print_brain(img_filename, output_filename, chance_level, color)
        else:
            print_brain(img_filename, output_filename, chance_level, color, max_value) #max_value=0.55

'''



toc = time.time()
print('total runtime: ' + str(toc - tic))
