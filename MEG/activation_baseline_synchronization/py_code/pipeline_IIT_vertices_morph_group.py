'''
Group-level (IIT): aggregate the per-subject morphed VertROI vertices
(from pipeline_IIT_vertices_morph.py) in fsaverage space, so the vertex
locations can be shown in one common brain (HPC can't render 3D, so the actual
plotting is done later/locally).

Loads each subject's VertROI_morphedv_dict.pkl and, per VertROI, averages the
morphed maps across subjects (the continuous morphed_v, and the binary presence
map -> fraction of subjects with a vertex there).
Saves morphedv_subdict + group_morphed_v_roidict (+ sub lists) as
"<task_id> morph vert <sub_list_name> <N>sub.pkl" under
path_allana/pipeline_IIT_vertices_morph/analysis_group_results/.

Uses the same config as the individual morph step
(/config_files/pipeline_IIT_vertices_morph/dAT.json); no argparse, runs directly.
'''
#%% import
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# for show time
from datetime import datetime
import numpy as np

# for save data
import pickle

from _help_functions import (general_param,
                             path_allana,
                             read_configfile,
                             get_sublist,
                             )

import mne_bids
#%% setting test

#%% load setting
base = Path(__file__).resolve().parent.parent
configfile = base / "config_files" / "pipeline_IIT_vertices_morph" / "dAT.json"
config_filename_with_ext = os.path.basename(configfile)
config_filename, _ = os.path.splitext(config_filename_with_ext)
print('--------- start ',config_filename)

sub_list = get_sublist(configfile)
# Read the configfile file:
param = read_configfile(configfile)

exp_id =param['exp_id']
task_id = param['task_id']
sub_list_name = param['sub_list_name']
# !!!!! specific for VertROI
roi_params = general_param['VertROI_params_dict']


#%% setting path    
# create folder with analyse name
path_group = os.path.join(path_allana, 
                               "pipeline_IIT_vertices_morph",
                               "analysis_group_results")
os.makedirs(path_group, exist_ok=True)

file_name = f"{task_id} morph vert {sub_list_name} {len(sub_list)}sub"
savefile_name  = os.path.join(path_group, f"{file_name}.pkl").replace(' ','_')

print('------','start to process and save' ,
        '\n',savefile_name,
    '\n',datetime.now(),
        )
#% load each subject morphed vertices index
morphedv_subdict = {}
for subject in sub_list:
    deriv_root = os.path.join(path_allana, 
                        "pipeline_IIT_vertices_morph")
    bids_path = mne_bids.BIDSPath(
            root=deriv_root, 
            subject= subject, 
            session= exp_id,  
            datatype='meg',  
            task= task_id,
            suffix="VertROI_morphedv_dict",
            extension='.pkl',
            check=False)
    pklfile_name = bids_path.fpath
    with open(pklfile_name, 'rb') as pickle_file_load:
        loaded_data = pickle.load(pickle_file_load)
    morphedv_subdict[subject] = loaded_data['morphedv_roidict']

group_morphed_v_roidict = {}
nsub_roidict = {}
for roi in roi_params.keys():

    roi_sublist =  [ sub for sub in sub_list
    if len(morphedv_subdict[sub][roi]) >0]

    nsub_roidict[roi] = roi_sublist
    morphedv_subv = np.squeeze(np.array(
        [ morphedv_subdict[sub][roi][
        [y for y in morphedv_subdict[sub][roi].keys() if ('all' in y) | ('vert' in y)| ('fusifor' in y)][0]
                    ]['morphed_v'] 
                    for sub in sub_list
                    if len(morphedv_subdict[sub][roi]) >0]))
    morphed_v_binary_subv = np.squeeze(np.array(
        [ morphedv_subdict[sub][roi][
        [y for y in morphedv_subdict[sub][roi].keys() if ('all' in y) |('vert' in y)| ('fusifor' in y)][0]
                    ]['morphed_v_binary'] 
                    for sub in sub_list
                    if len(morphedv_subdict[sub][roi]) >0]))
    
    morphedv_avg = np.mean(morphedv_subv,axis=0)
    morphedv_binary_avg = np.mean(morphed_v_binary_subv,axis=0)
    group_morphed_v_roidict[roi] = {
        'morphedv_avg':morphedv_avg,
        'morphedv_binary_avg':morphedv_binary_avg
    }
# Save results
save_data = {
    'morphedv_subdict': morphedv_subdict,
    'group_morphed_v_roidict': group_morphed_v_roidict,
    'param_config': param,
    'roi_params': roi_params,
    'sub_list': sub_list,
    'nsub_roidict': nsub_roidict,
    'VertROI_params_dict': general_param['VertROI_params_dict'],
    'roi_params_dict':general_param['roi_params_dict'],

}

with open(savefile_name, 'wb') as pickle_file:
    pickle.dump(save_data, pickle_file)  
print('------','save' ,
    '\n',savefile_name,
    '\n',datetime.now(),
        )

    
# %%
