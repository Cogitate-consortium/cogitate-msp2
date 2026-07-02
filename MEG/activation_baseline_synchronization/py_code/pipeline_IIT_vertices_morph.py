'''
Individual-level (IIT): morph a subject's IIT-selected VertROI vertices to the
standard brain (fsaverage), so the group can show vertex locations in one common
space (HPC can't render 3D brains, so the actual plotting is done later/locally).

Per subject + config:
  - load the pre-saved example stc + inverse (from presaved_folder_name),
  - build the subject -> fsaverage morph matrix (identity for subjects w/o T1),
  - for each VertROI label, mark its vertices and morph them, keeping both the
    continuous weights (morphed_v) and a binary presence map (morphed_v_binary).
Saves morphedv_roidict (per roi/label) as VertROI_morphedv_dict.pkl under
path_allana/<script_name>/. Skips if it already exists.

Next: pipeline_IIT_vertices_morph_group.py aggregates these for local 3D plotting.

Config folder: /config_files/pipeline_IIT_vertices_morph/
    dAT.json
  each sets presaved_folder_name, exp_id, task_id.
'''
#%% import
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime
import numpy as np

# for save data
import pickle

from EXP1_help_functions import (BidsPath)

from _help_functions import (sub_no_T1,
                             path_oct6_fs_src,
                             general_param,
                             path_allana,
                                 get_label_vertidx,
                                 read_labels_exp2,# for load VertROI labels
                                    read_configfile,
                                 )

import mne_bids
import mne
#%% setting test
flag_test = False
if   flag_test:
    print('---------------------------- test flag on',datetime.now())
    # Create an instance of Args and set the subject attribute
    class Args:
        def __init__(self):
            self.subject = None
    args = Args()
    base = Path(__file__).resolve().parent.parent
    args.subject = 'SA108'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_IIT_vertices_morph" / "dAT.json"

else:
    parser = argparse.ArgumentParser(
        description="Implements analysis of source erf for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                    help="Name of the subject")
    parser.add_argument('--configfile', type=str, default=None,
                        help="configfile file for analysis parameters (file name + path)")
    args = parser.parse_args()



oct6_fs_src = mne.read_source_spaces(path_oct6_fs_src)
oct6_expected_nvert = len(oct6_fs_src[0]['vertno'])
#%% load setting 
subject = args.subject
configfile  = args.configfile
config_filename_with_ext = os.path.basename(configfile)
config_filename, _ = os.path.splitext(config_filename_with_ext)
print('--------- start ',config_filename)

script_name = os.path.basename(__file__).replace('.py','')

# Read the configfile file:
param = read_configfile(configfile)


deriv_root_presaved_sourceinfo   = os.path.join(path_allana,param['presaved_folder_name'])
exp_id =param['exp_id']
task_id = param['task_id']
# !!!!! specific for VertROI
roi_params = general_param['VertROI_params_dict']

# Experiment1 defined fs path
bids_paths = BidsPath(subject_id=subject, visit_id=exp_id)
subjects_dir = bids_paths.fs_deriv_root

#%% setting path    
# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana,
                              script_name,
                          "test")
else:
    deriv_root = os.path.join(path_allana, script_name)

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= exp_id,  
        datatype='meg',  
        task= task_id,
        suffix="VertROI_morphedv_dict",
        extension='.pkl',
        check=False)
os.makedirs(os.path.dirname( bids_path.fpath), exist_ok=True)      

pklfile_name = bids_path.fpath
#%% get the results
if  os.path.exists(pklfile_name) :
    print('------------','exist subject' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )
    
    try:
        with open(pklfile_name, 'rb') as pickle_file_load:
            loaded_data = pickle.load(pickle_file_load)
        print(loaded_data.keys())
        print('------------','Successfully loaded' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )    
    except Exception :
        # Print the error for debugging purposes (optional)
        print('------------','Error loading' ,pklfile_name,
            '\n',config_filename,
            '\n',datetime.now(),
                )
        pass
else:
    #% load presaved stc and inv
    # filename of stc example
    bids_path_stc = mne_bids.BIDSPath(
            root=deriv_root_presaved_sourceinfo, 
            subject= subject, 
            session= exp_id,  
            datatype='meg',  
            task=task_id,
            suffix='example_stc',
            check=False)
    stc_filename = bids_path_stc.fpath 
    stc_exp = mne.read_source_estimate(stc_filename)

    # filename of inv
    bids_path_inv = mne_bids.BIDSPath(
            root=deriv_root_presaved_sourceinfo, 
            subject= subject, 
            session= exp_id,  
            datatype='meg',  
            task=task_id,
            suffix='inv',
            extension='.fif',
            check=False)
    inv_filename = bids_path_inv.fpath
    inv = mne.minimum_norm.read_inverse_operator(inv_filename)
    src = inv['src']



    #% Compute morph matrix or identity matrix
    # construct morphed stc with only one time point
    stc      = stc_exp.copy()
    stc.data = stc.data[:, :1]

    if subject not in sub_no_T1:
        morph = mne.compute_source_morph(
            stc, 
            subject_from=f"sub-{bids_path_stc.subject}", 
            src_to=oct6_fs_src, 
            subject_to='fsaverage', 
            subjects_dir=subjects_dir
        )
        morph_matrix = morph.morph_mat
        # morph_matrix[i, j]:
        # Vertex i of target (fsaverage) receives weight from vertex j of individual brain
        # 
        # Morphing logic:
        # - One vertex in individual brain is morphed to multiple vertices in fsaverage
        # - The "signal" from one individual vertex is distributed across nearby fsaverage vertices
        #
        # Matrix structure:
        # - Rows: target vertices (fsaverage)
        # - Columns: source vertices (individual brain)
        # - Each column sum ≈ 1 (signal conservation from one source vertex)
        # - Each row sum ≠ 1 (one target vertex receives from multiple sources)
        #
        # Example:
        # morph_matrix[:, j].sum() ≈ 1  # vertex j's signal is conserved
        # morph_matrix[i, :].sum() ≠ 1  # vertex i receives variable amounts
    else:
        morph_matrix = np.eye(stc.data.shape[0])
    print(f'get morph {subject}...')


    #% get the each ROI, each label origianl vert index in individual source space
    # ------------------ get label info ------------------ 
    label_dict = {}
    roilist_all = {}
    for one_roi in roi_params.keys():
        one_roi
        lb_use = read_labels_exp2(
            bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   = exp_id), 
            parc       = roi_params[one_roi]['parc'], 
            labels_list= roi_params[one_roi]['labels_list'],
            rois_list  = roi_params[one_roi]['rois_list'], 
            combine_labels_list=True, 
            merge_hemi=True
            )  
        if lb_use:
            print('read label for',one_roi, list(lb_use.keys()))
            ## get vertno index
            label_vertidx_dict=\
                        get_label_vertidx(lb_use =lb_use,
                                            src    = src,
                                            flag_return_dict =1)
            label_dict[one_roi] = label_vertidx_dict
            roi_params[one_roi]['sublabel_list'] = list( lb_use.keys())
            roilist_all.update({one_roi:lb_use})
        else:
            print('no label for',one_roi)
            label_dict[one_roi] = {}
            roi_params[one_roi]['sublabel_list'] =[]
    print(roi_params[one_roi]['sublabel_list'])
    # all roi should have label
    assert len(roi_params[one_roi]) == len(roi_params[one_roi].keys()),'some roi have no label'
        
    #% morphed vertices in standard brain space
    morphedv_roidict = {}
    for roii,roi in enumerate(roi_params.keys()):
        morphedv_roidict [roi] ={}
        if roi_params[roi]['sublabel_list'] ==[]:
            continue
        for label_name in roi_params[roi]['sublabel_list']:
            # get the roi 
            label_vidx_lr = label_dict[roi]['vidx_dic_allowoverlap'][label_name]
            label_vidx_l  = label_dict[roi]['vidx_l_dic']           [label_name]
            label_vidx_r  = label_dict[roi]['vidx_r_dic']           [label_name]
            

            vert_2d = np.zeros_like(stc.data) 
            vert_2d[ label_vidx_lr, 0] = 1 
            
            morphed_v = morph_matrix @ vert_2d  # continuous values  [0, 1]
            # morphed_v[i] = sum of weights from all N marked vertices to fsaverage vertex i
            # if N is large, morphed_v values can be much greater than 1

            # Binary version:
            # No matter how many source vertices contribute to a target vertex,
            # if at least one source vertex contributes, set target vertex to 1
            # reflects presence/absence of signal 
            # Then in group analysis, can further compute the proportion of subjects with signal at each vertex
            # When to use: If only care about presence/absence of signal: binarize the morphed_v
            morphed_v_binary = (morphed_v > 0).astype(int) # any vertex >0 is set to 1
            morphedv_roidict[roi][label_name] ={
                'morphed_v':morphed_v,
                "morphed_v_binary":morphed_v_binary
            }     
    save_data = {
                'morphedv_roidict': morphedv_roidict,
                'roi_params':roi_params,
                'param_config': param,
            }    

    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)

    print('------','save' ,
          '\n',pklfile_name,
        '\n',subject,
        '\n',config_filename,
        '\n',datetime.now(),
            )


#%%