'''
Individual-level: source-space time-frequency (TFR) power for one ROI's labels,
for a two-condition comparison (cond1 vs cond2, e.g. dAT-seen vs dAT-unseen).

Runs ONE ROI per call (param['roi']). Per subject + config:
  - build source estimates for all epochs (get_stcs_allsrc),
  - for each label in the ROI, restrict to the label vertices and split trials
    into cond1 / cond2 (get_trl_indices on compare_trltypes_select_column),
  - compute single-trial multitaper power with mne.time_frequency.tfr_array_multitaper
    (output='power') using the freq_range_name settings in tfr_map, then average
    over trials -> per-condition (vertices, freqs, times) power.
Saves vft_labeldict (+ metadata, df_nepoch, tfr_map, label_dict, roi_params) as
<epoch_data_name>_<compare_trltypes>_<tfr_method>_<roi>_<freq_range_name>.pkl
under path_allana/<script_name>/<epoch_info>/. Skips if the file already exists.

Two frequency settings are run separately (Ferrante et al. 2022; see config below):
  1_30   -> 1-30 Hz, step 1 Hz, 1 taper,  n_cycles = freq/2
  30_100 -> 30-100 Hz, step 2 Hz, 3 tapers, n_cycles = freq/4

Note: running per label (not per ROI then extracting labels) is slower, but since
labels are saved per label rather than per ROI it keeps vertex extraction clearer.

Consumed by pipeline_tfr_source_group_band.py / pipeline_tfr_source_group_ft.py
(and their _plot.py).

Config folder: /config_files/pipeline_tfr_source/
    dAT_probe_face_noft_nobc_deci5_nocut_PFC_1_30.json
    dAT_probe_face_noft_nobc_deci5_nocut_PFC_30_100.json
    dAT_probe_obje_noft_nobc_deci5_nocut_PFC_1_30.json
    dAT_probe_obje_noft_nobc_deci5_nocut_PFC_30_100.json
    dAT_probe_stim_noft_nobc_deci5_nocut_PFC_1_30.json
    dAT_probe_stim_noft_nobc_deci5_nocut_PFC_30_100.json
  each sets epoch_setting_name, epoch_data_name, compare_trltypes,
  compare_trltypes_select_column, cond1/cond2_content_name + _content_list,
  roi, tfr_method, freq_range_name.
'''

#%% import
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime

# for save data
import pickle
from EXP1_help_functions import (BidsPath)

from _help_functions import (general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 read_configfile,
                                 get_fw_inv,
                                 get_stcs_allsrc,
                                 read_labels_exp2,
                                 get_label_vertidx
                                 )
from _help_module import (get_trl_indices # specific for tfr to extract trial index based on condition
                              )
import numpy as np
import pandas as pd

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
    args.configfile = base / "config_files" / "pipeline_tfr_source" / "dAT_probe_stim_noft_nobc_deci5_nocut_PFC_30_100.json"

else:
    parser = argparse.ArgumentParser(
        description="Implements analysis of source erf for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                    help="Name of the subject")
    parser.add_argument('--configfile', type=str, default=None,
                        help="configfile file for analysis parameters (file name + path)")
    args = parser.parse_args()

#%% load configure and setting path
subject = args.subject
configfile  = args.configfile

# get the analyse name and configfile name to create folder to save
config_filename_with_ext = os.path.basename(configfile)
config_filename, _ = os.path.splitext(config_filename_with_ext)
print('--------- start ',config_filename)

# script_name = os.path.basename(os.path.dirname(configfile))
script_name = os.path.basename(__file__).replace('.py','')
         

# Read the configfile file:
param = read_configfile(configfile)

# unpack the param 
epoch_setting = general_param['epoch_setting_dict'][param['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
roi_params = general_param['roi_params_dict']

# !!! Specific needed for tfr: run one roi at one time
roi = param['roi']

# !!! Specific needed for tfr: 
trial_type_col= param['compare_trltypes_select_column']
cond1_name= param['cond1_content_name']
cond2_name= param['cond2_content_name']
cond1_values= param['cond1_content_list']
cond2_values= param['cond2_content_list']

# !!! Specific for tfr: use multitaper method
tfr_map = general_param['tfr_map_dict'][param['tfr_method']]
freq_range_name = param['freq_range_name']


info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info)
# !!! Specific needed for tfr: run one roi at one time
bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param['epoch_data_name']}_{param['compare_trltypes']}_{param['tfr_method']}_{roi}_{param['freq_range_name']}",
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

    print('------------','start subject' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
            )
    eps_use_result = get_eps_use(subject, 
                                 debug= True if flag_test else False,
                                ** epoch_data, 
                                ** epoch_setting)

    fw_inv_result  = get_fw_inv(
                        subject,
                        **eps_use_result,
                        **epoch_data,
                        **epoch_setting,
                        **general_param['source_method_setting'])
    src = fw_inv_result['src']
    
    # get stc for all epoch
    stcs_results = get_stcs_allsrc (
                        subject,
                        **eps_use_result,
                        **epoch_setting,
                        **general_param['source_method_setting'],
                        **fw_inv_result,
                        )


    stcs_epoch      =  stcs_results['stcs_epoch']
    epochs_metadata =  stcs_results['stcs_metadata']
    assert stcs_epoch[0].data.shape[-1] == len(eps_use_result['times']),\
        (f'number of times not match '
         f'{stcs_epoch[0].data.shape[-1]} vs {len(eps_use_result["times"])}')

    
    # ------------------ get label info ------------------ 
    label_dict = {}
    roilist_all = {}
    for one_roi in roi_params.keys():
        one_roi
        lb_use = read_labels_exp2(
            bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   =  epoch_data['exp_id']), 
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
                                            src    = fw_inv_result ['src'],
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
    
    
    #% ------------------ loop through labels get vft ------------------ 
    vft_labeldict ={}
    for label_name,label in roilist_all[roi].items():
        # ---------- constrained label by src ----------
        # if label of two hemi, combine two hemi and check the label dict 
        if ('rh' in label.name) and ('lh' in label.name):
            
            label_src = np.sum(
                        [label.lh.restrict(src),
                            label.rh.restrict(src)])
            print('------------ start ' ,roi,label_name,'left ',len(label_src.lh.vertices),
                    ' right ',len(label_src.rh.vertices),
                        datetime.now(),)
            
            # if label.subject =='fsaverage':
            if ( src[0]['subject_his_id'] =='fsaverage') or (label.subject =='fsaverage'):
                label_src.subject = 'fsaverage'
            elif subject in label.subject:
                label_src.subject = 'sub-' + subject
            else:
                ValueError
                
            assert label_dict[roi] ['v_l_dic'][label_name] == list(label_src.lh.vertices),'label_src not match v_l_dic'
            assert label_dict[roi] ['v_r_dic'][label_name] == list(label_src.rh.vertices),'label_src not match v_r_dic'
        
        # if label only have one hemi, check the label_dict 
        else:
            label_src = label.restrict(src)
            print('start ' ,roi,label_name,' one hemi ',len(label_src.vertices))
            # if label.subject =='fsaverage':
            if  (src[0]['subject_his_id'] =='fsaverage' ) or (label.subject =='fsaverage'):
                label_src.subject = 'fsaverage'
            elif subject in label.subject:
                label_src.subject = 'sub-' + subject
            else:
                ValueError
            assert label_dict[roi] ['v_dic'][label_name] == list(label_src.vertices),'label_src not match v_l_dic'
        
        # ---------- extract the label data based on condition----------
        # ----- validation -----
        if isinstance(trial_type_col, str):
            if trial_type_col not in epochs_metadata.columns:
                raise ValueError(f"Missing column '{trial_type_col}' in epochs_metadata.")
        elif  isinstance(trial_type_col, (list, tuple)):
            assert all([col in epochs_metadata.columns for col in trial_type_col])
        else:
            raise TypeError("column must be str or list of str")
        
        # ----- select trials per condition -----
        idx1 = get_trl_indices(
                            df = epochs_metadata,
                            column = trial_type_col, 
                            condition= cond1_values)
        idx2 = get_trl_indices(
                        df = epochs_metadata,
                        column = trial_type_col, 
                        condition= cond2_values)
        # Safety: no overlap
        if set(idx1) & set(idx2):
            raise ValueError("A trial belongs to both conditions.")
        assert len(idx1) + len(idx2) == len(stcs_epoch), "Selected trials not matching epochs."
        
    
        nepoch_dict= {cond1_name: len(idx1),
                      cond2_name: len(idx2)}
        df_nepoch = pd.DataFrame([nepoch_dict])
        
        
        cond1_tlvt = np.array([stc.in_label(label_src).data for idx,stc in  enumerate(stcs_epoch) if idx in idx1])
        cond2_tlvt = np.array([stc.in_label(label_src).data for idx,stc in  enumerate(stcs_epoch) if idx in idx2])

        # get power
        cond1_tlvft =  mne.time_frequency.tfr_array_multitaper(
            cond1_tlvt, 
            sfreq = eps_use_result['sfreq'],
            freqs         = tfr_map[freq_range_name]['freqs'], 
            n_cycles      = tfr_map[freq_range_name]['n_cycles'], 
            time_bandwidth= tfr_map[freq_range_name]['time_bandwidth'],
            decim         = tfr_map[freq_range_name]['tfr_decim'],
            use_fft=True, 
            output='power',
            verbose=True)
        
        cond2_tlvft =  mne.time_frequency.tfr_array_multitaper(
            cond2_tlvt, 
            sfreq = eps_use_result['sfreq'],
            freqs         = tfr_map[freq_range_name]['freqs'], 
            n_cycles      = tfr_map[freq_range_name]['n_cycles'], 
            time_bandwidth= tfr_map[freq_range_name]['time_bandwidth'],
            decim         = tfr_map[freq_range_name]['tfr_decim'],
            use_fft=True, 
            output='power',
            verbose=True)
        
        cond1_vft = np.mean(cond1_tlvft,0)
        cond2_vft = np.mean(cond2_tlvft,0)
        vft_labeldict[label_name] = {
            cond1_name:cond1_vft,
            cond2_name:cond2_vft,
        }
    #  save data
    save_data = {
                'roi':roi,
                'vft_labeldict':vft_labeldict,
                'param_config':param,
                'general_param_config':general_param,
                'epochs_metadata':epochs_metadata,
                'times'   :eps_use_result ['times'],
                'sfreq'   :eps_use_result['sfreq'],
                'df_nepoch':df_nepoch,
                'tfr_map':tfr_map,
                'label_dict':label_dict,
                'roi_params':roi_params,
                } 
    
    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)
    print('------','save' ,pklfile_name,
          '\n',subject ,
        '\n',datetime.now()) 
    
#%%