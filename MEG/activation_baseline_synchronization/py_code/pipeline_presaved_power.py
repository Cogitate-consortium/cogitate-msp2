'''
Per-subject pre-save of source-level band power, so downstream ROI/vertex
analyses can load it instead of recomputing the source TFR each time.

For one subject + one config it:
  - builds source estimates for all epochs (all vertices) via the inverse,
  - computes Morlet time-frequency power for the config's band and averages
    over the band frequencies -> pow_tlvt (trial x vertex x time),
  - also stores per-ROI vertex indices (label_dict) for later extraction.
Saves <band_name>_power.pkl under path_allana/<script_name>/; skips if it exists.

Config folder: /config_files/pipeline_presaved_power/
    alphapower_replay_ep_noft_nobc_deci5_cutpre1000_1000.json
    alphapower_vg_ep_noft_nobc_deci5_cutpre1000_1000.json
    gammapower_replay_ep_noft_nobc_deci5_cutpre1000_1000.json
    gammapower_vg_ep_noft_nobc_deci5_cutpre1000_1000.json
  each sets band_name + epoch_setting_name + epoch_data_name.

Runtime / memory (whole source space): replay ~30 min, 350G; vg ~20 min, 100G.
Run ~2 subjects in parallel per node.
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
                                    get_label_vertidx,
                                 )

import numpy as np

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
    args.subject = 'SA121'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_presaved_power" / "alphapower_replay_ep_noft_nobc_deci5_cutpre1000_1000.json"

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
    
epoch_setting = general_param['epoch_setting_dict'][param['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
roi_params = general_param['roi_params_dict']
band_map = general_param['band_map']
band_name = param['band_name']


info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          "test")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          )

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f'{band_name}_power',
        extension='.pkl',
        check=False)
dir_analyse = os.path.dirname( bids_path.fpath)
os.makedirs(dir_analyse, exist_ok=True)      

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
    epochs = eps_use_result["eps_use"]
    epochs_metadata   = eps_use_result["epochs_metadata"]
    
    
    # Read forward model
    fw_inv_result  = get_fw_inv(
                        subject,
                        **eps_use_result,
                        **epoch_data,
                        **epoch_setting,
                        **general_param['source_method_setting'])
    
    # get stc for all epoch
    stcs_results = get_stcs_allsrc (
                        subject = subject,
                        **epoch_setting,
                        **general_param['source_method_setting'],
                        **fw_inv_result,
                        **eps_use_result)
    
    # Create label information for future analysis
    label_dict = {}
    for roi in roi_params.keys():
        lb_use = read_labels_exp2(
            bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   = epoch_data['exp_id']
                                    ), 
            parc       = roi_params[roi]['parc'], 
            labels_list= roi_params[roi]['labels_list'],
            rois_list  = roi_params[roi]['rois_list'], 
            combine_labels_list=True, 
            merge_hemi=True
            )  

        # # get vertno index
        label_vertidx_dict=\
                    get_label_vertidx(lb_use =  lb_use,
                                        src = fw_inv_result['inverse']['src'],
                                        flag_return_dict =1)
        label_dict[roi] = label_vertidx_dict
            
    
    # get power
    tlvt = np.array([x.data for x in   stcs_results['stcs_epoch']])
    pow_tlvft_morlet =  mne.time_frequency.tfr_array_morlet(
        tlvt, 
        sfreq         = stcs_results['sfreq'] ,
        freqs         = band_map[band_name]['freqs'], 
        n_cycles      = band_map[band_name]['n_cycles'], 
        decim         = band_map[band_name]['decim'],
        use_fft=True, 
        output='power',
        n_jobs = 1,
        verbose=True)
    pow_tlvt = np.mean(pow_tlvft_morlet,2) 

    #  save data
    save_data = {
                'pow_tlvt':pow_tlvt,
                'sfreq':stcs_results['sfreq'],
                'epochs_metadata':epochs_metadata,
                'times':stcs_results['times'],
                'label_dict':label_dict,
                'band_map':band_map,
                'roi_params':roi_params,
                'general_param_config':general_param,
                'param_config':param,
                } 

    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)
    print('------------','finish save '  ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )


#%%