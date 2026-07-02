'''
Individual-level control analysis: sensor-level ERF, per subject.

For one subject + one config it loads the epoched sensor data and, via
compute_all_combinations, builds the trial-mean condition-averaged channel x time
ERFs for cond1 vs cond2 with trial-count balancing (the larger condition is
subsampled to match the smaller):
  ct_dict['T_mean']['balanced'][cond_content_name] = (n_ch, n_time)
  (keyed by the two condition content names; an 'unbalanced' variant is also
   stored, but the balanced one is what is used downstream.)
Saves ct_dict + nepoch_df + times/sfreq/ch_names as
<epoch_data_name>_<compare_trltypes>.pkl under path_allana/<script_name>/<epoch_info>/.

Consumed by the group script pipeline_erf_sensor_group.py, which reads
ct_dict['T_mean']['balanced'].

Config folder: /config_files/pipeline_erf_sensor/
    dAT_probe_stim_ft_bc_deci1_nocut.json
  each sets epoch_setting_name, epoch_data_name, compare_trltypes,
  compare_trltypes_select_column, cond1/cond2_content_name + _content_list.
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

from _help_functions import (general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 turn_sub_code_to_num,
                                 read_configfile,
                                 )
from _help_module import compute_all_combinations
import pandas as pd

import mne_bids

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
    args.subject = 'SB031'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_erf_sensor" / "dAT_probe_stim_ft_bc_deci1_nocut.json"

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

info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info)

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param['epoch_data_name']}_{param['compare_trltypes']}",
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
    
    res = compute_all_combinations(
        epochs=epochs,
        epochs_metadata=epochs_metadata,
        order = ("trials",),   
        epochdataa_dims_list= ["trials", "channels", "times"],
        trial_type_col= param['compare_trltypes_select_column'],
        cond1_name= param['cond1_content_name'],
        cond2_name= param['cond2_content_name'],
        cond1_values= param['cond1_content_list'],
        cond2_values= param['cond2_content_list'],
        vertex_idx=None ,     
        n_subsamples= 20 if flag_test else general_param['Nsample'] ,
        seed= turn_sub_code_to_num(subject)            
    )

    nepoch_df = pd.DataFrame([res[list(res.keys())[0]]['counts']])
    
    #  save data
    save_data = {
                'ct_dict':res,
                'param_config':param,
                'nepoch_df':nepoch_df,
                'sfreq':eps_use_result['sfreq'],
                'epochs_metadata':epochs_metadata,
                'times'   :eps_use_result ['times'],
                'ch_names':eps_use_result ['ch_names'],
                } 

    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)
    print('------','save' ,pklfile_name,
        '\n',subject,
        '\n',config_filename,
        '\n',datetime.now(),
            )


#%%