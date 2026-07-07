'''
Per-subject pre-save of source-reconstruction info, so downstream source
analyses can reuse it instead of recomputing the inverse each time.

For one subject + one config it computes the forward/inverse solution
(get_fw_inv) and saves, under path_allana/<script_name>/:
  - the inverse operator                   (..._inv.fif)
  - inverse info: covariances + rank       (..._inv_info.pkl)
  - one example source estimate            (..._example_stc-lh/rh.stc)
If all three already exist, it loads them instead of recomputing.

Config folder: /config_files/pipeline_presave_sourceinfo/
    replay_ep_noft_nobc_deci1_nocut.json   (replay task)
    vg_ep_noft_nobc_deci1_nocut.json       (video-game task)
  each sets epoch_setting_name + epoch_data_name.
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
                                 read_configfile,
                                 get_fw_inv,
                                 get_stcs_allsrc,
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
    args.subject = 'SA121'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_presave_sourceinfo" / "replay_ep_noft_nobc_deci1_nocut.json"

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
print('--------- start ',config_filename,datetime.now())

script_name = os.path.basename(__file__).replace('.py','')
         
# Read the configfile file:
param = read_configfile(configfile)
    
epoch_setting = general_param['epoch_setting_dict'][param['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]

info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                         "test")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          )
# filename of stc example
bids_path_stc = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix='example_stc',
        check=False)
stc_filename = bids_path_stc.fpath 
os.makedirs(os.path.dirname( stc_filename), 
            exist_ok=True)  
# filename of inv
bids_path_inv = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix='inv',
        extension='.fif',
        check=False)
inv_filename = bids_path_inv.fpath
os.makedirs(os.path.dirname( inv_filename), 
            exist_ok=True)  


pklfile_name = str(inv_filename).replace('_inv.fif',
                                           '_inv_info.pkl')

#%% get the results
if  os.path.exists(f"{str(stc_filename)}-lh.stc") & (os.path.exists(inv_filename)) & (os.path.exists(pklfile_name)):
    print('------------','exist subject' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )
    try:
        stc_exp = mne.read_source_estimate(stc_filename)
        print('------------','Successfully loaded stc ' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )    
        

        inv = mne.minimum_norm.read_inverse_operator(inv_filename)
        print('------------','Successfully loaded inv' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )  
          
        with open(pklfile_name, 'rb') as pickle_file_load:
            loaded_data = pickle.load(pickle_file_load)
        print('------------','Successfully loaded' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )    
        
    except Exception :
        # Print the error for debugging purposes (optional)
        print('------------','Error loading' ,stc_filename,
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
                                ** epoch_data, 
                                ** epoch_setting)


    fw_inv_result  = get_fw_inv(
                        subject,
                        **eps_use_result,
                        **epoch_data,
                        **epoch_setting,
                        **general_param['source_method_setting'])
    
    # save inv
    mne.minimum_norm.write_inverse_operator(inv_filename, 
                                            fw_inv_result['inverse'],
                                            overwrite=True)   
    print('------------',' save inv '  ,subject,
            '\n',config_filename,
            '\n',datetime.now(),)
    
    # save inv info
    save_data = {
                'covs':fw_inv_result['covs'],
                'rank':fw_inv_result['rank'],
                } 
    with open(pklfile_name, 'wb') as pickle_file:
            pickle.dump(save_data, pickle_file)
    print('------------',' save inv info '  ,subject,
            '\n',config_filename,
            '\n',datetime.now(),)
    
    # get stc for all epoch
    stcs_results = get_stcs_allsrc (
                        subject,
                        eps_use = eps_use_result["eps_use"][0],
                        **epoch_setting,
                        **general_param['source_method_setting'],
                        **fw_inv_result,
                        )
    
    # save one stc_epoch
    stcs_results['stcs_epoch'][0].save(stc_filename,
                                       overwrite=True)
    print('------------',' save one stc '  ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )


#%%