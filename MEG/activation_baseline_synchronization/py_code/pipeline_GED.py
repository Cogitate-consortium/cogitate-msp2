'''
Individual-level: GED
construct a GED (Generalized Eigenvalue Decomposition) spatial
filter for one ROI / one GED_type per config, and extract & save its component
time courses (used later by pipeline_syn_timecourse_GED_ROI.py for the
synchronization analysis, GNW nodes).

One config = one roi (param['roi'], e.g. FF / PFC) + one GED_type
(param['GED_type'], e.g. FFface / FFobje / PFCact).

Steps:
- Build the source-level ROI data and split into the GED contrast:
  signal = stim active vs reference = no-stim active (seperate_stcs_for_GED).
- get_ged (EXP1_help_functions): trial-wise covariances -> remove outlier trials
  (>3 SD) and average -> shrinkage-regularize the reference covariance
  (gamma = 0.01) -> generalized eigendecomposition eigh(cov_signal, cov_ref),
  eigenvalues/eigenvectors sorted descending. The selected spatial filter is the
  eigenvector with the largest eigenvalue (eigenvectors[:,0]).
- ged_get_time_course applies the top filter (evecs[:,0].T @ data, i.e. w.T X) to
  the test-condition stcs to get single-trial GED component time courses.

Saved (suffix GED_{GED_type}):
  eigenvalues, eigenvectors (full matrix; downstream uses the first column only),
  eigenvector_posmaxsign (top filter sign-flipped to positive max-abs weight;
  kept for reference, not used downstream),
  roi, label_dict, ts_stim_trl / ts_nostim_trl / ts_irrstim_trl, times, sfreq,
  param_config.

Config folder: /config_files/pipeline_GED/
    noft_nobc_deci1_nocut_FFface.json     (roi FF,  wide-band)
    noft_nobc_deci1_nocut_FFobje.json     (roi FF,  wide-band)
    ft30_nobc_deci1_nocut_PFCact.json     (roi PFC, 30 Hz low-pass)
each config sets the epoch setting + epoch data and the roi / GED_type to build.

Output location:
  This script saves under  path_allana / <script_name> / <epoch_info>, i.e.
  .../pipeline_GED/<epoch_info>/  (folder named after this script; <epoch_info>
  is derived from the epoch_setting, not from the config file name).
  So FF filters land in    pipeline_GED/noft_nobc_deci1_nocut/  (wide-band), and
  the PFCact filter lands in pipeline_GED/ftN_30_nobc_deci1_nocut/  (30 Hz
  low-pass "frontal slow-frequency").
  Downstream, general_config.json GED_params_dict[<epochtype>][<GED_type>]
  ['ged_subfolder'] stores just the <epoch_info> run-folder name, so
  pipeline_syn_timecourse_GED_ROI.py rebuilds the path as
  os.path.join(path_allana, 'pipeline_GED', ged_subfolder) to load each filter, e.g.
  GED_params_dict['default_noft_nobc']['PFCact']['ged_subfolder']
      = "ftN_30_nobc_deci1_nocut"  ->  <path_allana>/pipeline_GED/ftN_30_nobc_deci1_nocut
'''

#%% import
from EXP1_help_functions import (get_ged,
                                 ged_get_time_course,)
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
                                 seperate_stcs_for_GED,
                                 get_stcs_of_label,
                                 read_labels_exp2,
                                 get_label_vertidx
                                 )

import numpy as np

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
    args.subject = 'SA108'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_GED" / "noft_nobc_deci1_nocut_FFface.json"

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
roi_params = general_param['roi_params_dict']

# !!! special for GED
roi = param['roi']
GED_type = param['GED_type']
epoch_data_list = param['epoch_data_list']

GEDpref_stim = param['GEDpref_stim']
GEDirre_stim = param['GEDirre_stim']


info =get_eps_use_info(**epoch_setting) 

# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}",
                          )
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info,
                          )

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        datatype='meg',  
        suffix=f"GED_{GED_type}",
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
    # ---- prepare data for calculate GED 
    stcs_stim_active_train   = []
    stcs_nostim_active_train = []
    stcs_stim_test    = []
    stcs_nostim_test  = []
    stcs_irrstim_test = []
    label_dict ={}
    for epoch_data_name in epoch_data_list:
        epoch_data = general_param['epoch_data_dict'][epoch_data_name]
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
        
        lb_use = read_labels_exp2(
            bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   =  epoch_data['exp_id']), 
            parc       = roi_params[roi]['parc'], 
            labels_list= roi_params[roi]['labels_list'],
            rois_list  = roi_params[roi]['rois_list'], 
            combine_labels_list=True, 
            merge_hemi=True
            ) 
        if len(lb_use) ==1:
            label_all = [v for k,v in lb_use.items()][0]
        else:
            label_all = [v for k,v in lb_use.items() if 'all' in k][0]
        print('read label for',roi, list(lb_use.keys()))
        
        ## get vertno index
        label_vertidx_dict=\
                    get_label_vertidx(lb_use =lb_use,
                                      src = fw_inv_result ['src'],
                                        flag_return_dict =1)
        label_dict[epoch_data_name] = label_vertidx_dict


        # get stc of label
        stcs_of_label = get_stcs_of_label (
                            subject = subject,
                            label   = label_all,
                            **epoch_setting,
                            **general_param['source_method_setting'],
                            **fw_inv_result,
                            **eps_use_result)
        
        # prepare stcs for GED
        stcs_prepare_for_GED = seperate_stcs_for_GED(
                            subject = subject,
                            **stcs_of_label,
                            **param)
        # validate the number of stcs match the number of epochs
        if param["cond_GEDirre_select_columns"] not in [None, 'None']:
            assert len(stcs_prepare_for_GED ['stcs_stim'])\
                +len(stcs_prepare_for_GED ['stcs_nostim'])\
                ==len(eps_use_result['eps_use']) ,'error: number of stcs not match number of epochs'

        if 'times' in locals():
            assert all(times == stcs_prepare_for_GED['times']),'error: times not match'
        else:
            times = stcs_prepare_for_GED['times']
            
        if 'sfreq' in locals():
            assert sfreq == stcs_prepare_for_GED['sfreq'],'error: times not match'
        else:
            sfreq = stcs_prepare_for_GED['sfreq']

        # combine the stcs from different epoch_data    
        stcs_stim_active_train   = stcs_stim_active_train +\
                                    stcs_prepare_for_GED['stcs_stim_active']
        stcs_nostim_active_train = stcs_nostim_active_train +\
                                    stcs_prepare_for_GED['stcs_nostim_active']
        stcs_stim_test    = stcs_stim_test +\
                                stcs_prepare_for_GED['stcs_stim']
        if param["cond_GEDirre_select_columns"] not in [None, 'None']:
            stcs_nostim_test  = stcs_nostim_test +\
                                stcs_prepare_for_GED['stcs_nostim']
            stcs_irrstim_test = stcs_irrstim_test +\
                                    stcs_prepare_for_GED['stcs_irrstim']
    
    # ---- Create GED 
    eigenvalues, eigenvectors = get_ged(
        stcs_cond_act   = stcs_stim_active_train, 
        stcs_nocond_act = stcs_nostim_active_train, 
        condition = None, 
        bids_task = None, 
        label_name = None, 
        bids_paths = None,
        save=False)
    
    # select first first_eigenvector that associated with the largest eigenvalue
    first_eigenvector = eigenvectors[:,0]
    
    # flip to make sure the sign of first_eigenvector to be positive 
    maxabs_idx = np.argmax(np.abs( first_eigenvector ))
    eigenvector_posmaxsign = first_eigenvector * np.sign(first_eigenvector[maxabs_idx])
    
    # ---- get GED time course
    ts_stim_trl = ged_get_time_course(
    stcs = stcs_stim_test, 
    evecs= eigenvectors, 
    desc = None, 
    bids_task = None, 
    bids_paths = None,
    prep_erf=False, 
    save=False)

    if param["cond_GEDirre_select_columns"]  in [None, 'None']:
        ts_nostim_trl = None
        ts_irrstim_trl = None
    else:
        # of nonstim conditon
        ts_nostim_trl = ged_get_time_course(
        stcs = stcs_nostim_test, 
        evecs= eigenvectors, 
        desc = None, 
        bids_task = None, 
        bids_paths = None,
        prep_erf=False, 
        save=False)
        
        # of irrstim conditon
        ts_irrstim_trl = ged_get_time_course(
        stcs = stcs_irrstim_test, 
        evecs= eigenvectors, 
        desc = None, 
        bids_task = None, 
        bids_paths = None,
        prep_erf=False, 
        save=False)


    #  save data
    save_data = {
                'eigenvalues':eigenvalues,
                'eigenvectors':eigenvectors,
                'eigenvector_posmaxsign':eigenvector_posmaxsign,
                'roi':roi,
                'label_dict':label_dict,
                'ts_stim_trl':ts_stim_trl,
                'ts_nostim_trl':ts_nostim_trl,
                'ts_irrstim_trl':ts_irrstim_trl,
                'times':times,
                'sfreq':sfreq,
                'param_config':param,
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