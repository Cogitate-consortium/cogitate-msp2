'''
Individual-level synchronization analysis: GNW prediction 1 and 2 using Pairwise Phase Consistency (PPC)

Configure file folder:
/config_files/pipeline_syn_ppc_GED_ROI/
    noft_nobc_deci1_nocut_AT_nontarget.json
    noft_nobc_deci1_nocut_AT_target.json
    noft_nobc_deci1_nocut_allseen.json
    noft_nobc_deci1_nocut_dAT_probeface.json
    noft_nobc_deci1_nocut_dAT_probeobje.json
    noft_nobc_deci1_nocut_dAT_probeseen.json

load time courses for the different ROIs (calculated in pipeline_syn_timecourse_GED_ROI.py)
calculate the Pairwise Phase Consistency (PPC, Vinck et al. 2010) between the
two ROIs' time courses, across trials, time- and frequency-resolved
(mne spectral_connectivity_epochs, method='ppc'). 

NOTE: this is the PPC. The MI-based dynamic functional connectivity(DFC) is computed by the sibling script
pipeline_syn_subsampling_dfc_GED_ROI.py (frites conn_dfc).

each config file sets the epoch data setting and epoch data so we know what
time course to load.
!!! To be noticed, the epoch data could be one epoch data or multiple.
each contains multiple ROI pairs (ROIpairs_dict) to be calculated.

in general_config.json
it specify the folder name depending on the roi type for the time course data calculated in previous step.

(This GED_ROI script only uses the GED time course.
The vert* entries using pipeline_syn_timecourse_PCA_VertROI are only used by
 pipeline_syn_subsampling_ppc_PCA_VertROI.py)
    "timecourse_data_dict":{
        "FFface": {
            "roi":"FF",
            "timecourse_folder": "pipeline_syn_timecourse_GED_ROI"
        }, 
        "FFobje": {
            "roi":"FF",
            "timecourse_folder": "pipeline_syn_timecourse_GED_ROI"
        },
        "PFCact": {
            "roi":"PFC",
            "timecourse_folder": "pipeline_syn_timecourse_GED_ROI"
            
        },
        "vertFFface": {
            "roi":"FF_face_select_vertices",
            "timecourse_folder": "pipeline_syn_timecourse_PCA_VertROI"
        },
        "vertFFobje": {
            "roi":"FF_obje_select_vertices",
            "timecourse_folder": "pipeline_syn_timecourse_PCA_VertROI"
        },
        "vertV1V2actVbas": {
            "roi":"V1V2_responsive_activewin_vs_baseline_vertices",
            "timecourse_folder": "pipeline_syn_timecourse_PCA_VertROI"
        }

'''

#%% import
from _help_module import lesser_greater
import statsmodels.api as sm

from mne_connectivity import spectral_connectivity_epochs

import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime

# for save data
import pickle

from _help_functions import (turn_string_to_num,
                             general_param,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
                                 get_trl_indices,
                                 
                                 )

import mne_bids
import numpy as np
import pandas as pd

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
    args.configfile = base / "config_files" / "pipeline_syn_ppc_GED_ROI" / "noft_nobc_deci1_nocut_allseen.json"

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

# !!! Specific for tfr
tfr_map = general_param['tfr_map_dict'][param['tfr_method']]
freq_range_name = param['freq_range_name']

# !!! Specific for subsampling
n_samples = 5 if flag_test else general_param['Nsample']

info = get_eps_use_info(**epoch_setting)

# !!! Specific for subsampling
ana_nosubsampling= 'pipeline_syn_ppc_GED_ROI'
deriv_root_nosubsampling = os.path.join(path_allana, ana_nosubsampling,
                          info.epoch_info)

# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info)
    

if "epoch_data_name" in param.keys():
    epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
    epoch_data_list = [epoch_data]
    epoch_data_name_list = [param['epoch_data_name']]

elif 'epoch_data_name_list' in param.keys():
    epoch_data_list = [general_param['epoch_data_dict'][name] for name in param['epoch_data_name_list']]
    epoch_data_name_list = param['epoch_data_name_list']
    epoch_data_name_list_name = param['epoch_data_name_list_name']
    assert(len(np.unique([general_param['epoch_data_dict'][name]['exp_id'] for name in param['epoch_data_name_list']])) ==1),\
        "All epoch_data in epoch_data_name_list should have the same exp_id"

    exp_id = general_param['epoch_data_dict'][param['epoch_data_name_list'][0]]['exp_id']

else:
    raise ValueError('No epoch_data_name or epoch_data_name_list in configfile')





for ROIpairs_name, ROIpairs in param['ROIpairs_dict'].items():
    print('------Processing PPC for ROI pairs:', ROIpairs_name, 
        '\n',subject,
        '\n',config_filename,
        '\n',datetime.now(),
            )
    ROI_signal1 = ROIpairs['signal1']
    ROI_signal2 = ROIpairs['signal2']
    # the path is for the input for PPC calculation:
    # the time series resulting from GED
    # (i.e. results of pipeline_syn_timecourse_GED_ROI.py)
    deriv_root_ged = os.path.join(path_allana, 
                                general_param['timecourse_data_dict'][ROI_signal1]['timecourse_folder'],
                                info.epoch_info)

    
    if 'epoch_data_name_list' in param.keys():
        bids_path_1 = mne_bids.BIDSPath(
                root=deriv_root_ged, 
                subject= subject,
                session= exp_id,  
                datatype='meg',  
                suffix=f"{param['epoch_data_name_list_name']}_{ROI_signal1}",
                extension='.pkl',
                check=False)
        bids_path_2 = mne_bids.BIDSPath(
                root= deriv_root_ged, 
                subject= subject, 
                session= exp_id,  
                datatype='meg',  
                suffix=f"{param['epoch_data_name_list_name']}_{ROI_signal2}",
                extension='.pkl',
                check=False)
    elif "epoch_data_name" in param.keys():
        bids_path_1 = mne_bids.BIDSPath(
                root= deriv_root_ged, 
                subject= subject, 
                session= epoch_data['exp_id'],  
                datatype='meg',  
                task=epoch_data['task_id'],
                suffix=f"{param['epoch_data_name']}_{ROI_signal1}",
                extension='.pkl',
                check=False)
        bids_path_2 = mne_bids.BIDSPath(
                root= deriv_root_ged, 
                subject= subject, 
                session= epoch_data['exp_id'],  
                datatype='meg',  
                task=epoch_data['task_id'],
                suffix=f"{param['epoch_data_name']}_{ROI_signal2}",
                extension='.pkl',
                check=False)
        
    pklfile_name_1 = bids_path_1.fpath
    pklfile_name_2 = bids_path_2.fpath
    with open(pklfile_name_1, 'rb') as pickle_file_load:
        loaded_data_1 = pickle.load(pickle_file_load)
    with open(pklfile_name_2, 'rb') as pickle_file_load:
        loaded_data_2 = pickle.load(pickle_file_load)
    print('------------Successfully loaded' ,subject,
        '\n',config_filename,
        '\n',datetime.now(),
            )
    tlt_all_1 = loaded_data_1['tlt']  # shape: n_trials * n_timepoints
    tlt_all_2 = loaded_data_2['tlt']  # shape: n_trials * n_timepoints
    sfreq = loaded_data_1['sfreq']
    times = loaded_data_1['times']
    # sanity check: timecourse from two ROIs should have the same shape, metadata, sfreq, times
    assert tlt_all_1.shape == tlt_all_2.shape, "Mismatch in shape of tlt_all_1 and tlt_all_2"
    assert loaded_data_1['stcs_metadata'].equals(loaded_data_2['stcs_metadata']),\
          "Mismatch in stcs_metadata of loaded_data_1 and loaded_data_2"
    assert loaded_data_1['sfreq'] == loaded_data_2['sfreq'],\
          "Mismatch in sfreq of loaded_data_1 and loaded_data_2"
    assert np.array_equal(loaded_data_1['times'], loaded_data_2['times']),\
          "Mismatch in times of loaded_data_1 and loaded_data_2"
    assert times[0] != 0, "time should not start from 0s"


    # ------ get the trial number to decide the minimum trial number across conditions
    idx1 = get_trl_indices(loaded_data_1['stcs_metadata'], 
                                       param['trl_select_columns'][0], 
                                       param['trl_select_conditions'][0])
    idx2 = get_trl_indices(loaded_data_1['stcs_metadata'],
                                        param['trl_select_columns'][1], 
                                        param['trl_select_conditions'][1])
    
    less_name, more_name, n_less, n_more, idx_less, idx_more = lesser_greater(
        name_a=param['trl_types'][0], 
        name_b=param['trl_types'][1], 
        idx_a=idx1, 
        idx_b=idx2)
    
    a ={ "subject": subject,less_name: n_less, more_name: n_more}
    df_count = pd.DataFrame.from_dict(a, orient='index').T
    print(f"Subject {subject}:less {less_name} = {n_less} trials; more {more_name} = {n_more} trials")
    




    for ii, trl_type in enumerate (param['trl_types']):
        cond_tlidx_1 = get_trl_indices(loaded_data_1['stcs_metadata'], 
                                       param['trl_select_columns'][ii], 
                                       param['trl_select_conditions'][ii]) 
        cond_tlidx_2 = get_trl_indices(loaded_data_2['stcs_metadata'], 
                                       param['trl_select_columns'][ii], 
                                       param['trl_select_conditions'][ii]) 
        assert np.array_equal(cond_tlidx_1, cond_tlidx_2),\
            f"Mismatch in trial indices for condition {trl_type} between two ROIs"
        assert len(cond_tlidx_1) >= n_less, \
            f"Number of trials for condition {trl_type} >= n_less {n_less}"
        assert len(cond_tlidx_2) == n_less or len(cond_tlidx_2) == n_more, \
            f"Number of trials for condition {trl_type} should be either n_less {n_less} or n_more {n_more}"
        
        tlt1_raw = tlt_all_1[cond_tlidx_1,:]
        tlt2_raw = tlt_all_2[cond_tlidx_2,:]

        for  input_type in ['NOrm','rmE']:

            # preset path for save ppc result
            if 'epoch_data_name_list' in param.keys():
                epoch_data_name = param['epoch_data_name_list_name']
                suffix=f"{epoch_data_name}_{input_type}_{param['tfr_method'].replace("_","")}_{param['freq_range_name']}_{ROIpairs_name}_{trl_type}"
                bids_path = mne_bids.BIDSPath(
                        root=deriv_root, 
                        subject= subject, 
                        session= exp_id,  
                        datatype='meg',  
                        suffix=suffix,
                        extension='.pkl',
                        check=False)
                bids_path_nosubsampling = mne_bids.BIDSPath(
                        root=deriv_root_nosubsampling, 
                        subject= subject, 
                        session= exp_id,  
                        datatype='meg',  
                        suffix=suffix,
                        extension='.pkl',
                        check=False)

            elif "epoch_data_name" in param.keys():
                epoch_data_name = param['epoch_data_name']
                suffix=f"{epoch_data_name}_{input_type}_{param['tfr_method'].replace("_","")}_{param['freq_range_name']}_{ROIpairs_name}_{trl_type}"
                bids_path = mne_bids.BIDSPath(
                        root=deriv_root, 
                        subject= subject, 
                        session= epoch_data['exp_id'],  
                        datatype='meg',  
                        suffix =suffix,
                        extension='.pkl',
                        check=False)
                bids_path_nosubsampling = mne_bids.BIDSPath(
                        root=deriv_root_nosubsampling, 
                        subject= subject, 
                        session= epoch_data['exp_id'],  
                        datatype='meg',  
                        suffix =suffix,
                        extension='.pkl',
                        check=False)
            os.makedirs(os.path.dirname( bids_path.fpath), exist_ok=True)      
            pklfile_name_ppc = bids_path.fpath

            if os.path.exists(pklfile_name_ppc):
                print('------------',ROIpairs_name, input_type, trl_type,'already exists, skip' ,
                    '\n',pklfile_name_ppc,
                    '\n',subject,
                    '\n',config_filename,
                    '\n',datetime.now(),)
                del suffix
                continue

            # directly resave non-subsampling data if the trial number equals to the less condition
            if len(cond_tlidx_1) == n_less:
                with open(bids_path_nosubsampling.fpath, 'rb') as pickle_file_load:
                    loaded_non_subsampling = pickle.load(pickle_file_load)

                with open(pklfile_name_ppc, 'wb') as pickle_file:
                    pickle.dump(loaded_non_subsampling, pickle_file)
                print('------------ directly resave non subsampling data',ROIpairs_name, input_type, trl_type,'save' ,
                    '\n',pklfile_name_ppc,
                    '\n',subject,
                    '\n',config_filename,
                    '\n',datetime.now(),)
                continue

            elif len(cond_tlidx_1) > n_less:
                print('------------ calculating subsampling PPC',ROIpairs_name, input_type, trl_type,
                    '\n',subject,
                    '\n',config_filename,
                    '\n',datetime.now(),)
                
                # ------ setting for PPC analysis
                freqs = np.array(tfr_map[freq_range_name]['freqs'])
                n_cycles = tfr_map[freq_range_name]['n_cycles']
                nf = len(freqs)
                nt= len(times)

                # ------ subsampling trials to calculate PPC
                seed = turn_string_to_num(f'{subject}_{epoch_data_name}_{ROIpairs_name}')
                rng = np.random.default_rng(seed)
                sample_ft = np.empty((n_samples, nf, nt))

                sample_indices = np.empty((n_samples, n_less), dtype=int)
                for samplei in range(n_samples):
                    print('------ subsample ', samplei+1, 'of', n_samples, subject, config_filename, datetime.now())

                    idx = rng.choice(n_more, size=n_less, replace=False)
                    sample_indices[samplei] = idx

                    tlt1 = tlt1_raw[idx,:] 
                    tlt2 = tlt2_raw[idx,:] 


                    # regresses out mean evoked response from individual trials
                    tlt1_mean = np.mean(tlt1, axis=0)
                    tlt1_remove = np.zeros_like(tlt1)
                    for tli in range(tlt1.shape[0]):
                        tlt1_remove[tli,:] = sm.OLS( tlt1[tli,:],tlt1_mean).fit().resid

                    tlt2_remove = tlt2.copy()
                    tlt2_mean = np.mean(tlt2, axis=0)
                    for tli in range(tlt2.shape[0]):
                        tlt2_remove[tli,:] = sm.OLS(
                            tlt2_remove[tli,:],
                            tlt2_mean).fit().resid

                    # trials * source of signals * times
                    tlst = np.stack( (tlt1, tlt2), axis=1)
                    tlst_remove = np.stack( (tlt1_remove, tlt2_remove), axis=1)

                    if input_type == 'NOrm':
                        tlst_input = tlst
                    elif input_type == 'rmE':
                        tlst_input = tlst_remove
                    else:
                        raise ValueError('input_type should be Norm or rmE')


                    # calculate time-frequency connectivity measure: PPC for each frequency & time point
                    con = spectral_connectivity_epochs(
                            tlst_input , 
                            sfreq=         sfreq, 
                            cwt_freqs=     freqs, 
                            cwt_n_cycles=  n_cycles,
                            method=        'ppc', 
                            mode=          param['tfr_method'],  
                            indices=       (np.array([0]),   # row indices
                                            np.array([1])),  # col indices
                            verbose = False) # shape: n_signal_pairs * n_freqs * n_times
                    ft = np.squeeze(con.get_data())  
                    assert np.equal(con.freqs, freqs).all(), "Mismatch in con frequency bins"
                    # con.times is from 0 
                    assert len(con.times )== len(times), "Mismatch in con time points"
                    assert con.get_data().shape == (1,len(freqs), len(times)), "Mismatch in con data shape"

                    sample_ft[samplei,:,:] = ft

                # average across subsamples
                ft = np.mean(sample_ft, axis=0)  # shape: n_freqs * n_times
                assert times[0] != 0, "time should not start from 0s"
                save_data = {
                            'ft':ft,
                            'freqs':freqs,
                            'times':times,
                            'sfreq':sfreq,
                            'tfr_map':tfr_map,
                            'param_config':param,
                            'general_param_config':general_param,
                            "sample_indices":sample_indices,
                            "seed": seed,
                            "df_count": df_count,
                            'n_trials':len(cond_tlidx_1),
                            "n_less": n_less,
                            "n_more": n_more,
                            'name_less': less_name,
                            'name_more': more_name,
                            }
                with open(pklfile_name_ppc, 'wb') as pickle_file:
                    pickle.dump(save_data, pickle_file)
                print('------------save data',ROIpairs_name, input_type, trl_type,
                    '\n',pklfile_name_ppc,
                    '\n',subject,
                    '\n',config_filename,
                    '\n',datetime.now(),)



print('------','final all' ,
    '\n',ROIpairs_name,
    '\n',subject,
    '\n',config_filename,
    '\n',datetime.now(),
        )


#%%