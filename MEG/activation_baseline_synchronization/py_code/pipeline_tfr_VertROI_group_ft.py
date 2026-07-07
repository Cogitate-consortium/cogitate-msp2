'''
Group-level (freq x time): oscillatory power difference between two conditions
across the full frequency x time plane, for one vertex-ROI, on the source-space
TFR power computed by pipeline_tfr_VertROI.py. Same as
pipeline_tfr_source_group_ft.py but on the IIT-selected vertex-ROIs
(VertROI_params_dict; stimulus vs blank baseline).

Run first: pipeline_tfr_VertROI.py (individual, per ROI + freq_range_name).

Loads each subject's per-label (vertices, freqs, times) power, then over the
group time window:
  - average power over vertices -> per-label (freqs, times),
  - log10(cond1/cond2) power ratio per subject / label / freq / time.
Statistics on the label-averaged log10 power ratio:
  - cluster-based permutation test (permutation_cluster_1samp_test, tail from
    config, clustering over labels x freqs x times) -> cbpt_* pkl/csv.
Unlike pipeline_tfr_source_group_ft.py, the per freq x time Bayesian BF map is
NOT computed here; it is computed in pipeline_tfr_VertROI_group_ft_plot.py.

Reduction of each label's (vertices, freqs, times) power for the test:
  vertices     -> averaged over all label vertices,
  frequencies  -> kept (no band averaging),
  times        -> kept.
So each label is a 2D freqs x times map and clustering is over labels x freqs x times.

Config folder: /config_files/pipeline_tfr_VertROI_group_ft_baseline/
    fast_30_100_pre500_0.json
    slow_1_30_pre500_0.json
  each sets cond1/cond2_timewindow, tail, cbpt_params_name.
Paired with an individual config /config_files/pipeline_tfr_VertROI/*.json.
'''

#%% import
# for statistic testing
import scipy

from _help_functions import (group_get_sublist,
                             get_file_name,
                             get_eps_use_info,
                                 read_configfile,
                                 turn_string_to_num,
                                 roi_longname_dict,# for construct dataframe
                                 )
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
                                 )
import numpy as np
import pandas as pd

import mne_bids
import mne

from IPython.display import display
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
    args.group_ana = 'pipeline_tfr_VertROI_group_ft'
    args.group_configfolder = 'pipeline_tfr_VertROI_group_ft_baseline'
    args.group_configfile = base / "config_files" / "pipeline_tfr_VertROI_group_ft_baseline" / "fast_30_100_pre500_0.json"

    
    args.individual_ana = 'pipeline_tfr_VertROI'
    args.individual_configfolder = 'pipeline_tfr_VertROI'
    args.individual_configfile      = base / "config_files" / "pipeline_tfr_VertROI" / "dAT_probe_stim_noft_nobc_deci5_nocut_vertFFstiVbla_30_100.json"



else:
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
            description="Implements analysis of GEDs for experiment2")
    parser.add_argument('--group_ana', type=str, default=None,
                        help="group analysis name)")
    parser.add_argument('--individual_ana', type=str, default=None,
                        help="individual analysis name")
    parser.add_argument('--group_configfolder', type=str, default=None,
                        help="group configure folder name)")
    parser.add_argument('--individual_configfolder', type=str, default=None,
                        help="individual configure folder name")
    parser.add_argument('--group_configfile', type=str, default=None,
                        help="configfile file for group analysis parameters (file name + path)")
    parser.add_argument('--individual_configfile', type=str, default=None,
                        help="configfile file for individual analysis parameters (file name + path)")
    args = parser.parse_args()


#%% load configure and setting path
group_ana = args.group_ana
individual_ana = args.individual_ana
group_configfolder = args.group_configfolder
individual_configfolder = args.individual_configfolder
group_configfile  = args.group_configfile
individual_configfile = args.individual_configfile

# get the analyse name and configfile name to create folder to save
config_group = get_file_name(group_configfile)
config_individual = get_file_name(individual_configfile)

config_setting = f"{config_individual}_{config_group}"
print('--------- start ',config_setting)

# script_name = os.path.basename(os.path.dirname(configfile))
script_name = os.path.basename(__file__).replace('.py','')
      

# Read the configfile file:
param_sub = read_configfile(individual_configfile)
param_group = read_configfile(group_configfile)


# get the sublist
path_sublist = os.path.join(path_allana, 
                            individual_ana,
                            'analysis_participant_list',)
os.makedirs(path_sublist, exist_ok=True) 

sub_list_name, sub_list = group_get_sublist(param_group,
                                            param_sub,
                                            group_configfile,
                                            individual_configfile,
                                            )
if flag_test:
    sub_list = sub_list[0:3] 
else:
    pd.DataFrame({'sub_list':sub_list}).to_csv(
        os.path.join(path_sublist,
        f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
        )


#%% unpack the param_sub 
epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param_sub['epoch_data_name']]

# !!!!! specific for VertROI
roi_params = general_param['VertROI_params_dict']

# !!! Specific needed for tfr: run one roi at one time
roi = param_sub['roi']
# !!! Specific needed for tfr: 
trial_type_col= param_sub['compare_trltypes_select_column']
cond1_content_name= param_sub['cond1_content_name']
cond2_content_name= param_sub['cond2_content_name']

# !!! Specific for tfr: use multitaper method
tfr_map = general_param['tfr_map_dict'][param_sub['tfr_method']]
freq_range_name = param_sub['freq_range_name']

info =get_eps_use_info(**epoch_setting) 
deriv_root = os.path.join(path_allana, individual_ana,
                          info.epoch_info)
#%% unpack the param_group
# setting statistic parameters
tail = param_group['tail']  
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

# unpack the genral_param
p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']
n_permutations = 20 if flag_test else cbpt_setting['n_permutations'] 

# !!! Specific for tfr: extract the frequency information
freqs         = tfr_map[freq_range_name]['freqs']




# setting the group result folder
if flag_test :
    path_group = os.path.join(path_allana, individual_ana,
                          "test_analysis_group_results",
                          group_configfolder,)
else:
    path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder,)
                          
os.makedirs(path_group, exist_ok=True)


#%% ----------- load individual data
df_nepoch_group = pd.DataFrame()
sub_dict = {}  
for subject in sub_list:
    subject

    bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param_sub['epoch_data_name']}_{param_sub['compare_trltypes']}_{param_sub['tfr_method']}_{roi}_{param_sub['freq_range_name']}",
        extension='.pkl',
        check=False)
    pklfile_name = bids_path.fpath

    with open(pklfile_name, 'rb') as pickle_file_load:
        loaded_data = pickle.load(pickle_file_load)
        

    if 'times' in locals():
        assert all(times == loaded_data['times']),'error: times not match'
    else:
        times = loaded_data['times']
        
    if 'sfreq' in locals():
        assert sfreq == loaded_data['sfreq'],'error: times not match'
    else:
        sfreq = loaded_data['sfreq']
        
    if 'roi_params' in locals():
        assert roi_params.keys() == loaded_data['roi_params'].keys(), "error: roi_params not match"
    else:
        roi_params = loaded_data['roi_params']

        
    # get vertices * frequency * time data
    label_dict = loaded_data ['vft_labeldict']
    sub_dict[subject] = label_dict
    assert label_dict[list(label_dict.keys())[0]]\
            [cond1_content_name].shape[-1] == len(times),\
            'error: times not match'
            
    # get epoch number 
    # !!! Specific name for tfr
    df_nepoch =loaded_data ['df_nepoch'] 
    # !!! Specific needed for tfr: run one roi at one time
    df_nepoch.insert(0,'roi',roi)

    df_nepoch.insert(0,'subject',subject)
    df_nepoch_group = pd. concat([df_nepoch_group,df_nepoch])
    
assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'
    
labels = [x for x in label_dict.keys() if 'all' not in x]
print('--------- prepare ready for data of ',config_setting)

#%% ----------- prepare data for statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
assert cond1_tidx == cond2_tidx, 'error: time window not match'
cond_tidx = cond1_tidx
# --- data with select the statistic time window
bandpower1_sub_label_f_t =  \
    np.stack([
    np.stack([
        np.nanmean(
             # (vertices, freqs, time)
            sub_dict[sub][lab][cond1_content_name][:, :, cond1_tidx], 
            axis=(0) # → average over vertices  → (freq,time,)
        )
        for lab in labels
    ], axis=0)  # → (n_labels, n_time)
    for sub in sub_dict.keys()
], axis=0)      # → (n_sub, n_labels, n_time)

bandpower2_sub_label_f_t =  \
    np.stack([
    np.stack([
        np.nanmean(
             # (vertices, freqs, time): select bandfreq
            sub_dict[sub][lab][cond2_content_name][:, :, cond2_tidx], 
            axis=(0) # → average over vertices  → (freq,time,)
        )
        for lab in labels
    ], axis=0)  # → (n_labels, n_time)
    for sub in sub_dict.keys()
], axis=0)      # → (n_sub, n_labels, n_time)

conddif_sublabelft  =np.log10(
                            bandpower1_sub_label_f_t  /\
                            bandpower2_sub_label_f_t )

#%% cluster based permutation statistic testing
clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.csv")

if ((os.path.exists(clusterfile_name)) or  (os.path.exists(blankfile_name)) ) &  (os.path.exists(cbpt_result_name)):
    print('------------','exist ' ,
            '\n',config_setting,
            '\n',datetime.now(),
    )
    if os.path.exists(clusterfile_name):
        try:
            with open(clusterfile_name, 'rb') as pickle_file_load:
                loaded_stat_data = pickle.load(pickle_file_load)
            print('------------','Successfully loaded' ,
                '\n',config_setting,
                '\n',datetime.now(),
                    )    
        except Exception :
            # Print the error for debugging purposes (optional)
            print('------------','Error loading' ,clusterfile_name,
                '\n',config_setting,
                '\n',datetime.now(),
                    )
            pass
else:
    print('------------','start ' ,
            '\n',config_setting,
            '\n',datetime.now(),
            )

    # -------- conduct permutation_cluster_1samp_test for each band each roi
    # setting for permutation_cluster_1samp_test(rather than permutation_cluster_test)
    X_observ_clulsterdim =  conddif_sublabelft
    n_observations = X_observ_clulsterdim.shape[0] 
    adjacency_labelft = mne.stats.combine_adjacency(
        np.zeros((X_observ_clulsterdim.shape[1], X_observ_clulsterdim.shape[1])),  # no adjacency between roi labels
        X_observ_clulsterdim.shape[-2],  # regular lattice adjacency for frequency
        X_observ_clulsterdim.shape[-1],  # regular lattice adjacency for times
        ) 

    
    # Compute threshold from t distribution (this is also the default)
    # If we use a two-tailed test, we need to divide alpha by 2.
    # Subtracting alpha from 1 guarantees that we get a positive threshold,
    # which MNE-Python expects for two-tailed tests.
    df = n_observations - 1  # degrees of freedom for the test
    if tail == 1:    # we want to test 1 tail a>b
        t_thresh = scipy.stats.t.ppf(1 - p_cluster_forming, df)      # one-tailed, upper critical value
    elif tail == -1: # we want to test 1 tail a<b
        t_thresh = scipy.stats.t.ppf(p_cluster_forming, df)          # one-tailed, lower critical value
    elif  tail == 0: # we want to test 2 tails a!=b
        t_thresh = scipy.stats.t.ppf(1 - p_cluster_forming / 2, df)  # two-tailed, t distribution
        
    cluster_stats= mne.stats.permutation_cluster_1samp_test(
        X = X_observ_clulsterdim, 
        threshold     = t_thresh, 
        out_type      = out_type, 
        n_permutations= n_permutations, 
        tail          = tail,    # tail is 0, the statistic is thresholded on both sides of the distribution.
        stat_fun      = None,    # None (the default), uses mne.stats.ttest_1samp_no_p which comparing the result against 0
        adjacency     = adjacency_labelft,#If None, a regular lattice adjacency is assumed, connecting each location to its neighbor(s) along the last dimension of X (or the last two dimensions if X is 2D). 
        n_jobs        = None, 
        seed          = np.random.default_rng(seed=turn_string_to_num(''.join( [group_configfolder,config_setting,individual_ana]))) , 
        max_step      =1, 
        exclude       =None, 
        step_down_p   =0, 
        t_power       =1, 
        check_disjoint=False, 
        buffer_size=1000, 
        verbose= False
                    )
    # -------------- Get Significant Clusters --------------
    # Extract cluster information
        
    T_obs, clusters, cluster_p_values, H0 = cluster_stats
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds))
    print(cluster_p_values)

    #  save data
    with open(cbpt_result_name, 'wb') as pickle_file:
        pickle.dump(cluster_stats, pickle_file)
    
    print('------------ save file ',
    '\n',cbpt_result_name,
    '\n','config ',config_setting,
    '\n',datetime.now(),
                    )
    
    # Check if there are any significant clusters
    flag_sig= len(good_cluster_inds)> 0
    if not flag_sig:
        #  save blankdata
        if  not os.path.exists(blankfile_name) :

            with open(blankfile_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
            print('------------ save blank file ' ,
                    '\n','file ',blankfile_name,
                    '\n','config ',config_setting,
                    '\n', 'roi ',roi,
                    '\n',datetime.now(),
                            )
    else :    
        # ------- save cluster info
        sig_cidx = [] 
        sig_cluster_info = {}
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            label_inds, freq_inds, time_inds = clusters[clu_idx]
            assert len(np.unique(label_inds))==1, 'error: more than one label in one cluster'
            l_inds =list(np.unique(label_inds))
            f_inds =list(np.unique(freq_inds))
            t_inds = list(np.unique(time_inds))
            
            # get index for each cluster
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_cidx = sig_cidx + l_inds[0]
            sig_freq = [freqs[x] for x in f_inds]
            
            T_sub = T_obs[np.ix_(l_inds, t_inds)]  
            # add pairwise_results to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'f_inds': f_inds,
                'l_inds': l_inds,
                'sig_time': sig_time,
                'sig_cidx': sig_cidx,
                'sig_freq': sig_freq,
                'cluster_p_values': cluster_p_values[good_cluster_inds[i_clu]],
                "cluster_T_values": T_sub,
            }
        #  save data
        with open(clusterfile_name, 'wb') as pickle_file:
            pickle.dump(sig_cluster_info, pickle_file)
        print('------------ save cluster information file ' ,
        '\n','file ',clusterfile_name,
        '\n','config ',config_setting,
        '\n', 'roi ',roi,
        '\n',datetime.now(),
                        )
                
        # ------- save table of the pvalue and time window of significant cluster  ---
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig', 'sig_cidx','pairwise_results']},
                'i_clu': k} for k, d in sig_cluster_info.items()]).iloc[::-1].reset_index(drop = True)  # Reverse the row order

        df['label_index'] = df.apply(lambda x: x['l_inds'][0], axis = 1)
        assert isinstance(df['sig_time'][0][0],float)
        df['cluster_time_start_ms']= df.apply(lambda x: int(np.round(x['sig_time'][0]*1000,0)) ,axis = 1)
        df['cluster_time_end_ms']  = df.apply(lambda x: int(np.round(x['sig_time'][-1]*1000,0)) ,axis = 1)
        df['cluster_p_values'] =  df.apply(lambda x: cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df['label'] =  df.apply(lambda x: labels[x['label_index']].replace('gnw_','').replace('iit_','').replace('&','_and_').replace(' ',''), axis = 1)
        df['label_longname']  =  df.apply(lambda x: roi_longname_dict[x['label']] if  x['label']  in roi_longname_dict.keys() 
        else  x['label'] , axis = 1)
        df['freq_range_name'] = freq_range_name
        df['roi'] = roi
        

        df_table = df[['roi','freq_range_name',
                       'label','label_longname',
                       'sig_freq',
                        'cluster_time_start_ms',
                        'cluster_time_end_ms',
                        'cluster_p_values']]
        display(df_table)
        df_table.to_csv(df_sigtable_name, index=False)
        print('------------ save table of significant cluster info',
                '\n','file ',df_sigtable_name,
                '\n','config ',config_setting,
                '\n', 'roi ',roi,)
        
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig', 'sig_cidx','pairwise_results']},
                'i_clu': k} for k, d in sig_cluster_info.items()])
        
        df['label_index'] = df.apply(lambda x: x['l_inds'][0], axis = 1)
        
        df_sig = df.sort_values(by='i_clu', ascending=True).reset_index(drop=True)
        df_sig['label'] =  df_sig.apply(lambda x: labels[x['label_index']], axis = 1)
        df_sig['cluster_p_values']=  df_sig.apply(lambda x: 
            cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df['freq_range_name'] = freq_range_name
        df['roi'] = roi
        
        display(df_sig)            
        df_sig.to_csv(df_siginfo_name, index=False)
        print('------------ save df of significant cluster info',
                '\n','file ',df_siginfo_name,
                '\n','config ',config_setting,
                '\n', 'roi ',roi,)
    
   
#%%