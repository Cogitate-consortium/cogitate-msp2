'''
Group-level control analysis: sensor-level ERF, comparing cond1 vs cond2.

Run individual analysis first: pipeline_erf_sensor.py

Configure file for group analysis: 
config_files/pipeline_erf_sensor_group_0_1000/T_mean_subsamp_grad_0_1000.json

Load the pre-saved individual channel x time condition averages (ct_dict,
selected by epoch_avg_type / epoch_subsampling_type) for the picked sensor
type (grad/mag), then run two complementary paired tests on cond1 - cond2:

- Cluster-based permutation test (permutation_cluster_1samp_test) on the
  channel x time difference, using sensor spatial adjacency -> spatio-temporal
  clusters (keeps channel topology).
- Bayesian t-test after collapsing channels to a single RMS time course
  (funrms), computed per time point and averaged over the test time window.
'''

#%% import
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime

# for statistic testing
import scipy

# for save data
import pickle

from _help_functions import (get_file_name,
                             general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 read_configfile,
                                 group_get_sublist,
                                  turn_string_to_num, # for permutation seed
                                  funrms,
                                 )
from bayes_factor_fun import bayes_ttest

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
    args.group_ana = 'pipeline_erf_sensor_group'
    args.group_configfolder = 'pipeline_erf_sensor_group_0_1000'
    args.group_configfile = base / "config_files" / "pipeline_erf_sensor_group_0_1000" / "T_mean_subsamp_grad_0_1000.json"


    args.individual_ana = 'pipeline_erf_sensor'
    args.individual_configfolder = 'pipeline_erf_sensor'
    args.individual_configfile      = base / "config_files" / "pipeline_erf_sensor" / "dAT_probe_stim_ft_bc_deci1_nocut.json"


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

script_name = os.path.basename(__file__).replace('.py','')
         
# Read the configfile file:
param_sub = read_configfile(individual_configfile)
param_group = read_configfile(group_configfile)

# get the sublist
sub_list_name, sub_list = group_get_sublist(param_group,
                                            param_sub,
                                            group_configfile,
                                            individual_configfile,
                                            )


    
path_sublist = os.path.join(path_allana, 
                            individual_ana,
                            'analysis_participant_list',)
os.makedirs(path_sublist, exist_ok=True)    
pd.DataFrame({'sub_list':sub_list}).to_csv(
    os.path.join(path_sublist,
    f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
    )


#%% unpack the param_sub
epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
epoch_data    = general_param['epoch_data_dict'][param_sub['epoch_data_name']]

# unpack the param_group
# setting load individual data type
epoch_avg_type =  param_group['epoch_avg_type']
epoch_subsampling_type=  param_group['epoch_subsampling_type']
chan_type = param_group['chan_type']
reduce_type = param_group['reduce_type']

# setting statistic parameters
tail = param_group['tail']  
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

# unpack the genral_param
p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']
n_permutations = 20 if flag_test else cbpt_setting['n_permutations'] 

# setting individual result path
cond1_content_name = param_sub['cond1_content_name']
cond2_content_name = param_sub['cond2_content_name']
info =get_eps_use_info(**epoch_setting)
deriv_root = os.path.join(path_allana, 
                              individual_ana,
                              info.epoch_info)

#%% setting the group result folder
if flag_test :
    path_group = os.path.join(path_allana, individual_ana,
                          "test_analysis_group_results",
                          group_configfolder,
                          config_setting
                          )
else:
    path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder,
                          
                          )
os.makedirs(path_group, exist_ok=True)

#%% ----------- load individual data
df_nepoch_group = pd.DataFrame()
sub_dict = {}
for subject in sub_list if flag_test else sub_list:
    subject

    bids_path = mne_bids.BIDSPath(
            root=deriv_root, 
            subject=  subject,
            session= epoch_data['exp_id'],  
            datatype='meg',  
            task=epoch_data['task_id'],
            suffix=f"{param_sub['epoch_data_name']}_{param_sub['compare_trltypes']}",
            extension='.pkl',
            check=False)
    dir_analyse = os.path.dirname( bids_path.fpath)
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
        
    if 'ch_names' in locals():
        assert ch_names == loaded_data['ch_names'],'error: times not match'
    else:
        ch_names = loaded_data['ch_names']
        
    # get channel * time data
    ct_dict = loaded_data ['ct_dict'][epoch_avg_type][epoch_subsampling_type]
    sub_dict[subject] = ct_dict
    assert ct_dict[cond1_content_name].shape[1] == len(times),'error: times not match'
    
    # get epoch number 
    nepoch_df =loaded_data ['nepoch_df']
    nepoch_df.insert(0,'subject',subject)
    df_nepoch_group = pd. concat([df_nepoch_group,nepoch_df])
print('--------- prepare ready for data of ',config_setting)

#% ----------- load example epoch
epoch_setting['debug'] = True
eps_use_example  = get_eps_use(
                            sub_list[0],
                            **epoch_data,
                            **epoch_setting,
                            )

example_avg = eps_use_example['eps_use'].average()
assert example_avg.ch_names == ch_names ,'example_avg ch_names not match with loaded data ch_names'

example_avg_use = example_avg.copy().decimate(epoch_setting['epoch_decim'])
if epoch_setting['cut_on']  not in ['None',None] :
    example_avg_use.crop(tmin=epoch_setting['cut_on'],
                        tmax=epoch_setting['cut_off'])

#% ----------- prepare data for statistic testing
example_avg_chan_type =  example_avg.copy().pick(
                            mne.pick_types(example_avg.info, 
                                            meg=chan_type)
                            )
sensor_adjacency, ch_names_picked = mne.channels.find_ch_adjacency(example_avg.info, chan_type)

ch_names_picked_idx = [ch_names.index(ch) for ch in ch_names_picked]
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]

conddif_subct = np.squeeze(np.array([
    [   sub_dict[sub][cond1_content_name][np.ix_(ch_names_picked_idx, cond1_tidx)]-
        sub_dict[sub][cond2_content_name][np.ix_(ch_names_picked_idx, cond1_tidx)]]
    for sub in sub_dict.keys()]
    ))


cond1_subct = np.squeeze(np.array([
    [   sub_dict[sub][cond1_content_name][np.ix_(ch_names_picked_idx, cond1_tidx)]]
    for sub in sub_dict.keys()]
    ))
cond2_subct = np.squeeze(np.array([
    [   sub_dict[sub][cond2_content_name][np.ix_(ch_names_picked_idx, cond2_tidx)]]
    for sub in sub_dict.keys()]
    ))

cond1_subt = funrms(cond1_subct, d_mean=1)
cond2_subt = funrms(cond2_subct, d_mean=1)
#%% Bayesian t test
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")
      

if tail == 1:    # we want to test 1 tail a>b
    print(f'----------- bayesian test {cond1_content_name} > {cond2_content_name}')
    alternative='greater'
elif tail == -1: # we want to test 1 tail a<b
    print(f'----------- bayesian test {cond1_content_name} < {cond2_content_name}')
    alternative='less'
elif  tail == 0: # we want to test 2 tails a!=b
    print(f'----------- bayesian test {cond1_content_name} != {cond2_content_name}')
    alternative='two-sided'
bf10, pval = bayes_ttest(cond1_subt, cond2_subt, 
                       paired=True, alternative=alternative, r=0.707, return_pval=True)
if isinstance(bf10, str):
    bf10 = float(bf10)
bf01 = 1 /bf10

bf01mask = np.where(bf01 > 3)[0]
pmask = np.where(pval < 0.05)[0]
if len(bf01mask)>0:
    print("bf01>3 ",    len(bf01mask), times[cond1_tidx [bf01mask[0]]], times[cond1_tidx [bf01mask[-1]]])
    print("f p<0.05 ",len(pmask),  times[cond1_tidx [pmask[0]]],  times[cond1_tidx [pmask[-1]]])
else:
    print("no bf01>3 ",bf01)
    print("no p<0.05 ")

# 2 average over time points to get one bayes factor for the intersted time window
bf10_avg, pval_avg = bayes_ttest(
    cond1_subt.mean(axis=1), 
    cond2_subt.mean(axis=1), 
    paired=True, alternative=alternative, r=0.707, return_pval=True)
if isinstance(bf10_avg, str):
    bf10_avg = float(bf10_avg)
bf01_avg = 1 /bf10_avg

print("bf01_avg ", bf01_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({
        "tail":tail,
        "alternative":alternative,
        "cond1_content_name":cond1_content_name,
        "cond2_content_name":cond2_content_name,
        "bf10":bf10,
                 "bf01":bf01,
                 "pval":pval,
                 "bf10_avg": bf10_avg,
                 "bf01_avg": bf01_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
        "tail":tail,
        "alternative":alternative,
        "cond1_content_name":cond1_content_name,
        "cond2_content_name":cond2_content_name,
    'time': times[cond1_tidx],
    'bf10': bf10,
    'bf01': bf01,
    'pval': pval,
    'bf01_over_3': bf01 > 3,
    'bf01_over_10': bf01 > 10,
    'p_05': pval < 0.05,
    'bf10_avg': bf10_avg,      # Same value repeated for all rows
    'bf01_avg': bf01_avg,      # Same value repeated for all rows
    'pval_avg': pval_avg,  # Same value repeated for all rows
})

csv_name = os.path.join(path_group, 'csv',f"bf_{config_setting}.csv")
os.makedirs(os.path.dirname(csv_name), exist_ok=True)
timepoint_df.to_csv(csv_name, index=False)
print('------------ save bayes factor file ' ,
        '\n','file ',bf_name,
        '\n','csv file ',csv_name,
        '\n','config ',config_setting,
        '\n',datetime.now(),
                        )

 
#%% get cluster based permutation statistic testing
clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.dvs")

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

    #% ----------- cluster based permutation statistic testing
    # setting for permutation_cluster_1samp_test(rather than permutation_cluster_test)
    X_observ_clulsterdim =  conddif_subct
    n_observations = X_observ_clulsterdim.shape[0] 
    
    adjacency_ct = mne.stats.combine_adjacency(
        sensor_adjacency,  
        len(times[cond1_tidx]), ) # regular lattice adjacency for times


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
        adjacency     = adjacency_ct, #If None, a regular lattice adjacency is assumed, connecting each location to its neighbor(s) along the last dimension of X (or the last two dimensions if X is 2D). 
        n_jobs        = None, 
        seed          = np.random.default_rng(seed=turn_string_to_num(config_setting)) , 
        max_step      =1, 
        exclude       =None, 
        step_down_p   =0, 
        t_power       =1, 
        check_disjoint=False, 
        buffer_size=1000, 
        verbose=None
                    )
    # -------------- Get Significant Clusters --------------
    # Extract cluster information
    T_obs, clusters, cluster_p_values, H0 = cluster_stats
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds))
    print(cluster_p_values)


    if not os.path.exists(cbpt_result_name) :
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
                    '\n', 'chan_type ',chan_type,
                    '\n',datetime.now(),
                            )
    else :    
        # ------- save cluster info
        sig_cidx = [] 
        sig_cluster_info = {}
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            channel_inds,time_inds, = clusters[clu_idx]
            t_inds = list(np.unique(time_inds))
            c_inds = list(np.unique(channel_inds))
            
            # get index for each cluster
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_cidx = sig_cidx + c_inds
            
            T_sub = T_obs[np.ix_(c_inds, t_inds)]  
            # add pairwise_results to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'sig_time': sig_time,
                'c_inds': c_inds,
                'sig_cidx': sig_cidx,
                'cluster_p_values': cluster_p_values[good_cluster_inds[i_clu]],
                "cluster_T_values": T_sub,
            }
        #  save data
        with open(clusterfile_name, 'wb') as pickle_file:
            pickle.dump(sig_cluster_info, pickle_file)
        print('------------ save cluster information file ' ,
        '\n','file ',clusterfile_name,
        '\n','config ',config_setting,
        '\n', 'chan_type ',chan_type,
        '\n',datetime.now(),
                        )
        # ------- save table of the pvalue and time window of significant cluster  ---
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig', 'sig_cidx','pairwise_results']},
                'i_clu': k} for k, d in sig_cluster_info.items()]).iloc[::-1].reset_index(drop = True)  # Reverse the row order
        df['chan_type'] = chan_type
        assert isinstance(df['sig_time'][0][0],float)
        df['cluster_time_start_ms']= df.apply(lambda x: int(np.round(x['sig_time'][0]*1000,0)) ,axis = 1)
        df['cluster_time_end_ms']  = df.apply(lambda x: int(np.round(x['sig_time'][-1]*1000,0)) ,axis = 1)
        df['cluster_p_values'] =  df.apply(lambda x: cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df['chans'] =  df.apply(lambda x: 
            [ ch_names_picked[i] for i in x['c_inds']],axis = 1)
            

        df_table = df[['chan_type','chans','cluster_time_start_ms','cluster_time_end_ms','cluster_p_values']]
        df_table.to_csv(df_sigtable_name, index=False)
        print('------------ save table of significant cluster info',
                '\n','file ',df_sigtable_name,
                '\n','config ',config_setting,
                '\n', 'chan_type ',chan_type,)
        
        df_sig = df.sort_values("chan_type")
        df_sig.to_csv(df_siginfo_name, index=False)
        print('------------ save df of significant cluster info',
                '\n','file ',df_siginfo_name,
                '\n','config ',config_setting,
                '\n', 'chan_type ',chan_type,)
    

                    
# %%
