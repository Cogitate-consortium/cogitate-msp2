'''
Group-level source-ERF statistics comparing cond1 vs cond2 across the ROI labels
(run individual- level first: pipeline_erf_source.py output).

Loads each subject's per-label ERF time courses
(t_dict_by_roi[roi][label][epoch_avg_type][epoch_subsampling_type][cond_content_name],
e.g. T_mean_V_rms / balanced), builds the cond1 - cond2 difference over
(subject x label x time), then runs:
- a Bayesian paired t-test (per time point and time-window-averaged), and
- a cluster-based permutation test (permutation_cluster_1samp_test) over
  label x time, with labels treated as independent (no cross-label adjacency).
Saves the BF and cluster results/tables per config.

Config folders:
  /config_files/pipeline_erf_source_group_250_500/
      T_mean_V_rms_subsamp_250_500_PFC.json
      T_mean_V_rms_subsamp_250_500_POS.json
  /config_files/pipeline_erf_source_group_GNW_baseline/
      T_mean_V_rms_subsamp_baseline_PFC.json
  each sets epoch_avg_type, epoch_subsampling_type, cond1/cond2 time windows,
  tail, cbpt_params_name; cond1/cond2_content_name come from the individual config.
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
from _help_functions import (group_get_sublist,
                             get_file_name,
                                 roi_longname_dict,# special for source analysis
                             get_eps_use_info,
                                 read_configfile,
                                 turn_string_to_num,
                                 funrms,)
from _help_functions import (general_param,
                             path_allana,
                                 )

import numpy as np
import pandas as pd

import mne_bids
import mne
# for statistic testing
import scipy
from IPython.display import display                 
from bayes_factor_fun import bayes_ttest

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
    args.group_ana = 'pipeline_erf_source_group'
    args.group_configfolder = 'pipeline_erf_source_group_250_500'
    args.group_configfile = base / "config_files" / "pipeline_erf_source_group_250_500" / "T_mean_V_rms_subsamp_250_500_PFC.json"

    
    args.individual_ana = 'pipeline_erf_source'
    args.individual_configfolder = 'pipeline_erf_source'
    args.individual_configfile      = base / "config_files" / "pipeline_erf_source" / "dAT_probe_face_ft_bc_deci1_nocut.json"


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
roi_params = general_param['roi_params_dict']
cond1_content_name = param_sub['cond1_content_name']
cond2_content_name = param_sub['cond2_content_name']

# unpack the param_group
epoch_avg_type =  param_group['epoch_avg_type']
epoch_subsampling_type=  param_group['epoch_subsampling_type']
roi = param_group['roi']

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
info =get_eps_use_info(**epoch_setting)
deriv_root = os.path.join(path_allana, 
                              individual_ana,
                              info.epoch_info)

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
for subject in sub_list[0:3] if flag_test else sub_list:
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
        
    if 'roi_params' in locals():
        assert roi_params.keys() == loaded_data['roi_params'].keys(), "error: roi_params not match"
    else:
        roi_params = loaded_data['roi_params']

        
    # get channel * time data
    if len(loaded_data ['t_dict_by_roi'][roi])==0:
        continue
    label_dict = loaded_data ['t_dict_by_roi'][roi]
    sub_dict[subject] = label_dict
    assert label_dict[list(label_dict.keys())[0]]\
            [epoch_avg_type][epoch_subsampling_type]\
            [cond1_content_name].shape[0] == len(times),\
            'error: times not match'
    
    # get epoch number 
    nepoch_df =loaded_data ['nepoch_df']
    nepoch_df.insert(0,'subject',subject)
    df_nepoch_group = pd. concat([df_nepoch_group,nepoch_df])
    
if not flag_test:
    assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'
    
labels = [x for x in label_dict.keys() if 'all' not in x]
print('--------- prepare ready for data of ',config_setting)

#% ----------- prepare data for statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]

conddif_sublabelt  = \
            np.array(
            [np.array(
            [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name][cond1_tidx]-
                sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name][cond2_tidx]
            for lab in labels])
            for sub in sub_dict.keys() ]
            )
cond1_sublabelt  = \
            np.array(
            [np.array(
            [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name][cond1_tidx]
            for lab in labels])
            for sub in sub_dict.keys() ]
            )
cond2_sublabelt  = \
            np.array(   
            [np.array(
            [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name][cond2_tidx]
            for lab in labels])
            for sub in sub_dict.keys() ]
            )
cond1_subt  = funrms(cond1_sublabelt,d_mean = 1)
cond2_subt  = funrms(cond2_sublabelt,d_mean = 1)


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
    try:
        print("p<0.05 ",len(pmask),  times[cond1_tidx [pmask[0]]],  times[cond1_tidx [pmask[-1]]])
    except Exception as e:
        print(e)
        print("p<0.05 ",len(pmask))
        pass
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
    pickle.dump({"tile": tail,
                 'alternative': alternative,
                    "cond1_content_name": cond1_content_name,
                    "cond2_content_name": cond2_content_name,
                    'time': times[cond1_tidx],
                "bf10":bf10,
                 "bf01":bf01,
                 "pval":pval,
                 "bf10_avg": bf10_avg,
                 "bf01_avg": bf01_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
    "tile": tail,
    "alternative": alternative,
    "cond1_content_name": cond1_content_name,
    "cond2_content_name": cond2_content_name,
    'time': times[cond1_tidx],
    'bf10': bf10,
    'bf01': bf01,
    'pval': pval,
    'bf01_over_3': bf01 > 3,
    'bf01_over_10': bf01 > 10,
    'p_05': pval < 0.05,
    'bf10_avg': bf10_avg,      # Same value repeated for all rows
    'bf01_avg': bf01_avg,      # Same value repeated for all rows
    'bf01_avg_over_3': bf01_avg > 3,      # Same value repeated for all rows
    'bf01_avg_over_10': bf01_avg > 10,      # Same value repeated for all rows
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
            print('------------','Successfully loaded cbpt_result_name' ,
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
    X_observ_clulsterdim =  conddif_sublabelt
    n_observations = X_observ_clulsterdim.shape[0] 

    adjacency_labelt = mne.stats.combine_adjacency(
        np.zeros((len(labels), len(labels))),  # no adjacency between roi labels
        len(times[cond1_tidx]),  # regular lattice adjacency for times
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
        adjacency     = adjacency_labelt, #If None, a regular lattice adjacency is assumed, connecting each location to its neighbor(s) along the last dimension of X (or the last two dimensions if X is 2D). 
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
    if not os.path.exists(cbpt_result_name) :
        #  save data
        with open(cbpt_result_name, 'wb') as pickle_file:
            pickle.dump(cluster_stats, pickle_file)
        
        print('------------ save file ',
        '\n',cbpt_result_name,
        '\n','config ',config_setting,
        '\n',datetime.now(),
                        )

    # -------------- Get Significant Clusters --------------
    # Extract cluster information
        
    T_obs, clusters, cluster_p_values, H0 = cluster_stats
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds))
    print(cluster_p_values)

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
            label_inds,time_inds, = clusters[clu_idx]
            t_inds = list(np.unique(time_inds))
            l_inds = list(np.unique(label_inds))
            
            # get index for each cluster
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_cidx = sig_cidx + l_inds
            
            T_sub = T_obs[np.ix_(l_inds, t_inds)]  
            # add cluster to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'sig_time': sig_time,
                'l_inds': l_inds,
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
        df['roi'] = roi
        

        df_table = df[['label','label_longname',
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
        df_sig['roi'] = roi
        display(df_sig)            
        df_sig.to_csv(df_siginfo_name, index=False)
        print('------------ save df of significant cluster info',
                '\n','file ',df_siginfo_name,
                '\n','config ',config_setting,
                '\n', 'roi ',roi,)
    
        

#%%