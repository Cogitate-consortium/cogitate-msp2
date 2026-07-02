'''
Group-level: is contralateral alpha power different between two conditions
(cond1 vs cond2, e.g. dAT-seen vs dAT-unseen), collapsed across hemifield, for the
POS ROI. Built on the per-trial alpha power saved by pipeline_contralateral_alpha.py.

Run first: pipeline_contralateral_alpha.py (individual, per subject).

For each subject / label, splits trials by hemifield (LVF/RVF) x response and, per
trial type, averages alpha power over the contralateral vs ipsilateral vertices
(LVF -> contra = right hemi; RVF -> contra = left hemi). Per condition, averages the
two locations to get the contralateral power (contra_power), selected via the config's
contralateral_calculation_method. LVF and RVF are treated as two estimates of the same
underlying contralateral effect per subject.

Statistics on the chosen measure (cond1 - cond2, over labels x time):
  - Bayesian paired t-test (bayes_ttest, tail from config, JZS r=0.707) per time
    point and on the time-window average -> bf/bf_avg, saved as bf_*.pkl and
    csv/bf_*.csv;
  - cluster-based permutation test (permutation_cluster_1samp_test, clustering over
    labels x time) -> cbpt_* pkl/csv.
Results are saved under analysis_group_results/<group_configfolder>/<method>/.

Config folder: /config_files/pipeline_contralateral_alpha_group_baseline/
    contra_power_baseline.json
  sets contralateral_calculation_method, tail, cond1/cond2_timewindow,
  cbpt_params_name, cond1/cond2, trl_types + trl_select_columns/conditions, etc.
Paired with an individual config /config_files/pipeline_contralateral_alpha/*.json.
'''

#%% import
# for statistic testing
import scipy

from _help_functions import (group_get_sublist,
                             get_file_name,
                             get_eps_use_info,
                                 read_configfile,
                                 turn_string_to_num,
                                 roi_longname_dict,
                                 get_trl_indices# specific for this analysis to extract trial index based on condition
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
    args.group_ana = 'pipeline_contralateral_alpha_group'
    args.group_configfolder = 'pipeline_contralateral_alpha_group_baseline'
    args.group_configfile = base / "config_files" / "pipeline_contralateral_alpha_group_baseline" / "contra_power_baseline.json"

    
    args.individual_ana = 'pipeline_contralateral_alpha'
    args.individual_configfolder = 'pipeline_contralateral_alpha'
    args.individual_configfile      = base / "config_files" / "pipeline_contralateral_alpha" / "dAT_probe_stim_noft_nobc_deci5_nocut_cla_POS.json"


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
    sub_list = sub_list [0:4 ]
else:
    pd.DataFrame({'sub_list':sub_list}).to_csv(
        os.path.join(path_sublist,
        f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
        )


#%% unpack the param_sub 
epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param_sub['epoch_data_name']]
roi_params = general_param['roi_params_dict']
band_map = general_param['band_map']

               
roi = param_sub['roi']
band_name = param_sub['band_name']


info =get_eps_use_info(**epoch_setting) 
deriv_root = os.path.join(path_allana, individual_ana,
                          info.epoch_info)
#%% unpack the param_group
#!! specific for this analysis: choose which measure to use
contralateral_calculation_method = param_group[ 'contralateral_calculation_method']  # 'contra_power'


# setting statistic parameters
tail = param_group['tail']  
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

# specific for this analysis
trl_types = param_group['trl_types']
trl_select_columns = param_group['trl_select_columns']
trl_select_conditions = param_group['trl_select_conditions']

# specific for this analysis
cond1 = param_group['cond1']
cond2 = param_group['cond2']
cond1_location1 = param_group['cond1_location1']
cond1_location2 = param_group['cond1_location2']
cond2_location1 = param_group['cond2_location1']
cond2_location2 = param_group['cond2_location2']

plotcond1 = param_group['plotcond1']
plotcond2 = param_group['plotcond2']
trl_plotcond1_1 = param_group['trl_plotcond1_1']
trl_plotcond1_2 = param_group['trl_plotcond1_2']
trl_plotcond2_1 = param_group['trl_plotcond2_1']
trl_plotcond2_2 = param_group['trl_plotcond2_2']


# unpack the genral_param
p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']
n_permutations = 20 if flag_test else cbpt_setting['n_permutations'] 

#%% create folder to save

if flag_test :
    path_group = os.path.join(path_allana, individual_ana,
                          "test_analysis_group_results",
                          group_configfolder,
                          contralateral_calculation_method)
else:
    path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder,
                          contralateral_calculation_method)
os.makedirs(path_group, exist_ok=True)
path_group_csv = os.path.join(path_group, 'csv')
os.makedirs(path_group_csv, exist_ok=True)
#%% ----------- load individual data
df_nepoch_group = pd.DataFrame()
sub_dict = {}  
for subject in sub_list:
    sub_dict[subject] = {}

    bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param_sub['epoch_data_name']}_contralateral_{band_name}_power_{roi}",
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
    tlvt_alphapower_by_label = loaded_data ['tlvt_alphapower_by_label']
    vidx_dict = loaded_data ['vidx_dict']
    epochs_metadata =loaded_data ['epochs_metadata']
    labels = [x for x in tlvt_alphapower_by_label.keys() if 'all' not in x]
    for label_name in labels:
        sub_dict[subject][label_name] = {}
        tlvt = tlvt_alphapower_by_label[label_name] 
        assert tlvt.shape[-1] == len(times), 'error: times not match' 

        # ---------- get the trials for each condition ----------
        assert len(epochs_metadata) == tlvt.shape[0], 'error: epochs_metadata not match epochs'

        # loop conditions to select trial for each condition
        for trl_type, trl_select_column, trl_select_condition in zip (
                                                            trl_types,
                                                            trl_select_columns,
                                                            trl_select_conditions
                                                        ):
            # ----------- select trials for the condition
            cond_tlidx = get_trl_indices(epochs_metadata, 
                                        trl_select_column, 
                                        trl_select_condition)  
                      
            if "leftVF" in trl_type:
                vidx_contralateral = [idx for idx,vidx in enumerate(vidx_dict[label_name]['both']) if vidx in vidx_dict[label_name]['right']]
                vidx_ipsilateral   = [idx for idx,vidx in enumerate(vidx_dict[label_name]['both']) if vidx in vidx_dict[label_name]['left']]
            elif "rightVF" in trl_type:
                vidx_contralateral = [idx for idx,vidx in enumerate(vidx_dict[label_name]['both']) if vidx in vidx_dict[label_name]['left']]
                vidx_ipsilateral   = [idx for idx,vidx in enumerate(vidx_dict[label_name]['both']) if vidx in vidx_dict[label_name]['right']]
            else:
                raise ValueError('error: trl_type should contain leftVF or rightVF')
            tlt_contralateral = np.mean(tlvt[cond_tlidx,:,:][:, vidx_contralateral, :], axis=1)
            tlt_ipsilateral   = np.mean(tlvt[cond_tlidx,:,:][:, vidx_ipsilateral  , :], axis=1)
            sub_dict[subject][label_name][trl_type] = {
                'tlt_contralateral': tlt_contralateral, # average over vertices
                'tlt_ipsilateral'  : tlt_ipsilateral,
            }
                
    # Calculate all counts first
    epoch_counts = {
        trl_type: len(get_trl_indices(epochs_metadata, 
                                    trl_select_column, 
                                    trl_select_condition))
        for trl_type, trl_select_column, trl_select_condition in zip(
                        trl_types,
                        trl_select_columns,
                        trl_select_conditions)
    }

    # Create DataFrame with explicit index
    df_nepoch = pd.DataFrame(epoch_counts, index=['count'])
    # Or transpose for columns as trial types:
    df_nepoch = pd.DataFrame([epoch_counts])

    # !!! Specific needed for tfr: run one roi at one time
    df_nepoch.insert(0,'roi',roi)

    df_nepoch.insert(0,'subject',subject)
    df_nepoch_group = pd. concat([df_nepoch_group,df_nepoch])
    
assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'


cond_dict ={}
for subject in sub_list:
    cond_dict[subject] = {}
    for label_name in labels:
        cond_dict[subject][label_name] = {}
        # ---------- calculate the contralateral / ipsilateral power for each condition ----------
        for cond in [cond1, cond2]:
            # for each location, extract and average the contralateral and ipsilateral alpha power
            if cond == cond1:
                cond_loc1 = cond1_location1
                cond_loc2 = cond1_location2
            elif cond == cond2:
                cond_loc1 = cond2_location1
                cond_loc2 = cond2_location2
            else:
                raise ValueError(f'error: cond should be {cond1} or {cond2}')
            
            # ---------- pool the trials of the two locations ----------
            # contra = mean over trials of (leftVF right-hemi + rightVF left-hemi)
            # ipsi   = mean over trials of (leftVF left-hemi  + rightVF right-hemi)
            # Concatenate the trial data of two locations
            tlt_contra_both = np.concatenate([
                sub_dict[subject][label_name][cond_loc1]['tlt_contralateral'],
                sub_dict[subject][label_name][cond_loc2]['tlt_contralateral']
            ], axis=0)

            tlt_ipsi_both = np.concatenate([
                sub_dict[subject][label_name][cond_loc1]['tlt_ipsilateral'],
                sub_dict[subject][label_name][cond_loc2]['tlt_ipsilateral']
            ], axis=0)

            # average across all trials
            contra_power = np.mean(tlt_contra_both, axis=0)  # Shape: (n_time,)
            ipsi_power   = np.mean(tlt_ipsi_both, axis=0)


            cond_dict[subject][label_name][cond] = {
                'contra_power': contra_power,
                'ipsi_power': ipsi_power,
            }

print('--------- prepare ready for data of ',config_setting)


#%% 
group_df_name  = os.path.join(path_group, f"group_condition_trial_number_{config_setting}.csv")
df_nepoch_group.to_csv(group_df_name,index=False)

#%% ----------- prepare data for statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
assert cond1_tidx == cond2_tidx, 'error: cond1_tidx not match cond2_tidx'




cond1_sublabelt = np.array([
    np.array([cond_dict[sub][lab][cond1][contralateral_calculation_method][cond1_tidx]
              for lab in labels])
    for sub in sub_list
])

cond2_sublabelt = np.array([
    np.array([cond_dict[sub][lab][cond2][contralateral_calculation_method][cond2_tidx]
              for lab in labels])
    for sub in sub_list
])

# difference 
conddif_sublabelt = cond1_sublabelt - cond2_sublabelt

# average over labels
cond1_subt = cond1_sublabelt.mean(axis=1)  # (n_sub, n_time)
cond2_subt = cond2_sublabelt.mean(axis=1)  # (n_sub, n_time)


#%% Bayesian t test
# 1 Bayesian t test for each time point, each frequency
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")

if tail == 1:    # we want to test 1 tail a>b
    print(f'----------- bayesian test {cond1} > {cond2}')
    alternative='greater'
elif tail == -1: # we want to test 1 tail a<b
    print(f'----------- bayesian test {cond1} < {cond2}')
    alternative='less'
elif  tail == 0: # we want to test 2 tails a!=b
    print(f'----------- bayesian test {cond1} != {cond2}')
    alternative='two-sided'
# normal Cauchy prior on the standardized effect 
# Bayes Factor for the alternative hypothesis (H₁) over the null hypothesis (H₀).
bf10, pval = bayes_ttest(cond1_subt, 
                         cond2_subt, 
                    paired=True, alternative=alternative, r=0.707, return_pval=True)
# 2 average over time points to get one bayes factor for the intersted time window
bf_avg, pval_avg = bayes_ttest(
    cond1_subt.mean(axis=1), 
    cond2_subt.mean(axis=1), 
    paired=True, alternative=alternative, r=0.707, return_pval=True)

print("bf_avg ", bf_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({
        "tail":tail,
        "alternative":alternative,
        "cond1_content_name":cond1,
        "cond2_content_name":cond2,
        "bf":bf10,
                 "pval":pval,
                 "bf_avg": bf_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
            "tail":tail,
        "alternative":alternative,
        "cond1_content_name":cond1,
        "cond2_content_name":cond2,
    'time': times[cond1_tidx],
    'bf': bf10,
    'pval': pval,
    'bf_over_3': bf10 > 3,
    'bf_over_10': bf10 > 10,
    'p_05': pval < 0.05,
    'bf_avg': bf_avg,      # Same value repeated for all rows
    'pval_avg': pval_avg,  # Same value repeated for all rows
})
# save csv  
csv_name = os.path.join(path_group_csv, f"bf_{config_setting}.csv")
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
    if (os.path.exists(cbpt_result_name)):
        with open(cbpt_result_name, 'rb') as pickle_file_load:
            cluster_stats = pickle.load(pickle_file_load)
        print('------------','Successfully loaded' ,
            '\n',config_setting,
            '\n',datetime.now(),
                )  
    else:
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
    