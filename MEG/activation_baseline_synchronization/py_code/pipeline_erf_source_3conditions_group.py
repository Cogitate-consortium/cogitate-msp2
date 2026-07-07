'''
Group-level source-ERF statistics for the THREE-condition comparison
(seen / unseen / blank, per label) 
-- individual run: pipeline_erf_source_3conditions.py.

Loads each subject's per-label ERF time courses
(t_dict_by_roi[roi][label][epoch_avg_type][epoch_subsampling_type][cond_content_name])
and, per label, runs:
- ANOVA cluster test: a repeated-measures one-way ANOVA F (f_mway_rm /
  f_threshold_mway_rm) as the cluster-forming statistic in permutation_cluster_test,
  over label x time -> significant clusters (any seen/unseen/blank difference);
- post-hoc pairwise paired t-tests (ttest_rel) within each significant cluster,
  FDR-corrected (Benjamini-Hochberg) -> which pair (seen-unseen, seen-blank, ...) differs;
- a Bayesian paired t-test between conditions (used for unseen vs blank).

Outputs (under analysis_group_results/<group_configfolder>/):
  ANOVA cluster test:
    cbpt_clusterTable_*.csv   - per-cluster p-value / time window / label (summary)   [CSV]
    cbpt_clusterDf_*.csv      - per-cluster details                                    [CSV]
    cbpt_allresult_*.pkl      - raw cluster_stats (T_obs, clusters, p_values, H0)      [PKL]
    cbpt_cluster_*_sig.pkl / _no_sig.pkl - significant / no-sig cluster info           [PKL]
  Post-hoc pairwise (seen-unseen, seen-blank; FDR-corrected):
    cbpt_posthoc_meanclusterlabel_*.csv                                                [CSV]
    (columns: comparison, t_value, p_value, p_value_bonf, p_value_fdr, significant_fdr;
     "significant" = significant_fdr == True)
  Bayesian (unseen vs blank):
    csv/bf_*_{Face/Obje}Unseen_BlankUnseen.csv - read the bf01_avg column              [CSV]
    bf_*.pkl                                                                            [PKL]
  group_condition_trial_number_*.csv - trial counts per condition                      [CSV]

Reported values are read from the T_mean files:
  - Bayesian BF01 from csv/bf_*_T_mean_V_rms_*_{PFC/POS}_{Face/Obje}Unseen_BlankUnseen.csv
    (the bf01_avg column);
  - pairwise post-hoc from cbpt_posthoc_meanclusterlabel_*_T_mean_V_rms_*.csv.

Config folder: /config_files/pipeline_erf_source_3conditions_group_0_1500/
    T_mean_V_rms_subsamp_0_1500_PFC.json
    T_mean_V_rms_subsamp_0_1500_POS.json
  each sets epoch_avg_type, epoch_subsampling_type, cond1/2/3 time windows,
  cbpt_params_name; cond1/cond2/cond3_content_name come from the individual config.
'''

#%% import
# for 3 conditions post hoc t test
from itertools import combinations
# for 3 conditions post hoc t test
from scipy.stats import ttest_rel
# for 3 conditions FDR correction (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests


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
                             get_file_name,
                             group_get_sublist,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
                                          turn_string_to_num,
                                 roi_longname_dict,# special for source analysis
                                 funrms,
                                 )
# for F test of cluster based permutation statistic testing
from mne.stats import f_mway_rm, permutation_cluster_test,f_threshold_mway_rm

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
    args.group_ana = 'pipeline_erf_source_3conditions_group'
    args.group_configfolder = 'pipeline_erf_source_3conditions_group_0_1500'
    args.group_configfile = base / "config_files" / "pipeline_erf_source_3conditions_group_0_1500" / "T_mean_V_rms_subsamp_0_1500_PFC.json"

    
    args.individual_ana = 'pipeline_erf_source_3conditions'
    args.individual_configfolder = 'pipeline_erf_source_3conditions'
    args.individual_configfile      = base / "config_files" / "pipeline_erf_source_3conditions" / "dAT_probe_face_blank_ft_bc_deci1_nocut.json"


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

#%% how to reduce the data within the cluster for post hoc pairwise comparison
cluster_label_reducers= ['mean']

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

pd.DataFrame({'sub_list':sub_list}).to_csv(
    os.path.join(path_sublist,
    f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
    )

#%% unpack the param_sub
epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param_sub['epoch_data_name']]
roi_params = general_param['roi_params_dict']

# unpack the param_group
epoch_avg_type =  param_group['epoch_avg_type']
epoch_subsampling_type=  param_group['epoch_subsampling_type']
roi = param_group['roi']


# setting statistic parameters
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']
cond3_timewindow = param_group['cond3_timewindow']

# unpack the genral_param
p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']
n_permutations = 20 if flag_test else cbpt_setting['n_permutations'] 

# setting individual result path
cond1_content_name = param_sub['cond1_content_name']
cond2_content_name = param_sub['cond2_content_name']
cond3_content_name = param_sub['cond3_content_name']
info =get_eps_use_info(**epoch_setting)
deriv_root = os.path.join(path_allana, individual_ana,
                            info.epoch_info)

# setting the group result folder
if flag_test :
    path_group = os.path.join(path_allana, individual_ana,
                         "test_analysis_group_results",
                          group_configfolder,
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

#%% ----------- prepare data for statistic testing
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
cond3_tidx = [idx for idx,t in enumerate(times) if (t>=cond3_timewindow[0])&(t<=cond3_timewindow[1])]
if  (cond1_timewindow==cond2_timewindow) and (cond1_timewindow==cond3_timewindow):
    times_use = times[cond1_tidx]
else:
    ValueError
n_timepoint_stat = len(times_use)
sublabelcondt  = np.array(
[[np.array(
[   sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name],
    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name],
    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond3_content_name]])
for lab in [x for idx,x in enumerate(labels) ]]
for sub in sub_dict.keys() if len(sub_dict[sub])>0]
)



#%% BF test for unseen vs blank (averaged over time window)
cond2_sublabelt  = sublabelcondt[:,:,:,cond2_tidx][:,:,1,:]
cond3_sublabelt  = sublabelcondt[:,:,:,cond3_tidx][:,:,2,:]
cond2_subt  = funrms(cond2_sublabelt,d_mean = 1)
cond3_subt  = funrms(cond3_sublabelt,d_mean = 1)
#% Bayesian t test 
# 1 Bayesian t test  over time points
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")
alternative='two-sided'
bf10, pval = bayes_ttest(cond2_subt, cond3_subt, 
                       paired=True,alternative=alternative , r=0.707, return_pval=True)
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
    cond2_subt.mean(axis=1), 
    cond3_subt.mean(axis=1), 
    paired=True, alternative=alternative , r=0.707, return_pval=True)
if isinstance(bf10_avg, str):
    bf10_avg = float(bf10_avg)
bf01_avg = 1 /bf10_avg

print("bf01_avg ", bf01_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({"alternative": alternative,
        "bf10":bf10,
                 "bf01":bf01,
                 "pval":pval,
                 "bf10_avg": bf10_avg,
                 "bf01_avg": bf01_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
    "alternative": alternative,
    "cond2_content_name": cond2_content_name,
    "cond3_content_name": cond3_content_name,
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

csv_name = os.path.join(path_group, 'csv',f"bf_{config_setting}_{cond2_content_name}_{cond3_content_name}.csv")
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
    if not os.path.exists(cbpt_result_name):
        #% ----------- cluster based permutation statistic testing
        subcondlabelt_stat = np.transpose(
            sublabelcondt[:,:,:,cond1_tidx], (0, 2, 1,3))  # (subjects, conditions, labels, time)

        # Compute F-statistic and threshold
        n_subjects, n_conditions, n_subrois, n_times = subcondlabelt_stat.shape
        print(n_subjects, n_conditions, n_subrois, n_times)

        # -------------- ANOVA and Cluster Permutation Test --------------
        # Perform a multi-way repeated measures ANOVA with cluster-based permutation testing
        #------------ The ANOVA is not the final analysis—it’s a mass-univariate step to compute F-values at each point --------------
        #  i.e. the ANOVA is performed on each individual subroi-time point, which requires the data to be flattened into a list of points.

        # f_mway_rm is designed for multi-way ANOVA with repeated measures. 
        # f_mway_rm expects the data to be 2D or 3D, where the last dimension is the measurements. 
        # If the data has multiple spatial or temporal dimensions, 
        # they need to be flattened into a single dimension because the function treats each measurement point independently.

        # by flattening subrois and time into a single dimension, 
        # each "measurement" in the ANOVA is a specific subroi-time point.
        #  The ANOVA is then run for each of these points independently. 
        sub_cond_labelt = subcondlabelt_stat.reshape(n_subjects, n_conditions, -1)# Reshape data to (n_subjects, n_conditions, n_subrois * n_times)
        
        # Define factor structure
        factor_levels = [n_conditions]  #  One factor with 3 levels (three conditions)
        effects = 'A'  # Analyze the main effect of the first (and only) factor
        
        # Get F-statistic for each subroi-time point
        f_vals, p_vals = f_mway_rm(sub_cond_labelt, 
                                factor_levels=factor_levels, 
                                effects=effects)
        # Threshold: Determine critical F-value for cluster formation
        p_F_test = p_cluster_forming
        F_threshold = f_threshold_mway_rm(
            n_subjects =n_subjects,
            factor_levels=factor_levels,
            effects=effects,
            pvalue=p_F_test
        )
        # -------------- Cluster Permutation Test----------
        # define F-statistic function for permutations (critical step!)
        def stat_fun(*args):
            # Stack permuted data and flatten subroi/time dimensions
            data_perm_flat = np.stack(args, axis=1).reshape(n_subjects, n_conditions, -1)
            
            # Compute F-values for permuted data
            f_vals_perm, _ = f_mway_rm(
                data_perm_flat, 
                factor_levels=factor_levels, 
                effects=effects
            )
            return f_vals_perm
        # Cluster Permutation Test: Group adjacent F-values into clusters and assess significance
        
        #  permutation_cluster_test function is designed to handle multiple groups or conditions
        #  by taking a list of arrays where each array represents a condition. 
        #  Each array should have the shape (n_observations, n_features). 
        #  The features here would be the combination of subrois and time points, hence the flattening into a 2D array.
        
        #  original data is 4D: subjects × conditions × subrois × time. 
        #  To use permutation_cluster_test, they need to restructure this into a list 
        #  where each condition is a 2D array (subjects × features). 
        #  Flattening subrois and time into a single dimension makes sense because the function treats each feature independently, 
        #  and the adjacency matrix is used to define which features are connected (like adjacent time points within the same subroi).
        
        # X should be List of 3 conditions, each shape (n_subjects, n_subrois × n_timepoints)
        # X = [cond1_data, cond2_data, cond3_data]  
        X = [sub_cond_labelt[:, i, :] for i in range(n_conditions)]  # List of 3 arrays, each (10, 750)
        

        # Define adjacency (time within subrois, no cross-subroi connections)
        # adjacency_time = mne.stats.combine_adjacency(None, n_times)  # Time adjacency
        # adjacency_subroi = np.eye(n_subrois)  # No adjacency between subrois
        # adjacency_labelt = mne.stats.combine_adjacency(adjacency_subroi, adjacency_time)
        
        adjacency_labelt = mne.stats.combine_adjacency(
            np.zeros((len(labels), len(labels))),  # no adjacency between roi labels
            len(times[cond1_tidx]),  # regular lattice adjacency for times
            ) 

        cluster_stats = permutation_cluster_test(
            X = X ,
            stat_fun = stat_fun,
            threshold = F_threshold,
            adjacency = adjacency_labelt,
            n_permutations= n_permutations,
            tail= 1,  # ANOVA uses an F-test, which is inherently one-tailed. as F-distribution is not symmetric and only has one tail that matters for significance; F-test in ANOVA checks if there's any variance among group means, without specifying direction
                n_jobs        = None, 
                seed          = np.random.default_rng(seed=turn_string_to_num(config_setting)) , 
                max_step      =1, # default 1: Maximum allowed step between adjacent points to be considered part of the same cluster.
            exclude       =None, # Exclude points already in a cluster from being part of other clusters.
            step_down_p   =0,  # Step-down p-value threshold for iterative cluster refinement (controls FWER).Use 0 (no step-down) or set to a value like 0.05 for iterative correction.
            t_power       =1,# default 1: Exponent applied to the test statistic (e.g., t_power=1 for raw t/F-values, t_power=2 for squared values).
            check_disjoint=False,# default False: Set to True for strict non-overlapping clusters. Ensure clusters are disjoint (non-overlapping).
            buffer_size=1000,  # default 1000: Number of data chunks processed at once (memory optimization). Increase for faster processing (if memory allows); decrease for large datasets.
            verbose=None )
        # -------------- save cbpt_result_name --------------
        if not os.path.exists(cbpt_result_name) :
            #  save data
            with open(cbpt_result_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
            
            print('------------ save file ',
            '\n',cbpt_result_name,
            '\n','config ',config_setting,
            '\n',datetime.now(),
                            )
    else:
        # load data
        with open(cbpt_result_name, 'rb') as pickle_file_load:
            cluster_stats = pickle.load(pickle_file_load)
        print('------------ load file ',
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
        with open(blankfile_name, 'wb') as pickle_file:
            pickle.dump(cluster_stats, pickle_file)
        print('------------ save blank file ' ,
                '\n','file ',blankfile_name,
                '\n','config ',config_setting,
                '\n', 'roi ',roi,
                '\n',datetime.now(),
                        )
    else :    
        # ------- conduct post hoc pairwise comparisons for each significant cluster
        sig_lidx = [] 
        sig_cluster_info = {}
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            # Get the cluster's indices in the FLATTENED feature space (subroi × time)

            flat_indices = clusters[clu_idx][0]# unpack cluster information, get unique indices
            # Convert to original subroi and time indices
            subroi_indices = flat_indices // n_timepoint_stat
            time_indices = flat_indices % n_timepoint_stat
            
            t_inds = list(np.unique(time_indices))
            t_inds_inraw= [cond1_tidx[x] for x in t_inds]
            l_inds = list(np.unique(subroi_indices))
            sig_label = [labels[x] for x in l_inds]
            
            # get index for each cluster
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_lidx = sig_lidx + l_inds
                        
            # --- Step 1. Extract Data from the Significant Cluster ---
            # Construct sublabelcondt_cluster
            condition_names = [cond1_content_name, cond2_content_name, cond3_content_name]
            sublabelcondt_cluster = np.array([
                [np.array([
                    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][condition_name][t_inds_inraw] 
                    for  condition_name in condition_names
                ])
                for lab in [x for idx, x in enumerate(labels) if idx in l_inds]]
                for sub in sub_dict.keys() if len(sub_dict[sub]) > 0
            ])

            pairwise_results_allreduce = {}
            # Average the data over the cluster for each all timepoint. 
            # This gives an array of shape (n_subjects, n_conditions), then transpose
            sublabelcond_cluster = np.mean(sublabelcondt_cluster, axis=-1) # Average over time within the cluster
            for cluster_label_reduce in cluster_label_reducers:
                
                if cluster_label_reduce == 'mean':
                    subcond_cluster = np.mean(sublabelcond_cluster, axis=1)  # Average over labels in the cluster
                elif cluster_label_reduce == 'rms':
                    subcond_cluster = funrms(sublabelcond_cluster, d_mean=1) # RMS over labels in the cluster
                else:
                    raise ValueError('cluster_label_reduce has to be mean or rms')
                
                # Now subcond_cluster is of shape (n_subjects, n_conditions)
                condsub_sig = np.transpose(subcond_cluster, (1, 0))  # Transpose to (n_conditions, n_subjects)

                # --- Step 2. Perform Pairwise Post Hoc Comparisons ---
                # Generate all unique pairs of condition names
                pairs = list(combinations(condition_names, 2))
                index_map = {name: idx for idx, name in enumerate(condition_names)} # Create a dictionary to map condition names to their indices

                ## Perform pairwise tests
                # Initialize storage
                
                all_p_values = []
                pairwise_results = {}
                for (name_i, name_j) in pairs:
                    i = index_map[name_i]
                    j = index_map[name_j]
                    
                    # Calculate t-test
                    t_val, p_val = ttest_rel(condsub_sig[i, :], condsub_sig[j, :])
                    
                    # Store results
                    key = f"{name_i}_{name_j}"
                    pairwise_results[key] = {
                        't_value': t_val,
                        'p_value': p_val, }
                    all_p_values.append(p_val)
                    
                # --- Step 3. Multiple Comparisons Correction ---
                # Bonferroni correction
                n_tests = len(all_p_values)
                p_values_bonf = np.minimum(np.array(all_p_values) * n_tests, 1.0)

                # False Discovery Rate (FDR) Correction using Benjamini–Hochberg
                significant_fdr, pvals_fdr, _, _ = multipletests(all_p_values, 
                                                        alpha=p_cluster_forming, 
                                                        method='fdr_bh')

                # Bonferroni results
                print(f"\ngood_cluster_inds {i_clu} Bonferroni corrected:")
                reject_bonf = p_values_bonf <= p_cluster_forming
                for idx, (name_i, name_j) in enumerate(pairs):
                    sig_status = "SIGNIFICANT" if reject_bonf[idx] else "not significant"
                    print(f"{name_i} vs {name_j}: p = {p_values_bonf[idx]:.3f} ({sig_status})")
                # print FDR results
                print(f"\ngood_cluster_inds {i_clu} FDR corrected (Benjamini-Hochberg):")
                for idx, (name_i, name_j) in enumerate(pairs):
                    sig_status = "SIGNIFICANT" if significant_fdr[idx] else "not significant"
                    print(f"{name_i} vs {name_j}: p = {pvals_fdr[idx]:.3f} ({sig_status})")
                
                # Add corrections to results dictionary
                for idx, (name_i, name_j) in enumerate(pairs):
                    key = f"{name_i}_{name_j}"
                    pairwise_results[key].update({
                        'p_value_bonf': p_values_bonf[idx],
                        'p_value_fdr': pvals_fdr[idx],
                        'significant_fdr': significant_fdr[idx]
                    })
                                        
                # Generate significance info
                fdr_sig_pairs = [f"{name_i}-{name_j}" for idx, (name_i, name_j) in enumerate(pairs) if significant_fdr[idx]]

                # Create compact summary string
                sig_info = f"{' '.join(sig_label)} FDR-BH {' '.join(fdr_sig_pairs).replace('nseen', '').replace('ace', '').replace('lank', '').replace('een', '')}"
                print(f"{cluster_label_reduce} Cluster {i_clu} significant comparisons: {sig_info}")
                
                # store Pairwise results for each cluster_label_reduce 
                pairwise_results_allreduce[cluster_label_reduce] = pairwise_results
                    
            # add pairwise_results to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'sig_time': sig_time,
                'l_inds': l_inds,
                'sig_lidx': sig_lidx,
                'sublabelcondt_cluster': sublabelcondt_cluster,
                "condsub_sig": condsub_sig,
                'pairwise_results_allreduce': pairwise_results_allreduce
            }

        #  save Cluster data
        with open(clusterfile_name, 'wb') as pickle_file:
            pickle.dump(sig_cluster_info, pickle_file)


        # ------- save the pvalue and time window of significant cluster  ---
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['sublabelcondt_cluster', 'condsub_sig', 'sig_lidx','pairwise_results']},
                'i_clu': k} for k, d in sig_cluster_info.items()]).iloc[::-1].reset_index(drop = True)  # Reverse the row order
        df['roi_idx'] = df.apply(lambda x: x['l_inds'][0], axis = 1)
        assert isinstance(df['sig_time'][0][0],float)
        df['cluster_time_start_ms']= df.apply(lambda x: int(np.round(x['sig_time'][0]*1000,0)) ,axis = 1)
        df['cluster_time_end_ms']  = df.apply(lambda x: int(np.round(x['sig_time'][-1]*1000,0)) ,axis = 1)
        df['cluster_p_values'] =  df.apply(lambda x: cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df['ROI'] =  df.apply(lambda x: labels[x['roi_idx']].replace('gnw_','').replace('iit_','').replace('&','_and_').replace(' ',''), axis = 1)
        df['ROI_longname']  =  df.apply(lambda x: roi_longname_dict[x['ROI']] if  x['ROI']  in roi_longname_dict.keys() 
        else  x['ROI'] , axis = 1)
        df_table = df[['ROI','ROI_longname',
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