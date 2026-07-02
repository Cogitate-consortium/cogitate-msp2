'''
Group-level statistics on the DFC results (GNW: FF category-selective GED vs PFC GED).

This script does NOT compute DFC; it loads the per-subject, per-condition DFC
(freq x window, key 'fw'; Gaussian-Copula Mutual Information over sliding time
windows) already computed by the individual subsampling script
pipeline_syn_subsampling_dfc_GED_ROI.py, then runs the group statistics.

Individual results are read from  path_allana / <individual_ana> / <epoch_info>.
--individual_ana is pipeline_syn_subsampling_dfc_GED_ROI 

The group config selects which per-subject files to load:
  input_type = 'NOrm', ROIpairs_name, cond1 / cond2 (trl_type);
  tfr_method / freq_range_name / MI_method_name come from the individual config.
On the cond1 - cond2 difference (n_sub x n_freq x n_window) it runs:
- a Bayesian paired t-test, and
- a cluster-based permutation test (permutation_cluster_1samp_test) over the
  freq x window lattice, saving significant-cluster tables/pkls.
  NOTE (dfc-specific): cluster times are reported from the window mean
  timepoints (sl_win_meantimepoints), not raw sample times.

Config folder: /config_files/pipeline_syn_power_dfc_GED_ROI_group_250_500/ (NOrm used)
    NOrm_FFface_PFCact_preferface_250_500.json
    NOrm_FFface_PFCact_preferseen_250_500.json
    NOrm_FFobje_PFCact_preferobje_250_500.json
    NOrm_FFobje_PFCact_preferseen_250_500.json
  each sets: tail, cond1/cond2 + their time windows, cbpt_params_name,
  input_type, ROIpairs_name.
'''
#%% import
from frites.conn import  define_windows
import seaborn as sns
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
                             get_eps_use_info,
                                 read_configfile,
                                 turn_string_to_num,
                                 )
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
    args.group_ana = 'pipeline_syn_power_dfc_GED_ROI_group'
    args.group_configfolder = 'pipeline_syn_power_dfc_GED_ROI_group_250_500'
    args.group_configfile = base / "config_files" / "pipeline_syn_power_dfc_GED_ROI_group_250_500" / "NOrm_FFface_PFCact_preferface_250_500.json"

    
    args.individual_ana = 'pipeline_syn_subsampling_dfc_GED_ROI'
    args.individual_configfolder = 'pipeline_syn_subsampling_dfc_GED_ROI'
    args.individual_configfile      = base / "config_files" / "pipeline_syn_subsampling_dfc_GED_ROI" / "noft_nobc_deci1_nocut_allseen_mi.json"


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
    sub_list = sub_list# [0:3]
else:
    pd.DataFrame({'sub_list':sub_list}).to_csv(
        os.path.join(path_sublist,
        f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
        )

#%% unpack the param_sub
epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
roi_params = general_param['roi_params_dict']

# !!! Specific for tfr
tfr_map = general_param['tfr_map_dict'][param_sub['tfr_method']]
freq_range_name = param_sub['freq_range_name']




#%% unpack the genral_param
# setting statistic parameters
tail = param_group['tail']  
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1 = param_group['cond1']
cond2 = param_group['cond2']
cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

#!!! group config specify the data used
input_type = param_group['input_type']  # 'NOrm' or 'rmE'
ROIpairs_name = param_group['ROIpairs_name']


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
path_group_csv = os.path.join(path_group, 'csv',)   
os.makedirs(path_group_csv, exist_ok=True) 

#%% ----------- load individual data
if "epoch_data_name" in param_sub.keys():
    epoch_data = general_param['epoch_data_dict'][param_sub['epoch_data_name']]
    epoch_data_list = [epoch_data]
    epoch_data_name_list = [param_sub['epoch_data_name']]
    exp_id = epoch_data['exp_id']
    epoch_data_name = param_sub['epoch_data_name'] 

elif 'epoch_data_name_list' in param_sub.keys():
    epoch_data_list = [general_param['epoch_data_dict'][name] for name in param_sub['epoch_data_name_list']]
    epoch_data_name_list = param_sub['epoch_data_name_list']
    epoch_data_name_list_name = param_sub['epoch_data_name_list_name']
    epoch_data_name = epoch_data_name_list_name 
    assert(len(np.unique([general_param['epoch_data_dict'][name]['exp_id'] for name in param_sub['epoch_data_name_list']])) ==1),\
        "All epoch_data in epoch_data_name_list should have the same exp_id"

    exp_id = general_param['epoch_data_dict'][param_sub['epoch_data_name_list'][0]]['exp_id']

else:
    raise ValueError('No epoch_data_name or epoch_data_name_list in configfile')




sub_dict = {}
for subi, subject in enumerate(sorted(list(set(sub_list)))):
    sub_dict[subject] = {}
    for ii, trl_type in enumerate ([cond1, cond2]):
        ############ load individual ppc result ############
        # preset path for save ppc result
        # !!!!! Specific for MI method: add MI_method_name 
        suffix=f"{epoch_data_name}_{input_type}_{param_sub['tfr_method'].replace("_","")}_{param_sub['freq_range_name']}_{param_sub['MI_method_name']}_{ROIpairs_name}_{trl_type}"
        bids_path = mne_bids.BIDSPath(
                root=deriv_root, 
                subject= subject, 
                session= exp_id,  
                datatype='meg', 
                suffix = suffix,
                extension='.pkl',
                check=False)

        pklfile_name_ppc = bids_path.fpath

        with open(pklfile_name_ppc, 'rb') as pickle_file_load:
            loaded_data = pickle.load(pickle_file_load)
        print('------------','Successfully loaded' ,subject,
            '\n',config_setting,
            '\n',datetime.now(),
                )    
        if 'times' in locals():
            assert np.all(np.array(times) == np.array(loaded_data['times'])),'error: times not match'
        else:
            times = loaded_data['times']

        #!!!! specific for dfc, not use time but time-window average
        if ('sl_win' in locals()) and ('sl_win_meantimepoints' in locals()):
            if ('sl_win' in loaded_data.keys())  and ('sl_win_meantimepoints' in loaded_data.keys()):
                assert np.all(np.array(sl_win) == np.array(loaded_data['sl_win'])),'error: sl_win not match'
                assert np.all(np.array(sl_win_meantimepoints) == np.array(loaded_data['sl_win_meantimepoints'])),'error: sl_win_meantimepoints not match'
            else:
                sl_win_load,sl_win_meantimepoints_load = define_windows(times = times, 
                                        slwin_len  =loaded_data ['param_config']['dfc_setting']['window_len'],
                                        slwin_step = loaded_data ['param_config']['dfc_setting']['step'])
                assert np.all(np.array(sl_win) == np.array(sl_win_load)),'error: sl_win not match'
                assert np.all(np.array(sl_win_meantimepoints) == np.array(sl_win_meantimepoints_load)),'error: sl_win_meantimepoints not match'
        else:
            if ('sl_win' in loaded_data.keys())  and ('sl_win_meantimepoints' in loaded_data.keys()):
                sl_win = loaded_data['sl_win']
                sl_win_meantimepoints = loaded_data['sl_win_meantimepoints']
            else:
                sl_win,sl_win_meantimepoints = define_windows(times = times, 
                                        slwin_len  =loaded_data ['param_config']['dfc_setting']['window_len'],
                                        slwin_step = loaded_data ['param_config']['dfc_setting']['step'])


        if 'freqs' in locals():
            assert np.all(np.array(freqs) == np.array(loaded_data['freqs'])),'error: freqs not match'
        else:
            freqs = loaded_data['freqs']
            
        
        # get freq * window data
        #!!!! specific for dfc, not use time but time-window average
        fw = loaded_data ['fw']
        assert fw.shape == (len(freqs), len(sl_win_meantimepoints) ), 'error: data shape not match'
        sub_dict[subject][trl_type] = fw
        
print('--------- prepare ready for data of ',config_setting)

#%% ----------- prepare data for statistic testing
# ---set the time for statistic
#!!!! specific for dfc, not use time but time-window average
cond1_tidx = [idx for idx,t in enumerate(sl_win_meantimepoints) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(sl_win_meantimepoints) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
assert cond1_tidx == cond2_tidx, 'error: time window not match'
cond_tidx = cond1_tidx

# --- data with select the statistic time window (freq,time)
cond1_subft = np.stack([sub_dict[s][cond1][:, cond_tidx] for s in sub_dict.keys()], axis=0)  # (n_sub, n_freq, n_time)
cond2_subft = np.stack([sub_dict[s][cond2][:, cond_tidx] for s in sub_dict.keys()], axis=0)  # (n_sub, n_freq, n_time)


# --- calculate the difference between conditions
# current each value is one dfc value, so we do subtract between conditions
conddiff_subft = cond1_subft - cond2_subft

#%% --------- Bayesian t test
from bayes_factor_fun import bayes_ttest
# 1 Bayesian t test for each time point, each frequency
# NOTE: two-sided here. A tail-based version (alternative derived from config 'tail':
# greater/less for tail=+/-1, two-sided for tail=0) is computed in
# pipeline_syn_group_plot_mannual_loop_sameclim_pcolormesh_logscale.py, which writes
# to the same bf_{config_setting}.pkl and thus overwrites this one.
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")


bf, pval = bayes_ttest(cond1_subft, cond2_subft,
                       paired=True, alternative='two-sided', r=0.707, return_pval=True)
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({"bf":bf,
                 "pval":pval}, pickle_file)
    
# 2 average over time points and frequency to get one bayes factor for the intersted time window
#% set up the freq band of interest
if "vert" in config_setting.lower():
    band_names = ['gamma',]
    bands_range = [(60,90),]

else:
    band_names = ['theta','beta','gamma']
    bands_range = [(4,8),(13,30),(60,90),]
bf_avg_band = {}
bf_avg_band_df = pd.DataFrame()
for band_name,band_range in zip(
    band_names,
    bands_range):
    
    assert cond1_subft.shape[1] == len(freqs), 'error: freq not match'
    assert cond1_subft.shape[2] == len(cond_tidx), 'error: time points not match'
    fidx = [idx for idx,f in enumerate(freqs) if (f>=band_range[0])&(f<=band_range[1])]


    bf_avg, pval_avg = bayes_ttest(
        cond1_subft[:, fidx, :].mean(axis=(1,2)), 
        cond2_subft[:, fidx, :].mean(axis=(1,2)), 
        paired=True, alternative='two-sided', r=0.707, return_pval=True)
    if isinstance(bf_avg, str):
        bf_avg = float(bf_avg)

    # print violin plot using sns
    sns.boxplot(data = pd.DataFrame({
        f'{cond1}': cond1_subft[:, fidx, :].mean(axis=(1,2)),
        f'{cond2}': cond2_subft[:, fidx, :].mean(axis=(1,2)),
    })).set_title(f'Bayes Factor {band_name} {bf_avg:.2f}, pval {pval_avg:.4f}')



    bf_avg_band.update({f"bf_avg_{band_name}": bf_avg,
        f"pval_avg_{band_name}": pval_avg})
    a ={"band_name": band_name,
        "bf_avg": bf_avg,
        "pval_avg": pval_avg}
    bf_avg_band_df = pd.concat([bf_avg_band_df, pd.DataFrame(a, index=[0])], ignore_index=True)

    print(f"-----{band_name} bf_avg", bf_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({"bf":bf,
                 "pval":pval,
                 "bf_avg_band": bf_avg_band}, pickle_file)
# save csv  
csv_name = os.path.join(path_group_csv, f"bf_{config_setting}.csv")
bf_avg_band_df.to_csv(csv_name, index=False)
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

    # -------- conduct permutation_cluster_1samp_test for each band each roi
    # setting for permutation_cluster_1samp_test(rather than permutation_cluster_test)
    X_observ_clulsterdim =  conddiff_subft
    n_observations = X_observ_clulsterdim.shape[0] 
    adjacency_ft = mne.stats.combine_adjacency(
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
        
    seed = turn_string_to_num(''.join( [group_configfolder,config_setting,individual_ana]))

    cluster_stats= mne.stats.permutation_cluster_1samp_test(
        X = X_observ_clulsterdim, 
        threshold     = t_thresh, 
        out_type      = out_type, 
        n_permutations= n_permutations, 
        tail          = tail,    # tail is 0, the statistic is thresholded on both sides of the distribution.
        stat_fun      = None,    # None (the default), uses mne.stats.ttest_1samp_no_p which comparing the result against 0
        adjacency     = adjacency_ft,#If None, a regular lattice adjacency is assumed, connecting each location to its neighbor(s) along the last dimension of X (or the last two dimensions if X is 2D). 
        n_jobs        = None, 
        seed          = np.random.default_rng(seed), 
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
                    '\n',datetime.now(),
                            )
    else :    
        # ------- save cluster info
        sig_cluster_info = {}
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            freq_inds, time_inds = clusters[clu_idx]
            f_inds =list(np.unique(freq_inds))
            t_inds = list(np.unique(time_inds))
            
            # get index for each cluster
            # !!!specific for dfc, not use time but time-window average
            sig_time = [sl_win_meantimepoints[cond1_tidx[x]] for x in t_inds]
            sig_freq = [freqs[x] for x in f_inds]
            
            T_sub = T_obs[np.ix_(f_inds, t_inds)]  
            # add pairwise_results to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'f_inds': f_inds,
                'sig_time': sig_time,
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
        '\n',datetime.now(),
                        )
                
        # ------- save table of the pvalue and time window of significant cluster  ---
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig','pairwise_results']},
                'i_clu': k} 
                for k, d in sig_cluster_info.items()]).iloc[::-1].reset_index(drop = True)  # Reverse the row order

        assert isinstance(df['sig_time'][0][0],float)
        df['cluster_time_start_ms']= df.apply(lambda x: int(np.round(x['sig_time'][0]*1000,0)) ,axis = 1)
        df['cluster_time_end_ms']  = df.apply(lambda x: int(np.round(x['sig_time'][-1]*1000,0)) ,axis = 1)
        df['cluster_p_values'] =  df.apply(lambda x: cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df['freq_range_name'] = freq_range_name
        

        df_table = df[['freq_range_name',
                       'sig_freq',
                        'cluster_time_start_ms',
                        'cluster_time_end_ms',
                        'cluster_p_values']]
        display(df_table)
        df_table.to_csv(df_sigtable_name, index=False)
        print('------------ save table of significant cluster info',
                '\n','file ',df_sigtable_name,
                '\n','config ',config_setting,
                )
        
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig','pairwise_results']},
                'i_clu': k} 
                for k, d in sig_cluster_info.items()]).iloc[::-1].reset_index(drop = True)  # Reverse the row order
        
        df_sig = df.sort_values(by='i_clu', ascending=True).reset_index(drop=True)
        df_sig['cluster_p_values']=  df_sig.apply(lambda x: 
            cluster_p_values[good_cluster_inds[x['i_clu']]], axis = 1)
        df_sig['freq_range_name'] = freq_range_name
        
        display(df_sig)            
        df_sig.to_csv(df_siginfo_name, index=False)
        print('------------ save df of significant cluster info',
                '\n','file ',df_siginfo_name,
                '\n','config ',config_setting,
                )