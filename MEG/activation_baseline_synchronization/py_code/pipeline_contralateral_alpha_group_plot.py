'''
Plot of the group-level contralateral alpha analysis
(pipeline_contralateral_alpha_group.py): contralateral alpha power time courses per
condition (cond1 vs cond2) and their difference, for the POS ROI, with significant
clusters highlighted.

Run first: pipeline_contralateral_alpha.py (individual) then
pipeline_contralateral_alpha_group.py (group Bayesian + CBPT).

Reloads each subject's per-trial alpha power, splits trials by hemifield (LVF/RVF) x
response, averages over contralateral vs ipsilateral vertices, and per condition
pools the two locations to get the contralateral power (contra_power, selected via
contralateral_calculation_method). Recomputes the Bayesian t-test (saved as bf_*.pkl
and csv/bf_*.csv) and reads the cluster results (cbpt_*), then produces the
trial-count distribution and the per-cluster / per-label / combined ERF-style plots.
Plots + Bayesian stats; the CBPT is computed by pipeline_contralateral_alpha_group.py.

Config folder: /config_files/pipeline_contralateral_alpha_group_baseline/
    contra_power_baseline.json
Paired with an individual config /config_files/pipeline_contralateral_alpha/*.json.
'''

#%% import
# for statistic testing
from itertools import chain

from _help_module import (
    plot_erf_and_diff_reduce_and_save,
    plot_erf_reduce_and_save,
    plot_and_save_trial_distribution_multiple,
    subplot_erf_and_diff_reduce_and_save)

from _help_functions import (general_param,
                             get_file_name,
                             group_get_sublist,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
                                 roi_longname_dict,# special for source analysis
                                 colordict,
                                 get_trl_indices
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

import numpy as np
import pandas as pd

import mne_bids

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
    args.group_ana = 'pipeline_contralateral_alpha_group_plot'
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
sub_list_name, sub_list = group_get_sublist(param_group,
                                            param_sub,
                                            group_configfile,
                                            individual_configfile,
                                            )
if flag_test:
    sub_list = sub_list[0:3]

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
path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder,
                          contralateral_calculation_method)
path_group_csv = os.path.join(path_group, 'csv',)   
os.makedirs(path_group_csv, exist_ok=True) 

if flag_test :
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "test_analysis_group_results_plot",
                            group_configfolder ,
                          contralateral_calculation_method
                            )
else:
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "analysis_group_results_plot",
                            group_configfolder,
                          contralateral_calculation_method
                            )
os.makedirs(path_group_plot, exist_ok=True)
                          
pdf_path      = os.path.join(path_group_plot, 'pdf')
jpg_path      = os.path.join(path_group_plot, 'jpg')
plotdata_path = os.path.join(path_group_plot, 'plotdata')

os.makedirs(pdf_path, exist_ok=True)
os.makedirs(jpg_path, exist_ok=True)  
os.makedirs(plotdata_path, exist_ok=True)  
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
        # ---------- calculate the lateralization index for each condition ----------
        # lateralization index = (contra - ipsi)/(contra + ipsi)
        # for each condtion, calculate the lateralization index
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



            contralateral_alpha_index = {
                'contra_power': contra_power,
                'ipsi_power': ipsi_power,
            }
            cond_dict[subject][label_name][cond] = contralateral_alpha_index[contralateral_calculation_method]
# constract a dict to used in following plot funcion  
sub_roi_dict = {}
for subidx,sub in  enumerate(sub_list):
    sub_roi_dict[sub] ={}
    for cond in  [cond1, cond2]:
        # here did not cut the statistic time window, because we need the full time for plot
        labelt = np.array([ cond_dict[sub][lab][cond]
                for lab in labels])
        sub_roi_dict[sub][cond] = labelt
print('--------- prepare ready for data of ',config_setting)



#%% plot condition trial number info
group_df_name  = os.path.join(path_group, f"group_condition_trial_number_{config_setting}.csv")

plot_and_save_trial_distribution_multiple(
    df = df_nepoch_group,
    conditions = [cond1_location1,
                  cond1_location2,
                  cond2_location1,
                  cond2_location2],
    title=f"Trial_Distribution_{config_setting}_{len(sub_list)}_{sub_list_name}",
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    )
#%% prepare data for statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
assert cond1_tidx == cond2_tidx, 'error: cond1_tidx not match cond2_tidx'




cond1_sublabelt = np.array([
    np.array([cond_dict[sub][lab][cond1][cond1_tidx]
              for lab in labels])
    for sub in sub_list
])

cond2_sublabelt = np.array([
    np.array([cond_dict[sub][lab][cond2][cond2_tidx]
              for lab in labels])
    for sub in sub_list
])

# difference 
conddif_sublabelt = cond1_sublabelt - cond2_sublabelt

# average over labels
cond1_subt = cond1_sublabelt.mean(axis=1)  # (n_sub, n_time)
cond2_subt = cond2_sublabelt.mean(axis=1)  # (n_sub, n_time)

#%% ----------- plot settings
reduce = 'mean'  
flag_plot_every_sig_cluster = 1
flag_plot_every_sig_cluster_topo = 1
flag_plot_every_sig_label = 1
flag_plot_combined_sig_cluster  = 1
flag_plot_sig_label_subplot = 1

flag_preset_ylim = 0
ylim_erf = (-0.1,0.45)
ylim_erf_seplabel = (-0.1,0.5)
ylim_erfdif = (-0.1,0.1)
xtick_interval =0.5

cond_statistic_timewindow=cond1_timewindow

if  cond1_timewindow==cond2_timewindow:
    times_use = times[cond1_tidx]
else:
    ValueError
    
if 'baseline' in group_configfolder:
    times_use_w = [-0.7,1.1]
else:
    times_use_w = [-0.5,1.5]
    
xtick_interval

times_use_idx = [idx for idx,t in enumerate(times) if (t<= times_use_w[1])&(t>= times_use_w[0])]
times_use_plot = times[times_use_idx] 
linewidth_list =[2,2,4]
linestyle_list=['-' ,'-','--']

vlines           = [0] + [float(x) for x in cond1_timewindow if x!= None ]
vlines_colors    = ['k'] + ['k','k']
vlines_linestyle =  ['-'] +['--','--']
vlines_linewidth = [1] + [1,1]#%% 

# ------- plot setting
common_kwargs = dict(
    # --- plot avg setting (over labels)
    reduce=reduce,
    # ---plot data---
    sub_dict=sub_roi_dict,
    # ---plot channel---
    chan_type=roi,
    ch_names_all=labels,
    flag_scale_channel_value=False,
    # ---plot times---
    times_use_idx=times_use_idx,
    times_use=times_use_plot,
    # ---data setting---
    fs=1000/epoch_setting['epoch_decim'],
    # ---whether plot diff seperately
    flag_plot_diff_separate=True,
    # ---whether plot each participant
    flag_plot_each_participant=1,
    # ---plot setting
    colordict = colordict,
    cond1_name=cond1,
    cond2_name=cond2,
    # ---save paths
    erfdiffdata_dir=plotdata_path,
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    data_dir=plotdata_path,
    # ---peaks
    compute_peaks=True,
    peak_mode="pos",
    peak_plot_alpha_erf=1 ,   # compute & draw peaks 
    peak_plot_alpha_diff=1.0,  # compute & draw peaks
    # ---style
    flag_preset_ylim=flag_preset_ylim,
    ylim_erf=ylim_erf,
    linewidth_list=linewidth_list,
    linestyle_list=linestyle_list,
    xtick_interval=xtick_interval,
    vlines=vlines,
    vlines_colors=vlines_colors,
    vlines_linestyle=vlines_linestyle,
    vlines_linewidth=vlines_linewidth,
    diff_color="grey",
    # ---for subplot_erf_and_diff_reduce_and_save
    cond1_content_name = cond1,
    cond2_content_name = cond2,
    erfdiffdata_stem= None,
    )
#%% --------- Bayesian t test
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
if isinstance(bf10, str):
    bf10 = float(bf10)
bf01 = 1 /bf10
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
        "cond1_content_name":cond1,
        "cond2_content_name":cond2,
        "bf10":bf10,
                    "bf01":bf01,
                 "pval":pval,
                 "bf10_avg": bf10_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
            "tail":tail,
        "alternative":alternative,
        "cond1_content_name":cond1,
        "cond2_content_name":cond2,
    'time': times[cond1_tidx],
    'bf10': bf10,
    'bf01': bf01,
    'pval': pval,
    'bf01_over_3': bf01 > 3,
    'bf01_over_10': bf01 > 10,
    'p_05': pval < 0.05,
    'bf10_avg': bf10_avg,      # Same value repeated for all rows
    'bf01_avg': bf01_avg,      # Same value repeated for all rows
    'bf01_avg_over_3': bf01_avg > 3,  # Same value repeated for all rows
    'bf01_avg_over_10': bf01_avg > 10,  # Same value repeated for all rows
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

bf01over3mask = np.where(bf01 > 3)[0]
bf01over10mask = np.where(bf01 > 10)[0]
pmask = np.where(pval < 0.05)[0]

for mask, bf_info in zip([bf01over10mask, bf01over3mask, pmask],
                            ['bf01_over_10', 'bf01_over_3', 'pval05']):
    if any(mask):
        window_list=[times[cond1_tidx[mask[0]]],times[cond1_tidx[mask[-1]]]]
        print(f'------------  find {len(mask)} significant timepoints for {bf_info} ',
            '\n','config ',config_setting,
            '\n', 'roi ',roi,
            '\n', 'time window ',window_list,
            '\n',datetime.now(),
                            )

        pic_name_erf     = f"BF {config_setting} {bf_info}"
        window_list=[times[cond1_tidx[mask[0]]],times[cond1_tidx[mask[-1]]]]
        erf_out = plot_erf_reduce_and_save(
        # plot index
        sig_chans_use=labels,
        # plot window
        window_list=window_list,
        peak_windows_erf=window_list,
        peak_windows_erfdiff=window_list,

        # plot name
        title=pic_name_erf,
        **common_kwargs
    )
    else:
        print(f'------------  no significant timepoints for {bf_info} ' ,
            '\n','config ',config_setting,
            '\n', 'roi ',roi,
            '\n',datetime.now(),
                            )



#%% cluster based permutation statistic testing
clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.csv")
assert os.path.exists(cbpt_result_name), 'error: not exist cbpt_result_name'
if  os.path.exists(blankfile_name) :
    print('------------','start plot group result of no significant cluster found' ,
        '\n',config_setting,
        '\n',datetime.now(),)
    #% ------- plot group avg as no significant cluster found ---
    pic_name_erf     = f"Nosig {config_setting}"

    erf_out= plot_erf_and_diff_reduce_and_save(
        # plot index
        sig_chans_use=labels,
        # plot window
        window_list=None,
        peak_windows_erf=None,
        peak_windows_erfdiff=None,

        # plot name
        title=pic_name_erf,
        **common_kwargs
    )
    # save data
    plotdata_savefile_name = os.path.join(
        plotdata_path,
        f"Plotdata Nosig {config_setting}.pkl").replace(' ','_')
    with open(plotdata_savefile_name, 'wb') as pickle_file:
        pickle.dump({
            "erf_out":erf_out,
            }, pickle_file)

    
elif os.path.exists(clusterfile_name) :
    print('------------','start plot group result of ' ,
        '\n',config_setting,
        '\n',datetime.now(),)
    #% ----------- load cluster based permutation statistic testing result
    with open(cbpt_result_name, 'rb') as pickle_file_load:
        cluster_stats = pickle.load(pickle_file_load)
    
    # -------------- Get Significant Clusters --------------
    # Extract cluster information
    T_obs, clusters, cluster_p_values, H0 = cluster_stats
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds))
    print(cluster_p_values)

    #% ----------- load cluster information
    try:
        with open(clusterfile_name, 'rb') as pickle_file_load:
            sig_cluster_info = pickle.load(pickle_file_load)
        print('------------  load sig_cluster_info file')
    except:
        print('------------  no sig_cluster_info file, create new one')
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
            print('------------  save sig_cluster_info file',
                '\n',clusterfile_name)

            # ------- save table of significant cluster  ---
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
        
    #% ------- Step 1. Plot each Significant Cluster  ---
    if flag_plot_every_sig_cluster:
        print('------------ plotting every sig cluster ' ,
                ' ',config_setting,
                ' ',roi,
                ' ',band_name,
                ' ', datetime.now())
        df_sig = pd.read_csv(df_siginfo_name)
        for row_idx, row in df_sig.iterrows():
            print('------------ plotting single sig cluster \n',row_idx+1,'/',len(df_sig),
                    '\np=',row['cluster_p_values'],
                    )

            sig_cluster_times = [float(x) for x in row['sig_time'].replace('[','').replace(']','').split(', ')]
            sig_cluster_labels = [x.replace("'",'').replace('"','') 
                                    for x in  row['label'].replace('[','').replace(']','').split(', ')]
            window_list = [[sig_cluster_times[0],sig_cluster_times[-1]]]
            assert len(sig_cluster_labels)==1,'error sig_cluster_labels should be only one'
            
            sig_info =( 
                        f"{config_setting} "
                        f"{row_idx+1}of{len(df_sig)}cluster "
                        f"{str(int(float(sig_cluster_times[0])*1000))} {str(int(float(sig_cluster_times[-1])*1000))}"
                                            )
            pic_name_erf     = f"SigClus {reduce} {sig_info}"

            erf_out= plot_erf_reduce_and_save(
                # plot index
                sig_chans_use=sig_cluster_labels,
                # plot window
                window_list=window_list,
                peak_windows_erf=window_list,
                peak_windows_erfdiff=window_list,

                # plot name
                title=pic_name_erf,

                **common_kwargs
            )
            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path,
                f"Plotdata SigCluster {reduce} {sig_info}.pkl").replace(' ','_')
            with open(plotdata_savefile_name, 'wb') as pickle_file:
                pickle.dump({
                    "erf_out":erf_out,
                    }, pickle_file)
        

    #% ------- Step 2. Plot each Significant label (combined all time points) ---
    if flag_plot_every_sig_label:
        print('------------ plotting every sig label ' ,
                ' ',config_setting,
                ' ',roi,
                ' ',band_name,
                ' ', datetime.now())
        df_sig = pd.read_csv(df_siginfo_name)
        df_sig_label = df_sig.groupby('label').agg(list).reset_index()

        for labeli, label_df in df_sig_label.iterrows():
            print('------------ plotting single sig label \n',
                    labeli+1,'/',len(df_sig_label),
                    '\np=',label_df['cluster_p_values'],)

                    
            sig_label_times_list = [[ float(x)    for x in 
                                tw.replace('[','').replace(']','').split(', ')]
                                for tw  in label_df['sig_time']
                                ]
            sig_label = [label_df['label']]
            window_label_list = [[times[0],times[-1]] for times in sig_label_times_list]
            

            sig_label_info =( 
                            f"{config_setting} "
                            f"{labeli+1}of{len(df_sig_label)}siglabels"
                                )
            pic_name_erf     = f"SigLabel {reduce} {sig_label_info}"


            erf_label_out= plot_erf_reduce_and_save(
                # plot index
                sig_chans_use=sig_label,
                # plot window
                window_list=window_label_list,
                peak_windows_erf=window_label_list,
                peak_windows_erfdiff=window_label_list,
                # plot name
                title=pic_name_erf,

                # use common kwargs
                **common_kwargs,
            )
            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path,
                f"Plotdata SigLabel {sig_label_info}.pkl").replace(' ','_')
            with open(plotdata_savefile_name, 'wb') as pickle_file:
                pickle.dump({
                    "erf_label_out":erf_label_out,
                    }, pickle_file)
                
    #% ------- Step 3. Plot all label (combined all time points) in one plot ---
    if flag_plot_sig_label_subplot:
        print('------------ plotting sig label subplot ' ,
                ' ',config_setting,
                ' ',roi,
                ' ',band_name,
                ' ', datetime.now())
        df_sig = pd.read_csv(df_siginfo_name)
        df_sig_label = df_sig.groupby('label').agg(list).reset_index()
        
        channels_per_panel=[]
        windows_per_panel=[]
        titles_per_panel=[]
            
        for labeli, label_df in df_sig_label.iterrows():
            print('------------ plotting single sig label \n',
                    labeli+1,'/',len(df_sig_label),
                    '\np=',label_df['cluster_p_values'],)

                    
            sig_label_times_list = [[ float(x)    for x in 
                                tw.replace('[','').replace(']','').split(', ')]
                                for tw  in label_df['sig_time']
                                ]
            sig_label = [label_df['label']]
            window_label_list = [[times[0],times[-1]] for times in sig_label_times_list]
            
            label_longname = roi_longname_dict[label_df['label']] if  label_df['label']  in roi_longname_dict.keys() else  label_df['label']
            channels_per_panel.append(sig_label)
            windows_per_panel.append(window_label_list)
            titles_per_panel.append(label_longname)
            
        plot_what = 'conds'
        suptitle = ( 
            f"onepic {config_setting} {reduce}labels {plot_what} "
            f"total{len(df_sig)}clusters "
            f"{len(labels)}labels"
                ).replace('+',' ')
        out = subplot_erf_and_diff_reduce_and_save(
                suptitle= suptitle,
                channels_per_panel=channels_per_panel,
                windows_per_panel=windows_per_panel,
                titles_per_panel=titles_per_panel,
                # ---- layout ----
                ncols=2,                             # 2 columns → 2x2 grid for 3 panels
                tile_size_in=(7, 4),
                hspace = 1,
                wspace= 1,

                # ---- what to plot ----
                plot_what=plot_what,              # "conds", "diff", or "conds+diff"
                peak_plot_alpha = 1,

                # use common kwargs
                **common_kwargs,
                                    ) 
        # save data
        plotdata_savefile_name = os.path.join(
            plotdata_path, 
            f"Plotdata {suptitle}.pkl").replace(' ','_') 
        with open(plotdata_savefile_name, 'wb') as pickle_file:
            pickle.dump({
                "out":out,
                }, pickle_file)
                
    #% ----- 4 flag_plot_combined_sig_cluster: for each chan_type, plot combined channels with combined time windows
    if flag_plot_combined_sig_cluster:
        print('------------ plotting combined sig cluster ' ,
                ' ',config_setting,
                ' ',roi,
                ' ',band_name,
                ' ', datetime.now()
                )

        df_sig = pd.read_csv(df_siginfo_name)
        df_sig_label = df_sig.groupby('label').agg(list).reset_index()
        df_sig_roi = df_sig.groupby('roi').agg(list).reset_index()
        
        sig_roi_times_list = [[ float(x)    for x in tw.replace('[','').replace(']','').split(', ')]
                            for tw  in df_sig_roi['sig_time'][0]]
                            
        sig_roi_labels = list(set(
            chain.from_iterable([ list(set(lb)) 
            for lb in df_sig_roi['label']])))          
        window_roi_list = [[times[0],times[-1]] for times in sig_roi_times_list]
            

        sig_roi_info =( 
                        f"{config_setting} "
                        f"total{len(df_sig)}clusters "
                        f"{len(df_sig_label)}labels"
                            )
        pic_name_erf     = f"SigRoi {reduce} {sig_roi_info}"


        
        erf_roi_out= plot_erf_reduce_and_save(
                # plot index
                sig_chans_use=sig_roi_labels,
                # plot window
                window_list=window_roi_list,
                peak_windows_erf=window_roi_list,
                peak_windows_erfdiff=window_roi_list,
                # plot name
                title=pic_name_erf,
                ax_preset=None,
                ax_diff_preset = None,
                # use common kwargs
                **common_kwargs,
            )
        # save data
        plotdata_savefile_name = os.path.join(
            plotdata_path, 
            f"Plotdata SigRoi {sig_roi_info}.pkl").replace(' ','_') 
        with open(plotdata_savefile_name, 'wb') as pickle_file:
            pickle.dump({
                "erf_roi_out":erf_roi_out,
                }, pickle_file)
else:
    raise ValueError('no cluster or blank file exist')

print('------------ finish ',
        '\n','config ',config_setting,
        '\n', 'roi ',roi,
        '\n', 'band_name ',band_name,
        '\n', datetime.now()
       )

#%%