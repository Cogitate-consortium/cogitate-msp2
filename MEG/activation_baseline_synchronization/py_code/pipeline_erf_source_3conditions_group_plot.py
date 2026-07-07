'''
Plot of the group-level source-ERF THREE-condition analysis (seen / unseen / blank).
Run first: pipeline_erf_source_3conditions.py (individual) then
pipeline_erf_source_3conditions_group.py (group ANOVA cluster test).

This script only plots (no re-stats).
Loads each subject's per-label ERF time courses and the group's pre-computed
cluster result;  
It draws the trial-count distribution and, per significant cluster / label, the group ERF traces for the
three conditions (plot_erf_reduce_multi_and_save) plus combined subplots.

Config folder: /config_files/pipeline_erf_source_3conditions_group_0_1500/
    T_mean_V_rms_subsamp_0_1500_PFC.json
    T_mean_V_rms_subsamp_0_1500_POS.json
'''

#%% import
from itertools import chain
from _help_module import (
    plot_and_save_trial_distribution_multiple,
    plot_erf_reduce_multi_and_save,
    )

from bayes_factor_fun import bayes_ttest
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
                                 funrms,                             
                                 colordict
                                 )

from _help_module import (merge_sig_windows,)
                              
import numpy as np
import pandas as pd

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
#%% setting the reduce of labels for plot
reduce ='mean'

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

# setting individual result path
cond1_content_name = param_sub['cond1_content_name']
cond2_content_name = param_sub['cond2_content_name']
cond3_content_name = param_sub['cond3_content_name']

info =get_eps_use_info(**epoch_setting)
deriv_root = os.path.join(path_allana, 
                          individual_ana,
                            info.epoch_info)

# setting the group result folder
if flag_test :
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "test_analysis_group_results_plot",
                            group_configfolder ,
                            f"{epoch_subsampling_type}_{epoch_avg_type}_label{reduce}" )
else:
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "analysis_group_results_plot",
                            group_configfolder,
                            f"{epoch_subsampling_type}_{epoch_avg_type}_label{reduce}")
os.makedirs(path_group_plot, exist_ok=True)

pdf_path      = os.path.join(path_group_plot, 'pdf',roi)
jpg_path      = os.path.join(path_group_plot, 'jpg',roi)
plotdata_path = os.path.join(path_group_plot, 'plotdata',roi)

os.makedirs(pdf_path, exist_ok=True)
os.makedirs(jpg_path, exist_ok=True)  
os.makedirs(plotdata_path, exist_ok=True)  

path_group = os.path.join(path_allana, individual_ana,
                           "analysis_group_results",
                          group_configfolder,
                          )
#%% ----------- load individual data
df_nepoch_group = pd.DataFrame()
sub_dict = {}
for subject in  sub_list:
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
    
assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'
    
labels = [x for x in label_dict.keys() if 'all' not in x]

# constract a dict to used in following plot funcion  
sub_roi_dict = {}
for subidx,sub in enumerate(sub_list):
    sub_roi_dict[sub] ={}
    for cond in [cond1_content_name,cond2_content_name,cond3_content_name]:
        # here do not cut the statistic time window, because we need the full time for plot
        labelt = np.array([ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond]
                for lab in labels])
        sub_roi_dict[sub][cond] =labelt
print('--------- prepare ready for data of ',config_setting)
#%% save condition trial number info
group_df_name  = os.path.join(path_group, f"group_condition_trial_number_{config_setting}.csv")
df_nepoch_group.to_csv(group_df_name,index=False)

plot_and_save_trial_distribution_multiple(
    df = df_nepoch_group,
    conditions = [cond1_content_name,cond2_content_name,cond3_content_name],
    title=f"Trial_Distribution_{config_setting}_{len(sub_list)}_{sub_list_name}",
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    )
#%% ----------- prepare data for statistic testing
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
cond3_tidx = [idx for idx,t in enumerate(times) if (t>=cond3_timewindow[0])&(t<=cond3_timewindow[1])]
# if  (cond1_timewindow==cond2_timewindow) and (cond1_timewindow==cond3_timewindow):
#     times_use = times[cond1_tidx]
# else:
#     ValueError
# n_timepoint_stat = len(cond1_tidx)


sublabelcondt  = np.array(
[[np.array(
[   sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name],
    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name],
    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond3_content_name]])
for lab in [x for idx,x in enumerate(labels) ]]
for sub in sub_dict.keys() if len(sub_dict[sub])>0]
)

#%% plot cluster erf setting
flag_plot_every_sig_cluster = 0
flag_plot_every_sig_label = 0
flag_plot_combined_sig_cluster  = 0


flag_preset_ylim = 0
ylim_erf = (0.1,0.4)
ylim_erf_seplabel = (-0.1,0.5)
ylim_erfdif = (-0.02,0.07)
xtick_interval =0.5

cond_statistic_timewindow=cond1_timewindow

if  cond1_timewindow==cond2_timewindow:
    times_use = times[cond1_tidx]
else:
    ValueError
    
if 'baseline' in group_configfolder:
    times_use_w = [-0.9,1.5]
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
    # ---difference plot settings
    diff_color="grey",
    flag_plot_diff_separate=True,  # create separate diff plots
    # ---whether plot each participant
    flag_plot_each_participant=True,  # Changed: use True instead of 1
    # ---plot setting
    colordict=colordict,
    cond_names=[cond1_content_name, cond2_content_name, cond3_content_name],
    # ---save paths
    erfdiffdata_dir=plotdata_path,
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    data_dir=plotdata_path,
    # ---peaks
    compute_peaks=True,
    peak_mode="pos",
    peak_plot_alpha=1.0,  # peaks on main ERF plot
    peak_plot_alpha_diff=1.0,  # peaks on difference plots
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

)
#%% BF test for unseen vs blank (averaged over time window)
cond2_sublabelt  = sublabelcondt[:,:,:,cond2_tidx][:,:,1,:]
cond3_sublabelt  = sublabelcondt[:,:,:,cond3_tidx][:,:,2,:]
cond2_subt  = funrms(cond2_sublabelt,d_mean = 1)
cond3_subt  = funrms(cond3_sublabelt,d_mean = 1)
#% Bayesian t test 
# 1 Bayesian t test  over time points
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")

bf10, pval = bayes_ttest(cond2_subt, cond3_subt, 
                       paired=True, alternative='two-sided', r=0.707, return_pval=True)
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
    cond2_subt.mean(axis=1), 
    cond3_subt.mean(axis=1), 
    paired=True, alternative='two-sided', r=0.707, return_pval=True)
if isinstance(bf10_avg, str):
    bf10_avg = float(bf10_avg)
bf01_avg = 1 /bf10_avg

print("bf01_avg ", bf01_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({"bf10":bf10,
                 "bf01":bf01,
                 "pval":pval,
                 "bf10_avg": bf10_avg,
                 "bf01_avg": bf01_avg,
                 "pval_avg": pval_avg}, pickle_file)
# save csv  
timepoint_df = pd.DataFrame({
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
assert os.path.exists(cbpt_result_name), 'error: not exist cbpt_result_name'
if  os.path.exists(blankfile_name) :
    #% ------- plot group avg as no significant cluster found ---
    sig_info =( 
                f"{config_setting}"
                                    )
    # inside your for row_idx, row in df_sig.iterrows():
    pic_name_erf     = f"Nosig avgERF {sig_info}"
    # cluster-specific cache stem
    erfdiffdata_stem = (
        f"Nosig avgERFdiff {config_setting}"
    ).replace(' ','_')


    results = plot_erf_reduce_multi_and_save(
        sig_chans_use=labels,  # "channels" to use
        title=pic_name_erf,  
        
        # Difference pairs (which conditions to compare)
        diff_pairs=[[0, 2], [1, 2]],  # compare cond1-cond3 and cond2-cond3
        erfdiffdata_stem_prefix=erfdiffdata_stem,  

        # Windows
        window_list=None,  # No significance windows for hatching
        peak_windows=None,  # No peak_windows for hatching

        # Unpack common settings
        **common_kwargs
    )

    # save data
    plotdata_savefile_name = os.path.join(
        plotdata_path,
        f"Plotdata Nosig avg {reduce} {sig_info}.pkl").replace(' ','_')
    with open(plotdata_savefile_name, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)

    
elif os.path.exists(clusterfile_name) :
    print('------------','start plot group result of ' ,
        '\n',config_setting,
        '\n',datetime.now(),)
    #% ----------- load cluster based permutation statistic testing result
    with open(cbpt_result_name, 'rb') as pickle_file_load:
        cbpt_result = pickle.load(pickle_file_load)
    # Extract cluster information
    T_obs, clusters, cluster_p_values, H0 = cbpt_result
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
        
    #% ------- Step 1. Plot each Significant Cluster  ---
    if flag_plot_every_sig_cluster:
        df_sig = pd.read_csv(df_siginfo_name)
        for row_idx, row in df_sig.iterrows():
            print('------------ plotting single sig cluster \n',row_idx+1,'/',len(df_sig),
                    '\np=',row['cluster_p_values'],
                    )

            sig_cluster_times = [float(x) for x in row['sig_time'].replace('[','').replace(']','').split(', ')]
            sig_cluster_labels = [x.replace("'",'').replace('"','') 
                                    for x in  
                                    row['label'].replace('[','').replace(']','').split(', ')]
            window_list = [[sig_cluster_times[0],sig_cluster_times[-1]]]
            assert len(sig_cluster_labels)==1,'error sig_cluster_labels should be only one'
            
            sig_info =( 
                        f"{config_setting} "
                        f"{row_idx+1}of{len(df_sig)}cluster "
                        f"{str(int(float(sig_cluster_times[0])*1000))} {str(int(float(sig_cluster_times[-1])*1000))}"
                                            )
            # inside your for row_idx, row in df_sig.iterrows():
            pic_name_erf     = f"SigClusERF {sig_info}"

            # cluster-specific cache stem
            erfdiffdata_stem = (
                f"SigClusERFdiff {config_setting} {row_idx+1}of{len(df_sig)}cluster"
            ).replace(' ','_')

            results_cluster = plot_erf_reduce_multi_and_save(
                # plot index
                sig_chans_use=sig_cluster_labels,  # "channels" to use
                # Windows
                window_list=window_list,  
                peak_windows=window_list,  

                title=pic_name_erf,  
                
                # Difference pairs (which conditions to compare)
                diff_pairs=[[0, 2], [1, 2]],  # compare cond1-cond3 and cond2-cond3
                erfdiffdata_stem_prefix=erfdiffdata_stem,  

                # Unpack common settings
                **common_kwargs
            )

            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path,
                f"Plotdata SigCluster {reduce} {sig_info}.pkl").replace(' ','_')
            with open(plotdata_savefile_name, 'wb') as pickle_file:
                pickle.dump(results_cluster, pickle_file)
            

        #% ------- Step 2. Plot each Significant label (combined all time points) ---
        if flag_plot_every_sig_label:
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
                pic_name_erf     = f"SigLabelERF {sig_label_info}"

                # cluster-specific cache stem
                erfdiffdata_stem = (
                    f"SigLabelERFdiff {config_setting} {labeli+1}of{len(df_sig_label)}"
                ).replace(' ','_')


                results_label = plot_erf_reduce_multi_and_save(
                # plot index
                sig_chans_use=sig_label,  # "channels" to use
                # Windows
                window_list=window_label_list,  
                peak_windows=window_label_list,  

                title=pic_name_erf,  
                
                # Difference pairs (which conditions to compare)
                diff_pairs=[[0, 2], [1, 2]],  # compare cond1-cond3 and cond2-cond3
                erfdiffdata_stem_prefix=erfdiffdata_stem,  

                # Unpack common settings
                **common_kwargs )
           

               # save data
                plotdata_savefile_name = os.path.join(
                    plotdata_path,
                    f"Plotdata SigLabel {sig_label_info}.pkl").replace(' ','_')
                with open(plotdata_savefile_name, 'wb') as pickle_file:
                    pickle.dump(results_label, pickle_file)
                 
                
    #% ----- 4 flag_plot_combined_sig_cluster: for each chan_type, plot combined channels with combined time windows
    if flag_plot_combined_sig_cluster:
        print('------------ plotting combined sig cluster ' ,
                ' config=',config_setting,
                ' roi=',roi,
                        )
        
        df_sig = pd.read_csv(df_siginfo_name)
        df_sig_label = df_sig.groupby('label').agg(list).reset_index()
        df_sig_roi = df_sig.groupby('roi').agg(list).reset_index()
        
        sig_roi_times_list = [[ float(x)    for x in tw.replace('[','').replace(']','').split(', ')]
                            for tw  in df_sig_roi['sig_time'][0]]
              
        sig_roi_labels = list(set(
            chain.from_iterable([ list(set(lb)) 
            for lb in df_sig_roi['label']])))          
        
        # get unique sig time points across labels
        sig_timepoints = sorted(# sort will return a list
            list(                  
                set(               # remove duplicates
                    chain.from_iterable(
                        sig_roi_times_list  # lists of times
                    )
                )
            )
        )
        sig_tw_combined = merge_sig_windows(sig_timepoints =sig_timepoints, 
                                            times_use =times_use)

        sig_roi_info =( 
                        f"{config_setting} "
                        f"total{len(df_sig)}clusters "
                        f"{len(labels)}labels"
                            )
        pic_name_erf     = f"SigRoiERF {sig_roi_info}"

        # cluster-specific cache stem
        erfdiffdata_stem = (
                    f"SigROIERFdiff {config_setting}"
                ).replace(' ','_')

        results_roi = plot_erf_reduce_multi_and_save(
                # plot index
                sig_chans_use=sig_roi_labels,  # "channels" to use
                # Windows
                window_list=sig_tw_combined,  
                peak_windows=sig_tw_combined,  

                title=pic_name_erf,  
                
                # Difference pairs (which conditions to compare)
                diff_pairs=[[0, 2], [1, 2]],  # compare cond1-cond3 and cond2-cond3
                erfdiffdata_stem_prefix=erfdiffdata_stem,  

                # Unpack common settings
                **common_kwargs )
        
        # save data
        plotdata_savefile_name = os.path.join(
            plotdata_path, 
            f"Plotdata SigRoi {sig_roi_info}.pkl").replace(' ','_') 
        with open(plotdata_savefile_name, 'wb') as pickle_file:
            pickle.dump(results_roi, pickle_file)
else:
    raise ValueError('no cluster or blank file exist')

print('------------ finish ',
        '\n','config ',config_setting,
        '\n', 'roi ',roi,)

#%%