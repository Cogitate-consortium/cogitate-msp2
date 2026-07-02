'''
Plot of the group-level band TFR analysis (pipeline_tfr_VertROI_group_band.py):
band-averaged log power time courses per condition and their difference, for one
vertex-ROI, with significant clusters highlighted.

Run first: pipeline_tfr_VertROI.py (individual) then
pipeline_tfr_VertROI_group_band.py (group band CBPT + Bayesian).

Reloads each subject's per-label (vertices, freqs, times) power, averages over the
band frequencies and over vertices -> per-label (time,) for each condition, then
reads the cluster results (cbpt_* / cbpt_clusterDf_*) and produces:
  - trial-count distribution,
  - per significant cluster, per significant label, all labels in one figure, and
    the combined-cluster ROI plot (plot_erf_erfdiff / subplot_erf_and_diff_...),
saved as jpg/pdf/plotdata under analysis_group_results_plot/<group_configfolder>/.

Only plots; all statistics come from pipeline_tfr_VertROI_group_band.py.

Config folder: /config_files/pipeline_tfr_VertROI_group_band_baseline/
    higher_gamma_pre500_0.json
    lower_alpha_pre500_0.json
Paired with an individual config /config_files/pipeline_tfr_VertROI/*.json.
'''
#%% import
from itertools import chain
from _help_module import (
    plot_and_save_trial_distribution_multiple,
    subplot_erf_and_diff_reduce_and_save)


from _help_module import (plot_erf_erfdiff)

from _help_functions import (general_param,
                             get_file_name,
                             group_get_sublist,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
                                 roi_longname_dict,# special for source analysis
                                 colordict
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
    args.group_ana = 'pipeline_tfr_VertROI_group_band'
    args.group_configfolder = 'pipeline_tfr_VertROI_group_baseline'
    args.group_configfile = base / "config_files" / "pipeline_tfr_VertROI_group_baseline" / "higher_gamma_pre500_0.json"

    
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
reduce ='mean'

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

# !!! Specific for tfr band : extract the band name from group config
band_name = param_group['band_name']
# get band frequency index
band_range = tfr_map[freq_range_name]['include_bands'][band_name]
band_fidx = [idx for idx,x in enumerate( tfr_map[freq_range_name]['freqs']) 
                    if (x>=band_range[0]) and (x<=band_range[-1])]

path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder,)
# setting the group result folder
if flag_test :
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "test_analysis_group_results_plot",
                            group_configfolder 
                            )
else:
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "analysis_group_results_plot",
                            group_configfolder)
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
    
# get the label names
labels = [x for x in label_dict.keys() if 'all' not in x]


# constract a dict to used in following plot (without use statistic time window)  
sub_roi_dict = {}
for subidx,sub in   enumerate(sub_list):
    sub_roi_dict[sub] ={}
    for cond in [cond1_content_name,cond2_content_name]:
        # here do not cut the statistic time window, because we need the full time for plot
        labelt = np.stack([
                    np.nanmean(
                        # (vertices, freqs, time): select bandfreq
                        sub_dict[sub][lab][cond][:, band_fidx, :], 
                        axis=(0, 1) # → average over vertices & freq → (time,)
                    )
                    for lab in labels
                ], axis=0)  # → (n_labels, n_time)
        sub_roi_dict[sub][cond] =labelt
print('--------- prepare ready for data of ',config_setting,datetime.now())

#%% save condition trial number info
group_df_name  = os.path.join(path_group, f"group_condition_trial_number_{config_setting}.csv")
df_nepoch_group.to_csv(group_df_name,index=False)

plot_and_save_trial_distribution_multiple(
    df = df_nepoch_group,
    conditions = [cond1_content_name,cond2_content_name],
    title=f"Trial_Distribution_{config_setting}_{len(sub_list)}_{sub_list_name}",
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    )
#%% ----------- prepare data for plot statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]


#%% plot cluster erf setting
flag_plot_every_sig_cluster = 0
flag_plot_every_sig_cluster_topo = 0
flag_plot_every_sig_label = 0
flag_plot_combined_sig_cluster  = 0
flag_plot_sig_label_subplot = 0

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
    times_use_w = [-0.8,1.2]
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
    cond1_content_name=cond1_content_name,
    cond2_content_name=cond2_content_name,
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
    diff_color="grey")




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
    sig_info =( 
                f"{config_setting}"
                                    )
    # inside your for row_idx, row in df_sig.iterrows():
    pic_name_erf     = f"Nosig avgBandLogpower {sig_info}"
    pic_name_erfdiff = f"Nosig avgBandLogpower_diff {sig_info}"

    # cluster-specific cache stem
    erfdiffdata_stem = (
        f"Nosig avgBandLogpowerdiff {config_setting}"
    ).replace(' ','_')

    erf_out, erfdiff_out = plot_erf_erfdiff(
        # plot index
        sig_chans_use=labels,
        # plot window
        window_list=None,
        peak_windows_erf=None,
        peak_windows_erfdiff=None,

        # plot name
        pic_name_erf=pic_name_erf,
        pic_name_erfdiff=pic_name_erfdiff,
        erfdiffdata_stem=erfdiffdata_stem,
        **common_kwargs
    )
    # save data
    plotdata_savefile_name = os.path.join(
        plotdata_path,
        f"Plotdata Nosig avg {reduce} {sig_info}.pkl").replace(' ','_')
    with open(plotdata_savefile_name, 'wb') as pickle_file:
        pickle.dump({
            "erf_out":erf_out,
            "erfdiff_out":erfdiff_out,
            }, pickle_file)

    
elif os.path.exists(clusterfile_name) :
    print('------------','start plot group result of ' ,
        '\n',config_setting,
        '\n',datetime.now(),)
    #% ----------- load cluster based permutation statistic testing result
    with open(cbpt_result_name, 'rb') as pickle_file_load:
        cbpt_result = pickle.load(pickle_file_load)
    
    # -------------- Get Significant Clusters --------------
    # Extract cluster information
    T_obs, clusters, cluster_p_values, H0 = cbpt_result
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds))
    print(cluster_p_values)


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
        # add pairwise_results to sig_cluster_info
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
    df['band_name'] = band_name
    

    df_table = df[['roi','band_name',
        'label','label_longname',
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
    df['roi'] = roi
    df['band_name'] = band_name
    
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
                        f"{roi} label{sig_cluster_labels[0]} {config_setting} "
                        f"{row_idx+1}of{len(df_sig)}cluster "
                        f"{len(sig_cluster_labels)}of{len(labels)}label "
                        f"{str(int(float(sig_cluster_times[0])*1000))} {str(int(float(sig_cluster_times[-1])*1000))}"
                                            )
            # inside your for row_idx, row in df_sig.iterrows():
            pic_name_erf     = f"SigClusBandLogpower {sig_info}"
            pic_name_erfdiff = f"SigClusBandLogpowerdiff {sig_info}"

            # cluster-specific cache stem
            erfdiffdata_stem = (
                f"SigClusBandLogpowerdiff {config_setting} {row_idx+1}of{len(df_sig)}cluster"
            ).replace(' ','_')

            erf_out, erfdiff_out = plot_erf_erfdiff(
                # plot index
                sig_chans_use=sig_cluster_labels,
                # plot window
                window_list=window_list,
                peak_windows_erf=window_list,
                peak_windows_erfdiff=window_list,

                # plot name
                pic_name_erf=pic_name_erf,
                pic_name_erfdiff=pic_name_erfdiff,
                erfdiffdata_stem=erfdiffdata_stem,
                **common_kwargs
            )
            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path,
                f"Plotdata SigCluster {reduce} {sig_info}.pkl").replace(' ','_')
            with open(plotdata_savefile_name, 'wb') as pickle_file:
                pickle.dump({
                    "erf_out":erf_out,
                    "erfdiff_out":erfdiff_out,
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
                            f"{roi} label{label_df['label']} {config_setting} "
                            f"{labeli+1}of{len(df_sig_label)}siglabels"
                                )
            pic_name_erf     = f"SigLabelBandLogpower {sig_label_info}"
            pic_name_erfdiff = f"SigLabelBandLogpowerdiff {sig_label_info}"

            # cluster-specific cache stem
            erfdiffdata_stem = (
                f"SigLabelBandLogpowerdiff_p{labeli+1}of{len(df_sig_label)}"
            )
            erf_label_out, erfdiff_label_out = plot_erf_erfdiff(
                # plot index
                sig_chans_use=sig_label,
                # plot window
                window_list=window_label_list,
                peak_windows_erf=window_label_list,
                peak_windows_erfdiff=window_label_list,
                # plot name
                pic_name_erf=pic_name_erf,
                pic_name_erfdiff=pic_name_erfdiff,
                
                erfdiffdata_stem=erfdiffdata_stem,
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
                    "erfdiff_label_out":erfdiff_label_out,
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
            
        for plot_type , plot_what in zip(['erf','erfanddiff'],
                                        ['conds',"conds+diff"]):
            suptitle = ( 
            f"onepic {plot_type} {roi} {reduce}labels {config_setting} "
            f"total{len(df_sig)}clusters "
            f"{len(df_sig_label)}of{len(labels)}labels"
                )
            erfdiffdata_stem = (
                f"SigLabel_{plot_type}_{plot_what}_{config_setting}"
            ).replace('+','_')

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
                plot_what="conds+diff",              # "conds", "diff", or "conds+diff"
                peak_plot_alpha = 1,
                # ---- saved erfdiff name ----
                erfdiffdata_stem= erfdiffdata_stem,
                # use common kwargs
                **common_kwargs,
                                    ) 
            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path, 
                f"Plotdata AllSigLabel {plot_type} {plot_what} {suptitle}.pkl").replace(' ','_')
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
                        f"{roi} {reduce}labels {config_setting} "
                        f"total{len(df_sig)}clusters "
                        f"{len(df_sig_label)}of{len(labels)}labels"
                            )
        pic_name_erf     = f"SigRoiBandLogpower {sig_roi_info}"
        pic_name_erfdiff = f"SigRoiBandLogpowerdiff {sig_roi_info}"

        # cluster-specific cache stem
        erfdiffdata_stem =  (
            f"SigRoiBandLogpowerdiff_{plot_type}_{plot_what}_{config_setting}"
            ).replace('+','_')
        
        erf_roi_out, erfdiff_roi_out = plot_erf_erfdiff(
                # plot index
                sig_chans_use=sig_roi_labels,
                # plot window
                window_list=window_roi_list,
                peak_windows_erf=window_roi_list,
                peak_windows_erfdiff=window_roi_list,
                # plot name
                pic_name_erf=pic_name_erf,
                pic_name_erfdiff=pic_name_erfdiff,
                # ---- saved erfdiff name ----
                erfdiffdata_stem= erfdiffdata_stem,
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
                "erfdiff_roi_out":erfdiff_roi_out,
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