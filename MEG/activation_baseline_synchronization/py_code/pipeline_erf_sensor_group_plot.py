'''
Plot of the group-level sensor-ERF control analysis (cond1 vs cond2).
Run first: pipeline_erf_sensor.py (individual) then pipeline_erf_sensor_group.py (group CBPT).

Loads the per-subject channel x time condition averages
(ct_dict[epoch_avg_type][epoch_subsampling_type], e.g. T_mean / balanced) and the
group's PRE-COMPUTED cluster result (cbpt_allresult_*.pkl); this script does NOT
re-run the statistics, it only plots.

It produces:
  - a trial-count distribution figure,
  - group-average joint plots (butterfly + topomaps) for cond1, cond2 and their
    difference (same color limits across the two conditions),
  - per significant cluster: sensor-location topo, ERF / ERFdiff traces, and
    peak-time butterfly / joint / topo plots,
  - a combined figure over all significant clusters/channels.

Config folder: /config_files/pipeline_erf_sensor_group_0_1000/
    T_mean_subsamp_grad_0_1000.json
  (paired with the group config above; the individual config gives
   cond1/cond2_content_name and compare_trltypes.)
'''

#%% import
#flattens a list of lists into a single sequence
from itertools import chain

from _help_module import (_flatten_list  ,
                            unique_in_order, # iterable to list
                            _ensure_list   )   
from _help_module import plot_and_save_trial_distribution_multiple
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

# for show time
from datetime import datetime
# for save data
import pickle

from _help_functions import (get_file_name,
                             general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 read_configfile,
                                 group_get_sublist,
                                 colordict)

from _help_module import (plot_erf_erfdiff,
                              merge_sig_windows,
                              jointplot_fullsensor,
                              butterflyplot_fullsensor,
                              save_topo_of_window)

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
    args.group_ana = 'pipeline_erf_sensor_group_plot'
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
epoch_data    = general_param['epoch_data_dict'][param_sub['epoch_data_name']]

# unpack the param_group
# setting load individual data type
epoch_avg_type =  param_group['epoch_avg_type']
epoch_subsampling_type=  param_group['epoch_subsampling_type']
chan_type = param_group['chan_type']
reduce_type = param_group['reduce_type'][::-1]  # reverse the order for plotting from more average to less average

loaddata_info = f"{epoch_avg_type} {epoch_subsampling_type}"

# setting statistic parameters
tail = param_group['tail']  
cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

# unpack the genral_param
p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']

# setting individual result path
cond1_content_name = param_sub['cond1_content_name']
cond2_content_name = param_sub['cond2_content_name']
info =get_eps_use_info(**epoch_setting)
deriv_root = os.path.join(path_allana, 
                              individual_ana,
                              info.epoch_info)

# setting the group result folder
if flag_test :
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "test_analysis_group_results_plot",
                            group_configfolder                          )
else:
    path_group_plot = os.path.join(path_allana, individual_ana,
                            "analysis_group_results_plot",
                            group_configfolder                          )
os.makedirs(path_group_plot, exist_ok=True)

path_group = os.path.join(path_allana, individual_ana,
                          "analysis_group_results",
                          group_configfolder                          )


pdf_path      = os.path.join(path_group_plot, 'pdf')
jpg_path      = os.path.join(path_group_plot, 'jpg')
plotdata_path = os.path.join(path_group_plot, 'plotdata')

os.makedirs(pdf_path, exist_ok=True)
os.makedirs(jpg_path, exist_ok=True)  
os.makedirs(plotdata_path, exist_ok=True)  

#%% ----------- load individual data for plot
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
#%% ----------- load example epoch for plot
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

example_avg_chan_type =  example_avg_use.copy().pick(
                            mne.pick_types(example_avg_use.info, 
                                            meg=chan_type)
                            )
sensor_adjacency, ch_names_picked = mne.channels.find_ch_adjacency(example_avg_use.info, chan_type)

ch_names_picked_idx = [ch_names.index(ch) for ch in ch_names_picked]
#%% ----------- prepare example_avg_chan_type

# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]


conddif_subct = np.squeeze(np.array([
    [   sub_dict[sub][cond1_content_name][np.ix_(ch_names_picked_idx, cond1_tidx)]-
        sub_dict[sub][cond2_content_name][np.ix_(ch_names_picked_idx, cond1_tidx)]]
    for sub in sub_dict.keys()]
    ))


#%% prepare to plot the group avg plot
subcondct  =np.array([
[   sub_dict[sub][cond1_content_name],
    sub_dict[sub][cond2_content_name]]
for sub in sub_dict.keys()]
)

ERFdiff_subct = np.squeeze(np.array([
[   sub_dict[sub][cond1_content_name]-
    sub_dict[sub][cond2_content_name]]
for sub in sub_dict.keys()]
))

# Assume you already have `ERFdiff_subct` shape: (n_subjects, n_channels, n_times)
# Average across subjects:
ERF_diff_avg = ERFdiff_subct.mean(axis=0)  # shape: (n_channels, n_times)
ERF_c1_avg = subcondct[:,0,:,:].mean(axis=0)
ERF_c2_avg = subcondct[:,1,:,:].mean(axis=0)

# Create an Evoked object 
evoked_diff = mne.EvokedArray(ERF_diff_avg, 
                              example_avg_use.info, 
                              tmin=times[0])
evoked_c1 = mne.EvokedArray(ERF_c1_avg, 
                              example_avg_use.info, 
                              tmin=times[0])
evoked_c2 = mne.EvokedArray(ERF_c2_avg,
                                example_avg_use.info, 
                                tmin=times[0])



#%% plot all sensors all subject average
times_plot = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Choose relevant times in seconds
# Find global min/max across the two conditions (excluding difference)
from _help_module import _scale_by_chan_type

e1, unit, factor= _scale_by_chan_type(np.array(evoked_c1.copy().pick(picks =chan_type ).data), chan_type)
e2, unit, factor= _scale_by_chan_type(np.array(evoked_c2.copy().pick(picks =chan_type ).data), chan_type)
ediff, unit, factor= _scale_by_chan_type(np.array(evoked_diff.copy().pick(picks =chan_type ).data), chan_type)
print(np.abs(e1).max(), np.abs(e2).max(), np.abs(ediff).max())


vabs = max(np.abs(e1).max(), 
           np.abs(e2).max())
for evoked,plot_info in zip( [evoked_c1, 
                              evoked_c2,
                              evoked_diff],
                  [cond1_content_name, cond2_content_name,
                   f"{cond1_content_name}S{cond2_content_name}".strip()]):
        
    title = f"joint plot {loaddata_info} {plot_info} avg {chan_type}"
    if plot_info == f"{cond1_content_name}S{cond2_content_name}".strip():
        ylim = None
    else:
        ylim=(-vabs, vabs)

        
    jointplot_fullsensor(
        evoked=evoked,
        chan_type= chan_type,
        times_plot=times_plot,
        title=title,
        jpg_dir=jpg_path,
        pdf_dir=pdf_path,
        dpi=300,
        ylim=ylim,
    )

#%% plot cluster erf setting
flag_plot_every_sig_cluster = 1
flag_plot_every_sig_cluster_topo = 1
flag_plot_combined_sig_cluster  = 1

flag_peak_of_ERFdiff = 1

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
    # ---plot data---
    sub_dict=sub_dict,
    flag_test = flag_test,
    # ---plot channel---
    chan_type=chan_type,
    ch_names_all=ch_names,
    flag_scale_channel_value=True,
    # ---plot times---
    times_use_idx=times_use_idx,
    times_use=times_use_plot,
    # ---data setting---
    fs=1000/epoch_setting['epoch_decim'],
    # ---whether plot diff seperately
    flag_plot_diff_separate=1,
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

#%% plot detected cluster
clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.dvs")
assert os.path.exists(cbpt_result_name),'cbpt_result file not exist'
if (not os.path.exists(blankfile_name)):
    print('------------','start plot group result of ' ,
        '\n',config_setting,
        '\n',datetime.now(),)

    #% ----------- load cluster based permutation statistic testing result
    with open(cbpt_result_name, 'rb') as pickle_file_load:
        cbpt_result = pickle.load(pickle_file_load)
    T_obs, clusters, cluster_p_values, H0 = cbpt_result
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    
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
    #% ------- Step 1. Plot each Significant Cluster  ---
    if flag_plot_every_sig_cluster:
        df_sig = pd.read_csv(df_siginfo_name)
        for row_idx, row in df_sig.iterrows():
            print('------------ plotting single sig cluster \n',row_idx+1,'/',len(df_sig),
                    '\np=',row['cluster_p_values'],
                    )
            sig_cluster_times = [float(x) for x in row['sig_time'].replace('[','').replace(']','').split(', ')]
            sig_cluster_chans = [x.replace("'",'').replace('"','') for x in  row['chans'].replace('[','').replace(']','').split(', ')]
            window_list = [[sig_cluster_times[0],sig_cluster_times[-1]]]
            

            
            chs_sig = [ch for ch in 
                        example_avg_chan_type.info['ch_names'] if ch in set(sig_cluster_chans)]
            chs_other = [ch for ch in 
                            example_avg_chan_type.info['ch_names'] if ch not in set(sig_cluster_chans)]
                                            
            picks = mne.pick_channels(
                ch_names = example_avg_chan_type.info['ch_names'], 
                include=chs_sig)
            example_avg_sigcluster = example_avg_chan_type.copy().pick(picks)
            sig_info =( 
                        f"{config_setting} "
                        f"{row_idx+1}of{len(df_sig)}cluster "
                        f"{len(chs_sig)}of{len(ch_names_picked)}chans "
                        f"{str(int(float(sig_cluster_times[0])*1000))} {str(int(float(sig_cluster_times[-1])*1000))}"
                                )

            if flag_plot_every_sig_cluster_topo:
                # ------- plot location of sigchan
                info_sig = example_avg_sigcluster.info
                fig = mne.viz.plot_sensors(
                    info_sig,
                    kind='topomap',
                    show_names=False
                )
                pic_name = f"SigClusTopo {sig_info}"

                fig.savefig(os.path.join(jpg_path,f"{pic_name.replace(' ','_')}.jpg"), bbox_inches='tight')# ,dpi=300
                fig.savefig(os.path.join(pdf_path,f"{pic_name.replace(' ','_')}.pdf"), bbox_inches='tight',dpi=300) # 
                
                # ------- plot erp erfdiff
                for reduce in reduce_type:
                    
                    # inside your for row_idx, row in df_sig.iterrows():
                    pic_name_erf     = f"SigClusERF {reduce} {sig_info}"
                    pic_name_erfdiff = f"SigClusERFdiff {reduce} {sig_info}"
                   

                    erf_out, erfdiff_out = plot_erf_erfdiff(
                        # avg setting 
                        reduce=reduce,
                        # plot channel
                        sig_chans_use=chs_sig,
                        # plot window
                        window_list=window_list,
                        peak_windows_erf=window_list,
                        peak_windows_erfdiff=window_list,

                        # plot name
                        pic_name_erf=pic_name_erf,
                        pic_name_erfdiff=pic_name_erfdiff,
                        erfdiffdata_stem=pic_name_erfdiff,
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
             
                    # ------- plot topo plot with peak time point
                    for con,ev in zip([cond1_content_name,cond2_content_name,'ERF Difference'],
                                [evoked_c1,evoked_c2,evoked_diff]):
                        df_peak = erfdiff_out['peaks_table']
                        
                        
                        if     flag_peak_of_ERFdiff:
                            peak_t = list(df_peak[df_peak.condition =='ERF Difference']['t_peak'])
                            peakinfo = 'peakofDiff'

                        else:
                            peak_t = list(df_peak[df_peak.condition ==con]['t_peak'])
                            peakinfo = ''

                        evoked_plot = ev.copy().crop(tmin = times_use_plot[0],tmax = times_use_plot[-1])
                        
                        title  = f"butterflyplot {con} {reduce} {sig_info}"
                        _ = butterflyplot_fullsensor(
                                    evoked= evoked_plot,
                                    highlight_window =window_list[0],
                                    chan_type = chan_type,
                                    title=title,
                                    jpg_dir=jpg_path,
                                    pdf_dir=pdf_path,
                                    dpi=300,
                                    );
                        title  = f"jointplot {con} {reduce} {sig_info}{peakinfo}"
                        check_data = jointplot_fullsensor(
                                    evoked= evoked_plot,
                                    highlight_window =window_list[0],
                                    chan_type = chan_type,
                                    times_plot=peak_t,
                                    title=title,
                                    jpg_dir=jpg_path,
                                    pdf_dir=pdf_path,
                                    dpi=300,
                                    );

                        title  = f"avgtopo {con} {reduce} {sig_info}"
                        save_topo_of_window(
                                evoked=ev,
                                chan_type=chan_type,
                                chs_use = chs_sig,
                                time_window = window_list[0],
                                title = title,
                                    jpg_dir=jpg_path,
                                    pdf_dir=pdf_path,
                                    dpi=300,) ; 

         
    #% ----- 2 flag_plot_combined_sig_cluster: for each chan_type, plot combined channels with combined time windows
    if flag_plot_combined_sig_cluster:
        print('------------ plotting combined sig cluster ' ,
                ' config=',config_setting,
                ' chan_type=',chan_type,
                        )
        df_sig = pd.read_csv(df_siginfo_name)
            
        df = pd.DataFrame(
            [{**{k: v for k, v in d.items() 
                if k not in ['subcondct_cluster', 'condsub_sig', 'sig_cidx','pairwise_results']},
                'i_clu': k} for k, d in sig_cluster_info.items()])
        df['chan_type'] = chan_type
        df_sig_chantype = df.groupby('chan_type').agg(list).reset_index()
        df_sig_chantype["chans"] = df_sig_chantype["c_inds"].apply(
            lambda v: [ch_names_picked[i] for i in unique_in_order(_flatten_list(_ensure_list(v)))]
        )     
             
        sig_chantype_info =( 
                        f"{config_setting} "
                        f" All {len(df_sig)}cluster"
                                )
        

        # get unique channel index across clusters
        sig_cidx = list(set(chain.from_iterable([x['c_inds'] for x in sig_cluster_info.values()])))
        print('--total',len(sig_cidx),'sig lidx' )
        chs_sig_combine =   [ ch_names_picked[i] for i in sig_cidx]
        
        # get unique time windows across clusters
        sig_timepoints = sorted(# sort will return a list
            list(                  
                set(               # remove duplicates
                    chain.from_iterable(
                        [x['sig_time'] for x in sig_cluster_info.values()]  # lists of times
                    )
                )
            )
        )
        sig_tw_combined = merge_sig_windows(sig_timepoints, times_use)
    


        # ------- plot location of sigchan
        picks_cmb = mne.pick_channels(
            ch_names = example_avg_chan_type.info['ch_names'], 
            include=chs_sig_combine)
        
        info_sig = example_avg_chan_type.copy().pick(picks_cmb).info
        fig = mne.viz.plot_sensors(
            info_sig,
            kind='topomap',
            show_names=False
        )
        # Get the axes and modify the scatter plot properties
        ax = fig.axes[0]
        for collection in ax.collections:
            collection.set_facecolor('black')
            collection.set_edgecolor('black')
        pic_name = f"ALLClusSensor {sig_chantype_info}"

        fig.savefig(os.path.join(jpg_path,f"{pic_name.replace(' ','_')}.jpg"), bbox_inches='tight')# ,dpi=300
        fig.savefig(os.path.join(pdf_path,f"{pic_name.replace(' ','_')}.pdf"), bbox_inches='tight',dpi=300) # 
        

        # ------- plot erf erfdiff
        for reduce in reduce_type:
            reduce

            pic_name_erf     = f"Sig{chan_type}ERF {reduce} {sig_chantype_info}"
            pic_name_erfdiff = f"Sig{chan_type}ERFdiff {reduce} {sig_chantype_info}"
            
            peak_windows = sig_tw_combined

            erf_chantype_out, erfdiff_chantype_out = plot_erf_erfdiff(
                        # avg setting 
                        reduce=reduce,
                        # plot channel
                        sig_chans_use=chs_sig_combine,
                        # plot window
                        window_list=sig_tw_combined,
                        peak_windows_erf=peak_windows,
                        peak_windows_erfdiff=peak_windows,
                        # plot name
                        pic_name_erf=pic_name_erf,
                        pic_name_erfdiff=pic_name_erfdiff,
                        erfdiffdata_stem=pic_name_erfdiff,
                        **common_kwargs
                    )
            # save data
            plotdata_savefile_name = os.path.join(
                plotdata_path, 
                f"Plotdata Sig{chan_type} {sig_chantype_info}.pkl").replace(' ','_') 
            with open(plotdata_savefile_name, 'wb') as pickle_file:
                pickle.dump({
                    "erf_chantype_out":erf_chantype_out,
                    "erfdiff_chantype_out":erfdiff_chantype_out,
                    }, pickle_file)
        
            # ------- plot topo plot with peak time point
            for con,ev in zip([cond1_content_name,cond2_content_name,'ERF Difference'],
                        [evoked_c1,evoked_c2,evoked_diff]):
                
                df_peak = erfdiff_chantype_out['peaks_table']
                peak_t = list(df_peak[df_peak.condition ==con]['t_peak'].unique())
                if     flag_peak_of_ERFdiff:
                    peak_t = list(df_peak[df_peak.condition =='ERF Difference']['t_peak'].unique())
                    peakinfo = 'peakofDiff'

                else:
                    peak_t = list(df_peak[df_peak.condition ==con]['t_peak'].unique())
                    peakinfo = ''
                evoked_plot = ev.copy().crop(tmin = times_use_plot[0],tmax = times_use_plot[-1])
                
                title  = f"ALL butterflyplot {chan_type} {con} {sig_chantype_info}"
                check_data = butterflyplot_fullsensor(
                            evoked= evoked_plot,
                            highlight_window =peak_windows,
                            chan_type = chan_type,
                            title=title,
                            jpg_dir=jpg_path,
                            pdf_dir=pdf_path,
                            dpi=300,
                            ylim = (-vabs, vabs)
                            )
                
                title  = f"ALL jointplot {chan_type} {con} {sig_chantype_info}{peakinfo}"
                check_data = jointplot_fullsensor(
                            evoked= evoked_plot,
                            highlight_window =peak_windows,
                            chan_type = chan_type,
                            times_plot=peak_t,
                            title=title,
                            jpg_dir=jpg_path,
                            pdf_dir=pdf_path,
                            dpi=300,
                            ylim = (-vabs, vabs)
                            );
                for wini,win in enumerate(peak_windows):
                    print('---- plotting topo of window ',wini+1,win)
                    title  = f"ALL avgtopo {chan_type} {wini+1 }of{len(peak_windows)} {reduce} {con} {sig_chantype_info}"
                    save_topo_of_window(
                            evoked=ev,
                            chan_type=chan_type,
                            chs_use = chs_sig_combine,
                            time_window = win,
                            title = title,
                                jpg_dir=jpg_path,
                                pdf_dir=pdf_path,
                                        dpi=300,)    ; 


# %%
