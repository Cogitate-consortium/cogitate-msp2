'''
Plot of the group-level freq x time TFR analysis (pipeline_tfr_source_group_ft.py):
frequency x time power maps per condition and their log power ratio, for one ROI,
with Bayesian and significant-cluster masks overlaid.

Run first: pipeline_tfr_source.py (individual) then
pipeline_tfr_source_group_ft.py (group freq x time Bayesian + CBPT).

Reloads each subject's per-label (vertices, freqs, times) power, averages over
vertices -> per-label (freqs, times), then:
  - plots the group Bayesian bf / pval maps (loaded from bf_*.pkl) masked at
    BF>3 and p<0.05 over the freq x time plane,
  - if no significant cluster: plots the group-mean freq x time map per label,
  - if significant clusters: plots each cluster's freq x time map with the
    cluster mask (loaded from cbpt_allresult_*).
Uses cog_plot.plot_matrix; saved as jpg/pdf/plotdata under
analysis_group_results_plot/<group_configfolder>/.

Only plots; all statistics come from pipeline_tfr_source_group_ft.py.

Config folder: /config_files/pipeline_tfr_source_group_ft_250_500/
    fast_30_100_250_500.json
    slow_1_30_250_500.json
Paired with an individual config /config_files/pipeline_tfr_source/*.json.
'''


#%% import
import matplotlib.pyplot as plt
from cog_plot import (plot_matrix,
                      )

from _help_module import plot_and_save_trial_distribution_multiple


from _help_functions import (general_param,
                             get_file_name,
                             group_get_sublist,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
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
    args.group_ana = 'pipeline_tfr_source_group_ft'
    args.group_configfolder = 'pipeline_tfr_source_group_ft_250_500'
    args.group_configfile = base / "config_files" / "pipeline_tfr_source_group_ft_250_500" / "fast_30_100_250_500.json"

    
    args.individual_ana = 'pipeline_tfr_source'
    args.individual_configfolder = 'pipeline_tfr_source'
    args.individual_configfile      = base / "config_files" / "pipeline_tfr_source" / "dAT_probe_stim_noft_nobc_deci5_nocut_PFC_30_100.json"


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
print('--------- start ',config_setting,datetime.now())

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
roi_params = general_param['roi_params_dict']

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
    
labels = [x for x in label_dict.keys() if 'all' not in x]
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



#%% ----------- prepare data for statistic testing
# ---set the time for statistic   
cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
assert cond1_tidx == cond2_tidx, 'error: time window not match'
cond_tidx = cond1_tidx
cond_time  = times[cond_tidx]
# --- data with select the statistic time window
power1_sub_label_f_t =  \
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

power2_sub_label_f_t =  \
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
                            power1_sub_label_f_t  /\
                            power2_sub_label_f_t )



# --- logged power data avg over labels for Bayesian t test
cond1_subft = np.log10(np.mean(power1_sub_label_f_t,axis=1 ))
cond2_subft = np.log10(np.mean(power2_sub_label_f_t,axis=1 ))




sub_label_f_t =  \
    np.stack([
    np.stack([# (vertices, freqs, time): select bandfreq
np.log10( 
np.nanmean(sub_dict[sub][lab][cond1_content_name][:, :, cond1_tidx], axis=(0))  /\
np.nanmean(sub_dict[sub][lab][cond2_content_name][:, :, cond2_tidx], axis=(0)) 
         )
        for lab in labels
    ], axis=0)  # → (n_labels, n_time)
    for sub in sub_dict.keys()
], axis=0)  



# --- for plot: avg over labels and participant to get ft 
# here plot the raw for each condiion and log power ratio for difference
cond1_labelft = np.nanmean(power1_sub_label_f_t,axis=0 )
cond2_labelft = np.nanmean(power2_sub_label_f_t,axis=0 )

conddif_labelft = \
    np.stack([
np.log10( 
        # (freqs, time)
np.squeeze(cond1_labelft[li,:,:])  /\
np.squeeze(cond2_labelft[li,:,:]) 
         )
        for li,lab in enumerate(labels)
    ], axis=0)  # → (n_labels,n_frequ, n_time)
 


cond1_ft = np.nanmean(power1_sub_label_f_t,axis=(1,0) )
cond2_ft = np.nanmean(power2_sub_label_f_t,axis=(1,0) )
conddif_ft = np.log10(cond1_ft/cond2_ft)


#%% settings for plot
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


# settings for matrix plot
x0= cond_time[0]
x_end= cond_time[-1]
cmap = 'RdBu_r' 
y0   = freqs[0]
y_end= freqs[-1]

if  cond1_timewindow==cond2_timewindow:
    times_use = times[cond1_tidx]
else:
    ValueError
cond_statistic_timewindow=cond1_timewindow


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

#%% plot Bayesian t test
bf_name  = os.path.join(path_group, f"bf_{config_setting}.pkl")

if flag_test:
        
    from bayes_factor_fun import bayes_ttest
    bf, pval = bayes_ttest(cond1_subft, cond2_subft, 
                        paired=True, 
                        alternative='two-sided', 
                        r=0.707, 
                        return_pval=True)
    # with open(bf_name, 'wb') as pickle_file:
    #     pickle.dump({"bf":bf,
    #                  "pval":pval}, pickle_file)
else:
    with open(bf_name, 'rb') as pickle_file_load:
        loaded_bf = pickle.load(pickle_file_load)
    bf = loaded_bf['bf']
    pval = loaded_bf['pval']

print(bf.shape)
assert bf.shape == (len(freqs),len(cond_time)), 'error: bf shape not match'
for criterion_name in ['bf_3','p_0.05']:
    if criterion_name == 'bf_3':
        criterion = bf > 3
    elif criterion_name == 'p_0.05':
        criterion = pval < 0.05

    for data_ft,plot_content,cbar_label in zip([cond1_ft,cond2_ft,conddif_ft],
                                    [f'{cond1_content_name} avg power',
                                    f'{cond2_content_name} avg power',
                                    f'log power diff {cond1_content_name} {cond2_content_name}'],
                                    ['Power',
                                    'Power',
                                    'Log power difference'
                                    ],):
                                    
        print('max of log power diff',np.max(data_ft))
        print('min of log power diff',np.min(data_ft))

        # Create a grid of indices
        mask = criterion
        if 'diff' in plot_content:
            ylim =(-np.max(abs(data_ft)),np.max(abs(data_ft)))
        else:
            cond_min = np.min([cond1_ft,cond2_ft])
            cond_max = np.max([cond1_ft,cond2_ft])
            ylim = (cond_min,cond_max)
                
          
        ax,im = plot_matrix(
            plotdata_save_folder_path = plotdata_path,
            flag_colorbar = 1,
            flag_return_image= 1,
            data= data_ft, 
            x0   = x0, 
            x_end= x_end, 
            y0   = y0,
            y_end= y_end,  
            mask=mask, 
            cmap=cmap, 
            ax  =None, 
            ylim=ylim,
            midpoint=None,
            transparency=1.0, 
            interpolation='lanczos', 
            xlabel='Time (s)', 
            ylabel='Freq (Hz)', 
            xticks=None, 
            yticks=None, 
            cbar_label= cbar_label, 
            filename=None, 
            vlines = None, 
            hlines = None,
            title = f"BFft {criterion_name} {plot_content}", 
            square_fig=False, 
            dpi=300)

#%% cluster based permutation statistic testing
clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.csv")
assert os.path.exists(cbpt_result_name), 'error: not exist cbpt_result_name'


if  os.path.exists(blankfile_name) :
    #% ------- plot group avg as no significant cluster found ---
    print('------------','start plot group result as no significant cluster found' ,
        '\n',config_setting,
        '\n',datetime.now(),)
    
    for data_labelft,plot_content,cbar_label in zip(
                        [cond1_labelft,cond2_labelft,conddif_labelft],
                         [f'{cond1_content_name} avg power',
                                f'{cond2_content_name} avg power',
                                f'log power diff {cond1_content_name} {cond2_content_name}'],
                                ['Power',
                                'Power',
                                'Log power difference'
                                ],):
                                    
        print('max of log power diff',np.max(data_labelft))
        print('min of log power diff',np.min(data_labelft))
        if 'diff' in plot_content:
            ylim =(-np.max(abs(data_labelft)),np.max(abs(data_labelft)))
            cmap = 'RdBu_r'
        else:
            cond_min = np.min([cond1_labelft,cond2_labelft])
            cond_max = np.max([cond1_labelft,cond2_labelft])
            ylim = (cond_min,cond_max)   
            cmap = 'Reds'    
        # roi pic name
        groupmean_pic_name = f'GroupFT {plot_content} {config_setting} {len(sub_list)}sub'
        plotdata_path_groupmean_pic = os.path.join(plotdata_path, groupmean_pic_name.replace(' ', '_'))
        os.makedirs(plotdata_path_groupmean_pic, exist_ok=True)
        
        # parepare the labels to plot
        nplot = len(labels)
        nwarp = 4
        ncols =  np.min([nwarp,nplot])
        nrow = int(np.ceil(nplot/ncols))
        size_width = 5
        size_height = 4 
        fig, axs = plt.subplots(nrow,ncols,figsize=(ncols* size_width, nrow* size_height),
                constrained_layout=True) 
        freq_edge = [v  for key,x in tfr_map[freq_range_name]['include_bands'].items() for v in x]
        
        for labeli,label_name in enumerate(labels):                        
            row = labeli // ncols  # first fill first row so divide columnnum
            col = labeli % ncols if ncols!=1 else 1  # Calculate column index
            if  (nrow==1) & (ncols==1) :
                ax = axs
            elif  (nrow ==1) or (ncols==1)  :
                ax = axs[labeli]
            else:
                ax = axs[row, col]
            data_ft = np.squeeze(data_labelft[labeli,:,:])
            
            ax,im = plot_matrix(
                plotdata_save_folder_path = plotdata_path_groupmean_pic,
                flag_colorbar = 0,
                flag_return_image= 1,
                data= data_ft, 
                x0   = x0, 
                x_end= x_end, 
                y0   = y0, 
                y_end= y_end,  
                mask=None, 
                cmap=cmap, 
                ax  =ax, 
                ylim=ylim,
                midpoint=None,
                transparency=1.0, 
                interpolation='lanczos', 
                xlabel='Time (s)', 
                ylabel='Freq (Hz)', 
                xticks=None, 
                yticks=None, 
                cbar_label= cbar_label, 
                filename=None, 
                vlines = vlines, 
                hlines = freq_edge,
                title = label_name, 
                square_fig=False, 
                dpi=300)
        # cb = plt.colorbar(im)
        # cb.ax.set_ylabel(cbar_label)
        # cb.ax.set_yscale('linear')
        # create a new ax for color bar then add the colorbar to the new Axes

        cbar_ax = fig.add_axes([1.05, 0.1, 0.03, 0.8])  # [left, bottom, width, height] — Adjust these values as needed
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel(cbar_label)
        cbar.ax.set_yscale('linear')
    
        #% h, l = ax.get_legend_handles_labels()
        # fig.legend(h,l,loc="lower center",bbox_to_anchor=(1.06, 0.5))
        fig.suptitle(groupmean_pic_name, y = 1.021,fontsize=14)
        #%
        fig.savefig(os.path.join(jpg_path,f'{groupmean_pic_name}.jpg'), 
                    bbox_inches='tight',dpi=300)#     
        fig.savefig( os.path.join(pdf_path,f'{groupmean_pic_name}.pdf'), 
                    bbox_inches='tight',dpi=300)#      

#% ------- if cluster found, load the cluster info and plot each significant cluster --- 
if (os.path.exists(clusterfile_name)):
    print('------------','exist ' ,
            '\n',config_setting,
            '\n',datetime.now(),
    )
    with open(cbpt_result_name, 'rb') as pickle_file_load:
        cluster_stats = pickle.load(pickle_file_load)
    print('------------','Successfully loaded cluster_stats' ,
        '\n',config_setting,
        '\n',datetime.now(),
            )    
    # -------------- Get Significant Clusters --------------
    # Extract cluster information
        
    T_obs, clusters, cluster_p_values, H0 = cluster_stats
    good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
    print('------------  find good cluster ',len(good_cluster_inds),datetime.now())
    print(cluster_p_values)

    #% ------- Step 2. Plot each Significant cluster ---
    sig_cidx = [] 
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
    
    cond1_labelft_cluster = cond1_labelft[sig_cidx,:,:]
    cond2_labelft_cluster = cond2_labelft[sig_cidx,:,:]
    conddif_labelft_cluster = conddif_labelft[sig_cidx,:,:]
    
    for data_labelft,plot_content,cbar_label in zip(
                        [cond1_labelft_cluster,
                         cond2_labelft_cluster,
                         conddif_labelft_cluster],
                         [f'{cond1_content_name} power',
                                f'{cond2_content_name} power',
                                f'log power diff {cond1_content_name} {cond2_content_name}'],
                                ['Power',
                                'Power',
                                'Log power difference'
                                ],):
                                    
        print('max of log power diff',np.max(data_labelft))
        print('min of log power diff',np.min(data_labelft))
        if 'diff' in plot_content:
            ylim =(-np.max(abs(data_labelft)),np.max(abs(data_labelft)))
            cmap = 'RdBu_r'
        else:
            cond_min = np.min([cond1_labelft_cluster,cond2_labelft_cluster])
            cond_max = np.max([cond1_labelft_cluster,cond2_labelft_cluster])
            ylim = (cond_min,cond_max)   
            cmap = 'Reds'    
        
        plot_name = f'ClusterFT {config_setting} {len(good_cluster_inds)}sigcluster {len(sig_cidx)}siglabel {plot_content}'
        plotdata_path_ClusterFT_pic = os.path.join(plotdata_path, plot_name.replace(' ', '_'))
        os.makedirs(plotdata_path_ClusterFT_pic, exist_ok=True)

        # ------- plot onepic with subplot of each cluster cluster info
        nplot = len(good_cluster_inds)
        nwarp = 4
        ncols =  np.min([nwarp,nplot])
        nrow = int(np.ceil(nplot/ncols))
        size_width = 5
        size_height = 4 
        fig, axs = plt.subplots(nrow,
                                ncols,
                                figsize=(ncols* size_width, nrow* size_height),
                constrained_layout=True) 
        # plot current sig cluster
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            label_inds, freq_inds, time_inds = clusters[clu_idx]
            assert len(np.unique(label_inds))==1, 'error: more than one label in one cluster'
            l_inds =list(np.unique(label_inds))
            l_ind=l_inds[0]
            f_inds =list(np.unique(freq_inds))
            t_inds = list(np.unique(time_inds))
            
            # get index for each cluster
            sig_tidx = [cond_tidx[x] for x in t_inds]
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_cidx = sig_cidx + l_inds[0]
            sig_freq = [freqs[x] for x in f_inds]
            

                
            row = i_clu // ncols  # first fill first row so divide columnnum
            col = i_clu % ncols if ncols!=1 else 1  # Calculate column index
            if  (nrow==1) & (ncols==1) :
                ax = axs
            elif  (nrow ==1) or (ncols==1)  :
                ax = axs[i_clu]
            else:
                ax = axs[row, col]
            
            
            # ----- get the mask for each cluster
            mask_ft = np.zeros_like(T_obs[0], dtype=bool)  # (freq x time)
            mask_ft[freq_inds, time_inds] = True
            
            # ----- get the data for each label
            data_ft = np.squeeze(data_labelft[l_ind,:,:])
            
            
            vlines = [float(x) for x in cond_statistic_timewindow if x!= None ] + [0]
            freq_edge = [v  for key,x 
                            in tfr_map[freq_range_name]['include_bands'].items()
                            for v in x]
            conds_name     = [f'{cond1_content_name}_{cond2_content_name}']
            conds_colormap =  [colordict[cond1_content_name]]
            label_name = labels[l_ind]

            ax1,im = plot_matrix(
                plotdata_save_folder_path = plotdata_path_ClusterFT_pic,
                            ax = ax,
                            flag_colorbar = 0,
                            flag_return_image= 1,
                            data= data_ft, 
                            x0   = x0, 
                            x_end= x_end, 
                            y0   = y0, 
                            y_end= y_end,  
                            mask= mask_ft, 
                            cmap=None, 
                            midpoint=None,
                            transparency=1.0, 
                            interpolation='lanczos', 
                            xlabel='Time (s)', 
                            ylabel='Freq (Hz)', 
                            xticks=None, 
                            yticks=None, 
                            ylim=ylim,
                            cbar_label= cbar_label, 
                            filename= None , 
                            vlines = vlines, 
                            hlines = freq_edge,
                            title = label_name, 
                            square_fig=False, 
                            dpi=300)
            
        cbar_ax = fig.add_axes([1.05, 0.1, 0.03, 0.8])  # [left, bottom, width, height] — Adjust these values as needed
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel(cbar_label)
        cbar.ax.set_yscale('linear')
        
        fig.suptitle(plot_name, y = 1.021,fontsize=14)
        fig.savefig(os.path.join(jpg_path,
                                    f'{plot_name.replace(' ','_')}.jpg'), 
                            bbox_inches='tight',dpi=300) # 
        fig.savefig(os.path.join(pdf_path,
                                        f'{plot_name.replace(' ','_')}.pdf'), 
                                bbox_inches='tight' ,dpi=300) 

print('------------ finish ',
        '\n','config ',config_setting,
        '\n', 'roi ',roi,
        '\n', datetime.now()
       )
#%%