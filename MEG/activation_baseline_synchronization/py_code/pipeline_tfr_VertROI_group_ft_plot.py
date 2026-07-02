'''
Plot + Bayesian stats of the group-level freq x time TFR analysis for a vertex-ROI
(pipeline_tfr_VertROI_group_ft.py): frequency x time power maps per condition and
their log power ratio, with Bayesian BF masks overlaid.

Run first: pipeline_tfr_VertROI.py (individual) then
pipeline_tfr_VertROI_group_ft.py (group freq x time CBPT).

Reloads each subject's per-label (vertices, freqs, times) power, averages over
vertices -> per-label (freqs, times), then:
  - computes the Bayesian paired t-test per freq x time point (bayes_ttest, tail
    from config, JZS r=0.707) -> bf10/bf01, plus a band-averaged bf (alpha 8-13 Hz
    for 'slow' configs, gamma 60-90 Hz for 'fast' configs); saved as bf_*.pkl and
    csv/bf_*.csv,
  - plots the group freq x time power maps masked at bf01>3, bf01>10 and p<0.05.
Unlike pipeline_tfr_source_group_ft.py (where the Bayesian test lives in the group
script), here it is computed in this plot script. Uses cog_plot.plot_matrix; saved
as jpg/pdf/plotdata under analysis_group_results_plot/<group_configfolder>/.

Config folder: /config_files/pipeline_tfr_VertROI_group_ft_baseline/
    fast_30_100_pre500_0.json
    slow_1_30_pre500_0.json
Paired with an individual config /config_files/pipeline_tfr_VertROI/*.json.
'''


#%% import
from cog_plot import (plot_matrix,
                      )

from _help_module import plot_and_save_trial_distribution_multiple


from _help_functions import (general_param,
                             get_file_name,
                             group_get_sublist,
                             path_allana,
                                 get_eps_use_info,
                                 read_configfile,
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
    args.group_ana = 'pipeline_tfr_VertROI_group_ft_plot'
    args.group_configfolder = 'pipeline_tfr_VertROI_group_ft_baseline'
    args.group_configfile = base / "config_files" / "pipeline_tfr_VertROI_group_ft_baseline" / "fast_30_100_pre500_0.json"

    
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


#% set up the freq band of interest
if "vert" in config_setting.lower():
    hlines = [60,90]   # gamma for IIT
    band_names = ['gamma',]
    bands_range = [(60,90),]

else:
    hlines = [4,8,13,30,60,90]   # theta/beta/gamma    
    band_names = ['theta','beta','gamma']
    bands_range = [(4,8),(13,30),(60,90),]


#%% Bayesian t test
from bayes_factor_fun import bayes_ttest
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
bf10, pval = bayes_ttest(cond1_subft, cond2_subft, 
                       paired=True, 
                       alternative=alternative, 
                       r=0.707, 
                       return_pval=True)
assert bf10.shape == (len(freqs),len(cond_time)), 'error: bf shape not match'

if isinstance(bf10, str):
    bf10 = float(bf10)
bf01 = 1 /bf10

# 2 average over time points and frequency to get one bayes factor for the intersted time window
bf_avg_band = {}
bf_avg_band_df = pd.DataFrame()
if 'slow' in config_setting.lower():
    band_name ='alpha'
    band_range =[8,13]
elif 'fast' in config_setting.lower():
    band_name ='gamma'
    band_range =[60,90]
else:
    ValueError('error: cannot find band name in config_setting')

fidx = [idx for idx,f in enumerate(freqs) if (f>=band_range[0])&(f<=band_range[1])]
bf10_avg, pval_avg = bayes_ttest(
    cond1_subft[:, fidx, :].mean(axis=(1,2)), 
    cond2_subft[:, fidx, :].mean(axis=(1,2)), 
    paired=True, alternative=alternative, r=0.707, return_pval=True)
if isinstance(bf10_avg, str):
    bf10_avg = float(bf10_avg)
bf01_avg = 1 / bf10_avg
bf_avg_band.update({ f"bf10_avg_{band_name}": bf10_avg,
                    f"bf01_avg_{band_name}": bf01_avg,
                    f"pval_avg_{band_name}": pval_avg})
a ={"band_name": band_name,
    "tail": tail,
    "alternative": alternative,
    "cond1_content_name": cond1_content_name,
    "cond2_content_name": cond2_content_name,
    "bf10_avg": bf10_avg,
    "bf01_avg": bf01_avg,
    "pval_avg": pval_avg}
bf_avg_band_df = pd.concat([bf_avg_band_df, 
                            pd.DataFrame(a, index=[0])], ignore_index=True)

print(f"{band_name} bf01_avg", bf01_avg, " pval_avg ", pval_avg)
# save data
with open(bf_name, 'wb') as pickle_file:
    pickle.dump({"bf10":bf10,
                 "bf01":bf01,
                 "pval":pval,
                 "bf_avg_band": bf_avg_band,
                 }, pickle_file)
# save csv  
csv_name = os.path.join(path_group,'csv', f"bf_{config_setting}.csv")
os.makedirs(os.path.dirname(csv_name), exist_ok=True)
bf_avg_band_df.to_csv(csv_name, index=False)
print('------------ save bayes factor file ' ,
        '\n','file ',bf_name,
        '\n','csv file ',csv_name,
        '\n','config ',config_setting,
        '\n',datetime.now(),
                        )

#%% plot Bayesian t test
for criterion_name in ['bf01_over_3','bf01_over_10','p_0.05']:
    if criterion_name == 'bf01_over_3':
        criterion = bf01 > 3
    elif criterion_name == 'bf01_over_10':
        criterion = bf01 > 10
    elif criterion_name == 'p_0.05':
        criterion = pval < 0.05

    for data_ft,plot_content,cbar_label in zip(
        [cond1_ft,
         cond2_ft,
         conddif_ft],
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
            title = f"ft {criterion_name.replace('_',' ')} {plot_content}", 
            square_fig=False, 
            dpi=300)
      