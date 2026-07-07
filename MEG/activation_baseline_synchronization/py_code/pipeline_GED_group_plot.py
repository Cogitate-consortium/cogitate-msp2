'''
Plot group-level: GED component time courses (from pipeline_GED.py).

For the FF category-selective GED filters, plot the preferred vs non-preferred
(irrelevant) stimulus time courses:
  cond1 = GEDpref_stim  (e.g. face, from ts_stim_trl)
  cond2 = GEDirre_stim  (e.g. object, from ts_irrstim_trl)
Per subject, trials are balanced across the two conditions
(subsample_balance_conditions) and reduced by RMS (Balanced_RMS): RMS because
the GED filter sign is arbitrary, so a signed mean would cancel; balancing
equates face/object trial counts. Group mean +/- within-subject CI is plotted,
plus a trial-count distribution figure.

NOTE: only applies to GED types with an irrelevant condition (FFface/FFobje,
GEDirre_stim set). PFCact has GEDirre_stim=None (no ts_irrstim_trl) and is not
plotted here; the group_plot config selects FF only
(individual_configfile_contain_list: ["FF","nobc"]).

Config folder: /config_files/pipeline_GED_group_plot/
    0_500.json
Pairs with an individual pipeline_GED config (--individual_configfile);
cond1/cond2 time windows come from the group config.
'''

#%% import
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime

# for save data
import pickle

from _help_functions import (general_param, group_get_sublist,
                             path_allana,get_file_name,
                                 get_eps_use_info,
                                 read_configfile,
                                 colordict,
                                 turn_string_to_num
                                 
                                 )
from _help_module import (plot_and_save_trial_distribution_multiple,  subsample_balance_conditions)
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
    args.group_ana = 'pipeline_GED_group_plot'
    args.group_configfolder = 'pipeline_GED_group_plot'
    args.group_configfile = base / "config_files" / "pipeline_GED_group_plot" / "0_500.json"

    
    args.individual_ana = 'pipeline_GED'
    args.individual_configfolder = 'pipeline_GED'
    args.individual_configfile = base / "config_files" / "pipeline_GED" / "noft_nobc_deci1_nocut_FFface.json"


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
    sub_list = sub_list 
else:
    pd.DataFrame({'sub_list':sub_list}).to_csv(
        os.path.join(path_sublist,
        f'{config_setting}_{len(sub_list)}_{sub_list_name}.csv'),index=False
        )

#%% unpack the param_sub 
epoch_setting_name = param_sub['epoch_setting_name']
epoch_setting = general_param['epoch_setting_dict'][epoch_setting_name]
roi_params = general_param['roi_params_dict']
n_subsamples= general_param['Nsample'] 


# !!! special for GED
roi = param_sub['roi']
GED_type = param_sub['GED_type']
epoch_data_list = param_sub['epoch_data_list']


GEDpref_stim = param_sub['GEDpref_stim']
GEDirre_stim = param_sub['GEDirre_stim']


info =get_eps_use_info(**epoch_setting) 
deriv_root = os.path.join(path_allana, individual_ana,
                          info.epoch_info)
#%% unpack the param_group
cond1_timewindow = param_group['cond1_timewindow']
cond2_timewindow = param_group['cond2_timewindow']

#%% create folder to save
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
        datatype='meg',  
        suffix=f"GED_{GED_type}",
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
        
    if 'roi' in locals():
        assert roi == loaded_data['roi'], "error: roi not match"
    else:
        roi = loaded_data['roi']
    # check the data
    assert len(loaded_data['ts_stim_trl'][0]) == len(times),\
            'error: times not match'

    #% get GED timecourse  of stim conditon and nonstim conditon
    cond1_content_name = param_sub['GEDpref_stim'] 
    cond2_content_name = param_sub['GEDirre_stim']
    tlt_cond1 = np.array(loaded_data ['ts_stim_trl'])
    tlt_cond2 = np.array(loaded_data ['ts_irrstim_trl'])

    # balance trial counts across the two conditions, reduce each subsample by RMS
    _, _, t_cond1_Brms, t_cond2_Brms, sb_info = subsample_balance_conditions(
    cond1_tlt=tlt_cond1,
    cond2_tlt=tlt_cond2,
    n_samples=n_subsamples,                    # Number of subsamples
    trial_reducer="rms",              # or "median" or "rms"
    seed= turn_string_to_num(f"{subject}_{config_setting}"),                          # For reproducibility
    sample_reducer="mean",            # or "median" or "rms"
    verbose=True
    )

    sub_dict[subject] = {
        'Balanced_RMS':
            {
            cond1_content_name : t_cond1_Brms,
            cond2_content_name : t_cond2_Brms,
            }, }
        
    # get epoch number 
    nepoch ={
        param_sub['GEDpref_stim'] : len(loaded_data['ts_stim_trl']),
        f"NO_{param_sub['GEDpref_stim']}": len(loaded_data['ts_nostim_trl']),
        param_sub['GEDirre_stim']: len(loaded_data['ts_irrstim_trl']),
    }


    df =pd.DataFrame(nepoch, index=[0])
    df.insert(0,'subject',subject)
    df_nepoch_group = pd. concat([df_nepoch_group,df])
    
if not flag_test:
    assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'
    
print('--------- prepare ready for data of ',config_setting)
#%% plot condition trial number info
group_df_name  = os.path.join(path_group_plot, 
                              f"group_condition_trial_number_{config_setting}.csv")

plot_and_save_trial_distribution_multiple(
    df = df_nepoch_group,
    conditions = [param_sub['GEDpref_stim'],
                  f"NO_{param_sub['GEDpref_stim']}",
                  param_sub['GEDirre_stim']
                  ],
    title=f"Trial_Distribution_{config_setting}_{len(sub_list)}_{sub_list_name}",
    jpg_dir=jpg_path,
    pdf_dir=pdf_path,
    )
#%% ----------- prepare data for statistic testing
#  setup time for plot
assert cond1_timewindow==cond2_timewindow, f"{cond1_timewindow} {cond2_timewindow} Condition time windows do not match."

if 'baseline' in group_configfolder:
    times_use_w = [-0.9,1.01]
else:
    times_use_w = [-0.1, 1.1]


times_use_idx = [idx for idx,t in enumerate(times) if (t<= times_use_w[1])&(t>= times_use_w[0])]
times_use_plot = times[times_use_idx] 
linewidth_list =[2,2,4]
linestyle_list=['-' ,'-','--']

vlines           = [0] + [float(x) for x in cond1_timewindow if x!= None ]
vlines_colors    = ['k'] + ['k','k']
vlines_linestyle =  ['-'] +['--','--']
vlines_linewidth = [1] + [1,1]#%% 

from _help_functions import (compute_mean_exp2,
                             convert_to_ci_1D,
                             apply_lowpass_filter,
                             )
from _help_module import _slug
from cog_plot import plot_time_series_sigline
from pathlib import Path
# compute mean and ci
cond1_name= param_sub['GEDpref_stim']
cond2_name= param_sub['GEDirre_stim']


for plot_type in ['Balanced_RMS']:
    print(f"----- plotting {plot_type} -----")
    condsubt = np.array([[sub_dict[sub][plot_type][cond]
                for sub in sub_dict.keys()]
                for cond in [cond1_name, cond2_name]])


    condt, condt_ci_raw = compute_mean_exp2(
        condsubt, 
        axis_cond=0, 
        axis_sub=1, 
        zscore=False, 
        design="within"
    )
    condt_ci = convert_to_ci_1D(condt, condt_ci_raw)
    fs=1000/epoch_setting['epoch_decim']


    # --- low-pass + slice to requested window ---
    condt_filt = apply_lowpass_filter(condt, fs=fs)                 # (Cond, T)
    condt_ci_filt = apply_lowpass_filter(condt_ci, fs=fs)           # (Cond, T, 2)


    plat_condt = condt_filt[:, times_use_idx]                               
    plat_condt_ci = condt_ci_filt[:, times_use_idx, :]

    # --- plot settings ---
    flag_preset_ylim = 1
    if 'rms' in plot_type.lower():
        if '_nobc' in epoch_setting_name:

            ylim_erf = (1,1.5)
        else:
            ylim_erf = (1,1.8)
    elif 'mean' in plot_type.lower():
        if '_nobc' in epoch_setting_name:   
            ylim_erf = (-0.09,0.09)
        else:
            ylim_erf = (-0.12,0.12)
    xtick_interval =0.5

    conds_name = [cond1_name, cond2_name]
    conds_colors = [colordict[cond1_name], colordict[cond2_name]]
    ylabel = "Activity (a.u.)"
    ylim = ylim_erf if flag_preset_ylim else None


    title = f"{config_setting}_{len(sub_list)}sub_{plot_type}"

    ax = plot_time_series_sigline(
        plotdata_save_folder_path=os.path.join(plotdata_path, _slug(title)),
        data=plat_condt,
        err=None,
        ci_1D=plat_condt_ci,
        t0=times_use_plot[0],
        tend=times_use_plot[-1],
        ax=None,
        linewidth_list=linewidth_list,
        linestyle_list=linestyle_list,
        xtick_interval=xtick_interval,
        colors=conds_colors,
        vlines=vlines,
        vlines_colors=vlines_colors,
        vlines_linestyle=vlines_linestyle,
        vlines_linewidth=vlines_linewidth,
        xlim=None,
        ylim=ylim,
        xlabel="Time (s)",
        ylabel=ylabel,
        err_transparency=0.2,
        title=title,
        square_fig=False,
        conditions=conds_name,
        do_legend=False,
        sig_hatchedpatterns=None,
        sig_hatchedpattern_dataidx=[0, 1],
        dpi=300,
    )

    # legend & save
    h, l = ax.get_legend_handles_labels()
    fig = ax.figure
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)

    fig.savefig(Path(jpg_path) / f"{_slug(title)}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(Path(pdf_path) / f"{_slug(title)}.pdf", bbox_inches="tight", dpi=300)
    print(f"----- finished  {title}-----",datetime.now())

#%%