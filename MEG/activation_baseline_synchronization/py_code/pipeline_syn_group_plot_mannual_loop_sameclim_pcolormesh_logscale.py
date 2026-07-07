
'''
Plot Group-level: synchrony results as freq x time matrices
(pcolormesh, same color limits across conditions, y-axis on log scale, sharp
cluster contours). The long file name records the chosen options; the version
actually used is this log-scale one (flag_y_logscale = 1).

Manual loop (flag_manual = 1, no argparse) over the synchrony group analyses:
  - pipeline_syn_ppc_GED_ROI_group_250_500        (PPC, GNW: FF-GED vs PFC-GED)
  - pipeline_syn_power_dfc_GED_ROI_group_250_500   (DFC, GNW: FF-GED vs PFC-GED)

  - pipeline_syn_ppc_PCA_VertROI_group_100_600     (PPC, IIT: FF vertices vs V1/V2)
  - pipeline_syn_ppc_PCA_VertROI_group_baseline    (PPC, IIT baseline)
paired with the matching individual subsampling analyses.

For each (group config x individual config) it:
  - loads the per-subject, per-condition value (freq x time 'ft' for PPC, or
    freq x window 'fw' for DFC; for DFC the time axis is the sliding-window
    mean timepoints),
  - averages over subjects and takes the cond1 - cond2 difference,
  - computes a Bayesian paired t-test with alternative derived from the config
    'tail' (greater/less for tail=+/-1, two-sided for tail=0) and saves/plots it;
    this OVERWRITES the two-sided bf_{config}.pkl written by the group scripts,
  - loads the PRE-COMPUTED cluster result (cbpt_allresult_*.pkl) and rebuilds
    the significance mask in the plot window,
  - plots cond1, cond2 and their difference (and a zoom on the significant
    frequency range) via plot_matrix_pcolormesh.
Note: the CBPT is only loaded (computed in the group scripts), but the Bayesian
t-test IS (re)computed here in its tail-based form.
'''

#%% import
from cog_plot import plot_matrix_pcolormesh
from frites.conn import define_windows

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# for show time
from datetime import datetime

# for save data
import pickle


from _help_functions import (group_get_sublist,
                             get_file_name, 
                             get_eps_use_info,
                                 read_configfile,
                                 )
from _help_functions import (general_param,
                             path_allana,
                                 )

import numpy as np
import pandas as pd

import mne_bids

clim_preset_diff = [-0.1,0.1]
clim_preset_raw = [0,0.2]

flag_same_clim = 1
flag_y_logscale = 1

flag_no_newsave =0
flag_sharp_contour = 1

#%% setting test
flag_manual = 1
if   flag_manual:
    print('---------------------------- test flag on',datetime.now())
    # Create an instance of Args and set the subject attribute
    group_ana =  "pipeline_syn_group_plot"# 
    group_configfolders     = ['pipeline_syn_ppc_GED_ROI_group_250_500',
                    'pipeline_syn_ppc_PCA_VertROI_group_100_600' ,
                    'pipeline_syn_ppc_PCA_VertROI_group_baseline',
                    'pipeline_syn_power_dfc_GED_ROI_group_250_500']
    individual_anas=["pipeline_syn_subsampling_ppc_GED_ROI",
                    "pipeline_syn_subsampling_ppc_PCA_VertROI",
                    "pipeline_syn_subsampling_ppc_PCA_VertROI",
                    "pipeline_syn_subsampling_dfc_GED_ROI",
                    ]
    individual_configfolders= ["pipeline_syn_ppc_GED_ROI",
                    "pipeline_syn_ppc_PCA_VertROI",
                    "pipeline_syn_ppc_PCA_VertROI",
                    "pipeline_syn_power_dfc_GED_ROI"
                    ]
for group_configfolder,individual_ana,individual_configfolder in zip(group_configfolders, individual_anas, individual_configfolders):
    print('---------------------------- \n start group_configfolder ',group_configfolder,datetime.now())
    print(' individual_ana ',individual_ana)


    from _help_functions import Project_Dir
    def list_filetype(directory, extension=".json"):
        try:
            json_files = [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(extension)]
            return json_files
        except FileNotFoundError:
            return f"Error: {directory} is not a valid directory."


    pwd = Project_Dir #os.getcwd()
    file_sh       = os.path.join(pwd,'py_code','_group_run.sh')

    group_config_folder   = os.path.join(pwd,'config_files',group_configfolder)
    individual_config_folder   = os.path.join(pwd,'config_files',individual_configfolder)

    # Get the different configfile
    group_config_files = list_filetype(group_config_folder, extension=".json")
    print('-------- get group_config_files',len(group_config_files), 'configfile')
    individual_config_files = list_filetype(individual_config_folder, extension=".json")
    print('--------individual_config_files',len(individual_config_files), 'configfile')

    #% Launching a job for each:
    for gi,group_configfile in enumerate(group_config_files):
        print(gi,group_configfile)

        config_group = get_file_name(group_configfile)
        param_group = read_configfile(group_configfile)

        for ii, individual_configfile in enumerate (individual_config_files):
            print(ii,individual_configfile)

            config_indiv = get_file_name(individual_configfile)
            configi = (gi)*len(individual_config_files) + (ii+1)
            
            # selecting configfile based on the group configfile
            if 'individual_configfile_contain_list' in param_group.keys():
                if any(x not in config_indiv for x in param_group['individual_configfile_contain_list']):
                    print(f'skip {configi} ---- ','\n',config_indiv,' ',config_group,
                        '\n','for not contain ', param_group['individual_configfile_contain_list'])
                    continue
            if 'individual_configfile_contain_either_list' in param_group.keys():
                if not any(
                x in config_indiv for x 
                in param_group['individual_configfile_contain_either_list']):
                    print(f'skip {configi} ---- ','\n',config_indiv,' ',config_group,
                        '\n','for not contain either ', param_group['individual_configfile_contain_either_list'])
                    continue

            
            if 'individual_configfile_notcontain_list' in param_group.keys():
                if any(x in config_indiv for x in param_group['individual_configfile_notcontain_list']):
                    print(f'skip {configi} ---- ','\n',config_indiv,
                        '\n','for contain ', param_group['individual_configfile_notcontain_list'])
                    continue
            print(f'Run {configi} ++++++++',config_indiv ,' ' ,config_group,
                '\n',f'group {gi+1}of{len(group_config_files)}',
                f'indi {ii+1}of{len(individual_config_files)}',
                    )
                

            #% load configure and setting path
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

            sub_list_name, sub_list = group_get_sublist(param_group,
                                                        param_sub,
                                                        group_configfile,
                                                        individual_configfile,
                                                        )
            value_type = "DFC" if 'dfc' in individual_ana.lower()  else 'MI' if 'mi' in individual_ana.lower() else "PPC"

            #% unpack the param_sub
            epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
            roi_params = general_param['roi_params_dict']

            # !!! Specific for tfr
            tfr_map = general_param['tfr_map_dict'][param_sub['tfr_method']]
            freq_range_name = param_sub['freq_range_name']

            #% unpack the genral_param
            # setting statistic parameters
            tail = param_group['tail']  
            cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]

            cond1 = param_group['cond1']
            cond2 = param_group['cond2']
            cond1_timewindow = param_group['cond1_timewindow']
            cond2_timewindow = param_group['cond2_timewindow']

            #!!! group config specify the data used
            ROIpairs_name = param_group['ROIpairs_name']
            input_type = param_group['input_type']  # 'NOrm' 


            p_cluster_forming= cbpt_setting['p_cluster_forming'] 
            out_type = cbpt_setting['out_type']
            # n_permutations = 20 if flag_manual else cbpt_setting['n_permutations'] 
            n_permutations =  cbpt_setting['n_permutations'] 

            # setting individual result path
            info =get_eps_use_info(**epoch_setting)
            deriv_root = os.path.join(path_allana, 
                                        individual_ana,
                                        info.epoch_info)
            # setting the group result folder
            path_group = os.path.join(path_allana, individual_ana,
                                    "analysis_group_results",
                                    group_configfolder,)
            path_group_csv = os.path.join(path_group, 'csv',)   
            os.makedirs(path_group_csv, exist_ok=True) 

            # setting the group plot folder
            plot_folder_name='pipeline_syn_group_plot_pcolormesh'
            if flag_manual:
                plot_folder_name= f"{plot_folder_name}_mannual"
            if flag_same_clim:
                plot_folder_name = f"{plot_folder_name}_sameclim"
            if flag_y_logscale:
                plot_folder_name = f"{plot_folder_name}_ylogscale"
            if flag_sharp_contour:
                plot_folder_name = f"{plot_folder_name}_sharpc"

            path_group_plot = os.path.join(path_allana, plot_folder_name,
                                        "analysis_group_results_plot"
                                        )
            os.makedirs(path_group_plot, exist_ok=True)
                                    

            #% ----------- load individual data
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
                for trl_typei, trl_type in enumerate ([cond1, cond2]):
                    ############ load individual ppc result ############
                    # preset path for save ppc result
                    # !!!!! Specific for MI: add MI_method_name and data_type
                    MI_method_info = f"_{param_sub['MI_method_name']}" if 'MI_method_name' in param_sub.keys() else ''
                    data_type_info = f"_{param_group['data_type']}" if 'data_type' in param_group.keys() else ''
                    suffix=(
                        f"{epoch_data_name}_"
                        f"{input_type}_"
                        f"{param_sub['tfr_method'].replace('_','')}_"
                        f"{param_sub['freq_range_name']}"
                        f"{MI_method_info}"
                        f"{data_type_info}_"
                        f"{ROIpairs_name}_"
                        f"{trl_type}"
                    )
                    
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
                        
                    if 'freqs' in locals():
                        assert np.all(np.array(freqs) == np.array(loaded_data['freqs'])),'error: freqs not match'
                    else:
                        freqs = np.array(loaded_data['freqs'])
                        
                    if "dfc_setting" in loaded_data['param_config'].keys():
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
                        times_load = sl_win_meantimepoints
                    else:
                        times_load = times

                    # get freq * time data (PPC 'ft') or freq * window data (DFC 'fw')
                    ft = loaded_data ['ft'] if 'ft' in loaded_data.keys() else loaded_data['fw']
                    assert ft.shape == (len(freqs), len(times_load)), 'error: data shape not match'
                    sub_dict[subject][trl_type] = ft
                    
            print(f'--------- prepare ready for data of \n{gi}/{len(group_config_files)} {group_configfile} \n{ii}/{len(individual_config_files)} {individual_configfile} \n{config_setting}')

            #% ----------- prepare data for statistic testing
            # ---set the time for statistic   
            cond1_tidx = [idx for idx,t in enumerate(times_load) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
            cond2_tidx = [idx for idx,t in enumerate(times_load) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
            assert cond1_tidx == cond2_tidx, 'error: time window not match'
            cond_tidx = cond1_tidx
            cond_time  = times_load[cond_tidx]

            # --- data with full freq, full time window (freq,time)
            cond1_subft = np.stack([sub_dict[s][cond1] for s in sub_dict.keys()], axis=0)  # (n_sub, n_freq, n_time)
            cond2_subft = np.stack([sub_dict[s][cond2] for s in sub_dict.keys()], axis=0)  # (n_sub, n_freq, n_time)


            # --- calculate the difference between conditions
            # current each value is one ppc value, so we do subtract between conditions
            conddiff_subft = cond1_subft - cond2_subft

            cond1_ft = np.nanmean(cond1_subft,axis=0 )
            cond2_ft = np.nanmean(cond2_subft,axis=0)
            conddif_ft = cond1_ft - cond2_ft
            assert conddif_ft.shape == (len(freqs), len(times_load)), 'error: conddif_ft shape not match'

            #% settings for matrix plot
            if 'baseline' in group_configfolder:
                times_use_w = [-0.9,1.01]
            else:
                times_use_w = [-0.51,1.01]

            times_use_idx = [idx for idx,t in enumerate(times_load) if (t<= times_use_w[1])&(t>= times_use_w[0])]
            times_use = times_load[times_use_idx]

            cond1_ft_plot = cond1_ft[:, times_use_idx]
            cond2_ft_plot = cond2_ft[:, times_use_idx]
            conddif_ft_plot = conddif_ft[:, times_use_idx]     
            
            # Get the indices of cond_time within times_use
            cond_tidx_plot = [idx for idx, t in enumerate(times_use) if t in cond_time]

            x0= times_use[0]
            x_end= times_use[-1]

            y0   = freqs[0]
            y_end= freqs[-1]

            #% set up the freq band of interest
            if "vert" in config_setting.lower():
                hlines = [60,90]   # gamma for IIT
                band_names = ['gamma',]
                bands_range = [(60,90),]

            else:
                hlines = [4,8,13,30,60,90]   # theta/beta/gamma    
                band_names = ['theta','beta','gamma']
                bands_range = [(4,8),(13,30),(60,90),]
            hlines =None

            #% --------- Bayesian t test
            # NOTE: alternative is derived from the config 'tail' (greater/less for
            # tail=+/-1, two-sided for tail=0), consistent with the CBPT direction.
            # This is the tail-based version and it OVERWRITES the two-sided
            # bf_{config_setting}.pkl written by the group scripts
            # (pipeline_syn_ppc_GED_ROI_group.py / _power_dfc_GED_ROI_group.py /
            #  _ppc_PCA_VertROI_group.py), which save to the same path_group.
            from bayes_factor_fun import bayes_ttest
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
            bf10, pval = bayes_ttest(cond1_subft[:,:, cond_tidx], 
                                    cond2_subft[:,:, cond_tidx], 
                                paired=True, 
                                alternative=alternative, 
                                r=0.707, 
                                return_pval=True)
            
            # Handle numerical underflow:  
            # when evidence for H0 is extremely strong: BF10=0
            # In floating-point arithmetic, BF10 can underflow to 0. 
            # Here set a floor of 1e-10, which still represents decisive evidence against the alternative hypothesis.
            bf10 = np.maximum(bf10, 1e-10)  # floor at 1e-10

            if isinstance(bf10, str):
                bf10 = float(bf10)
            bf01 = 1 /bf10

            # 2 average over time points and frequency to get one bayes factor for the intersted time window
            bf_avg_band = {}
            bf_avg_band_df = pd.DataFrame()
            for band_name,band_range in zip(
                band_names,
                bands_range):
                fidx = [idx for idx,f in enumerate(freqs) if (f>=band_range[0])&(f<=band_range[1])]
                bf10_avg, pval_avg = bayes_ttest(
                    cond1_subft[:, fidx, :][:,:, cond_tidx].mean(axis=(1,2)), 
                    cond2_subft[:, fidx, :][:,:, cond_tidx].mean(axis=(1,2)), 
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
                    "cond1_content_name": cond1,
                    "cond2_content_name": cond2,
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
            csv_name = os.path.join(path_group_csv, f"bf_{config_setting}.csv")
            bf_avg_band_df.to_csv(csv_name, index=False)
            print('------------ save bayes factor file ' ,
                    '\n','file ',bf_name,
                    '\n','csv file ',csv_name,
                    '\n','config ',config_setting,
                    '\n',datetime.now(),
                                    )
            #% --------- plot bayes factor result
            with open(bf_name, 'rb') as pickle_file_load:
                loaded_bf = pickle.load(pickle_file_load)

            # ----- plot the bayes factor result
            for bf_type in ['bf01','bf10']:
                bf = loaded_bf[bf_type]
                pval = loaded_bf['pval']
                logbf = np.log10(bf)
                print('max logbf ',bf_type ,np.max(logbf))
                # ----- plot the bayes factor value
                try:
                    del data_ft
                except NameError:
                    pass
                for data_ft,plot_content,cbar_label,cmap in zip([logbf,pval],
                                                    [f'log{bf_type} of {value_type}',f'ttest pval of {value_type}'],
                                                    [f'log{bf_type}','ttest pval'],
                                                    ['BrBG','autumn_r']
                                                    ):
                    pdf_path      = os.path.join(path_group_plot, 'pdf',individual_ana,f'{bf_type}_values_{group_configfolder}',config_group)
                    jpg_path      = os.path.join(path_group_plot, 'jpg',individual_ana,f'{bf_type}_values_{group_configfolder}',config_group)
                    plotdata_path = os.path.join(path_group_plot, 'plotdata',individual_ana,f'{bf_type}_values_{group_configfolder}',config_group)

                    os.makedirs(pdf_path, exist_ok=True)
                    os.makedirs(jpg_path, exist_ok=True)  
                    os.makedirs(plotdata_path, exist_ok=True) 
                    if f'log{bf_type}' in plot_content:
                        ylim =(-np.max(abs(data_ft)),np.max(abs(data_ft)))
                        colorbar_hline =[1,np.log10(3)]
                        mask = bf > 10
                        mask_info = f'{bf_type}_10'
                    else:
                        ylim = (0,np.max(data_ft))
                        colorbar_hline =[0.05]
                        mask = pval < 0.05
                        mask_info = 'p_0.05'

                    bf_pic_name = (f"ft mask {bf_type} {config_setting} {plot_content} {mask_info}").replace(' ', '_')
                    fig_name_jpg = os.path.join(jpg_path,f'{bf_pic_name}.jpg')
                    if flag_no_newsave and os.path.exists(fig_name_jpg) and os.path.exists(fig_name_jpg.replace('jpg','pdf')):
                        print(f'{fig_name_jpg} already exists, skip plotting group mean plot')
                        del fig_name_jpg

                    title_name = f"{plot_content} {mask_info}"

                    ax,im = plot_matrix_pcolormesh(
                        plotdata_save_folder_path = plotdata_path,
                        flag_colorbar = 1,
                        colorbar_hline = None,#[1],#colorbar_hline,  # Xuan add
                        flag_return_image= 1,
                        x_coords= times_load[cond_tidx[0]:cond_tidx[-1]+1],
                        y_coords= freqs, 
                        data= data_ft, 
                        mask= mask, 
                        
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
                        hlines = hlines,
                        title = title_name, 
                        square_fig=False, 
                        dpi=300);
                    fig = ax.figure
                    fig.suptitle(bf_pic_name, y = 1.021,fontsize=14)
                    #%
                    fig.savefig(fig_name_jpg, 
                                bbox_inches='tight',dpi=300)#     
                    fig.savefig(fig_name_jpg.replace('jpg','pdf'), 
                                bbox_inches='tight',dpi=300)#

                # ----- plot the bayes mask result in raw value
                print(bf.shape)
                assert bf.shape == (len(freqs),len(cond_time)), 'error: bf shape not match'
                for criterion_name in [f'{bf_type}_10',f'{bf_type}_3','p_0.05']:
                    if criterion_name == f'{bf_type}_10':
                        criterion = bf > 10
                    if criterion_name == f'{bf_type}_3':
                        criterion = bf > 3
                    elif criterion_name == 'p_0.05':
                        criterion = pval < 0.05


                    if not np.any(criterion):
                        print('no significant point found for ',criterion_name)


                    pdf_path      = os.path.join(path_group_plot, 'pdf',individual_ana,f'{bf_type}_sig_{group_configfolder}',config_group)
                    jpg_path      = os.path.join(path_group_plot, 'jpg',individual_ana,f'{bf_type}_sig_{group_configfolder}',config_group)
                    plotdata_path = os.path.join(path_group_plot, 'plotdata',individual_ana,f'{bf_type}_sig_{group_configfolder}',config_group)

                    os.makedirs(pdf_path, exist_ok=True)
                    os.makedirs(jpg_path, exist_ok=True)  
                    os.makedirs(plotdata_path, exist_ok=True) 
                    del data_ft
                    for data_ft,plot_content,cbar_label in zip([cond1_ft_plot,
                                                                cond2_ft_plot,
                                                                conddif_ft_plot],
                                                    [f'{cond1} avg {value_type}',
                                                    f'{cond2} avg {value_type}',
                                                    f'{value_type} diff {cond1} {cond2}'],
                                                    [f'{value_type}',
                                                    f'{value_type}',
                                                    f'{value_type} difference'
                                                    ],):

                        print(f'max of {value_type}',np.max(data_ft))
                        print(f'min of {value_type}',np.min(data_ft))


                        bf_pic_name = (f"sigBFft {config_setting} {criterion_name} {plot_content}").replace(' ', '_')
                        fig_name_jpg = os.path.join(jpg_path,f'{bf_pic_name}.jpg')
                        if flag_no_newsave and os.path.exists(fig_name_jpg) and os.path.exists(fig_name_jpg.replace('jpg','pdf')):
                            print(f'{fig_name_jpg} already exists, skip plotting group mean plot')
                            del fig_name_jpg
                        title_name = f"sigBFft {criterion_name} {plot_content}"

                        # Create a grid of indices
                        mask = np.zeros_like(data_ft, dtype=bool)  # (freq x time)
                        mask[:, cond_tidx_plot] = criterion
                        
                        if 'diff' in plot_content:
                            ylim =(-np.max(abs(data_ft)),np.max(abs(data_ft)))
                            cmap = 'RdYlBu_r'
                        else:
                            cond_min = np.min([cond1_ft_plot,cond2_ft_plot])
                            cond_max = np.max([cond1_ft_plot,cond2_ft_plot])
                            ylim = (cond_min,cond_max)
                            cmap = 'Reds'  
                                
                        
                        ax,im = plot_matrix_pcolormesh(
                            plotdata_save_folder_path = plotdata_path,
                            flag_colorbar = 1,
                            flag_return_image= 1,
                            data= data_ft, 
                            x_coords= times_use,
                            y_coords= freqs,
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
                            hlines = hlines,
                            title = title_name, 
                            square_fig=False, 
                            dpi=300);
                        fig = ax.figure
                        fig.suptitle(bf_pic_name, y = 1.021,fontsize=14)
                        #%
                        fig.savefig(fig_name_jpg, bbox_inches='tight',dpi=300)#   
                        fig.savefig(fig_name_jpg.replace('jpg','pdf'),  bbox_inches='tight',dpi=300)#      
                                   


            #% cluster based permutation statistic testing
            clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
            blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
            cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")


            df_sigtable_name  = os.path.join(path_group, f"cbpt_clusterTable_{config_setting}.csv")
            df_siginfo_name   = os.path.join(path_group, f"cbpt_clusterDf_{config_setting}.csv")


            assert os.path.exists(cbpt_result_name), 'error: not exist cbpt_result_name'
            assert ((os.path.exists(clusterfile_name)) or  (os.path.exists(blankfile_name)) ), 'error: not exist clusterfile_name or blankfile_name'

            mask_ft = np.zeros_like(conddif_ft_plot, dtype=bool)  # (freq x time)
                    
            if os.path.exists(blankfile_name):
                #% ------- prepare for plotting group avg as no significant cluster found ---
                print('------------','start plot group result as no significant cluster found' ,
                    '\n',config_setting,
                    '\n',datetime.now(),)
                mask_ft = None
                plot_prefix= 'nosig'

                pdf_path      = os.path.join(path_group_plot, 'pdf',individual_ana,f'CBPT_nosig_{group_configfolder}',config_group)
                jpg_path      = os.path.join(path_group_plot, 'jpg',individual_ana,f'CBPT_nosig_{group_configfolder}',config_group)
                plotdata_path = os.path.join(path_group_plot, 'plotdata',individual_ana,f'CBPT_nosig_{group_configfolder}',config_group)

                os.makedirs(pdf_path, exist_ok=True)
                os.makedirs(jpg_path, exist_ok=True)  
                os.makedirs(plotdata_path, exist_ok=True) 
                
            else:
                plot_prefix= 'sig'
                pdf_path      = os.path.join(path_group_plot, 'pdf',individual_ana,f'CBPT_sig_{group_configfolder}',config_group)
                jpg_path      = os.path.join(path_group_plot, 'jpg',individual_ana,f'CBPT_sig_{group_configfolder}',config_group)
                plotdata_path = os.path.join(path_group_plot, 'plotdata',individual_ana,f'CBPT_sig_{group_configfolder}',config_group)

                os.makedirs(pdf_path, exist_ok=True)
                os.makedirs(jpg_path, exist_ok=True)  
                os.makedirs(plotdata_path, exist_ok=True) 
                
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
                
                # ----- get the mask for each cluster
                
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    # clusters[clu_idx] returns (freq_inds, time_inds) where:
                    #   freq_inds  → direct indices into the frequency array (no remapping needed)
                    #   time_inds  → indices RELATIVE to cond_tidx (the subset used for permutation test)
                    freq_inds, time_inds = clusters[clu_idx]

                    # map time_inds (relative to cond_tidx) → indices in full time array (times_load)
                    time_inds_full = np.array(cond_tidx)[time_inds]

                    # times_use_bool_in_full_time
                    # naming convention: [what we test]_bool_in_[reference]
                    #   what we test = times_use  (times_use_idx: the plot window timepoints)
                    #   reference    = full_time  (time_inds_full: absolute indices in times_load)
                    #
                    # Meaning: for each element of time_inds_full (reference/full time scale),
                    #          is it present in times_use_idx (the plot window)?
                    # True  → timepoint exists in plot window → keep both freq and time
                    # False → timepoint outside plot window  → discard both freq and time
                    times_use_bool_in_full_time = np.isin(time_inds_full, times_use_idx)


                    # use times_use_bool_in_full_time to filter BOTH freq and time in sync
                    freq_inds_in_use = freq_inds[times_use_bool_in_full_time]
                    time_inds_full_in_use = time_inds_full[times_use_bool_in_full_time]  # still in full-time-array space

                    # convert surviving time values → positions in times_use_idx
                    # np.searchsorted finds the insertion position = the index position
                    time_inds_in_use = np.searchsorted(times_use_idx, time_inds_full_in_use)

                    # STEP 4: mark significant cells in the mask
                    # mask_ft shape : 
                    #   conddif_ft_plot = conddif_ft[:, times_use_idx]  ← already sliced to plot window
                    #   so mask_ft is in PLOT time scale (times_use), NOT full time scale (times_load)
                    #   time_inds_in_use are therefore indices into times_use_idx, not times_load
                    if len(time_inds_in_use) > 0:
                        mask_ft[freq_inds_in_use, time_inds_in_use] = True


            #% --------- plot group average with cluster mask (if any) -----------
            try:
                del data_ft
            except NameError:
                pass
            for data_ft,plot_content,cbar_label in zip(
                                [cond1_ft_plot,
                                    cond2_ft_plot,
                                    conddif_ft_plot],
                                    [f'{cond1} avg {value_type}',
                                    f'{cond2} avg {value_type}',
                                    f'{value_type} diff {cond1} {cond2}'],
                                    [f'{value_type}',
                                    f'{value_type}',
                                    f'{value_type} difference'
                                        ],):

                if flag_same_clim:
                    if 'diff' in plot_content:
                        ylim =clim_preset_diff
                        cmap = 'RdYlBu_r'
                    else:
                        cond_min = np.min([cond1_ft_plot,cond2_ft_plot])
                        cond_max = np.max([cond1_ft_plot,cond2_ft_plot])
                        ylim = clim_preset_raw 
                        cmap = 'Reds'    

                else:
                    if 'diff' in plot_content:
                        ylim =(-np.max(abs(data_ft)),np.max(abs(data_ft)))
                        cmap = 'RdYlBu_r'
                    else:
                        cond_min = np.min([cond1_ft_plot,cond2_ft_plot])
                        cond_max = np.max([cond1_ft_plot,cond2_ft_plot])
                        ylim = (cond_min,cond_max)   
                        cmap = 'Reds'    
                
                # pic name
                groupmean_pic_name = f'{plot_prefix} GroupFT {config_setting} {plot_content} {len(sub_list)}sub'
                fig_name_jpg = os.path.join(jpg_path,f'{groupmean_pic_name}.jpg')
                if flag_no_newsave and os.path.exists(fig_name_jpg) and os.path.exists(fig_name_jpg.replace('jpg','pdf')):
                    print(f'{fig_name_jpg} already exists, skip plotting group mean plot')
                    del fig_name_jpg
                    continue

                plotdata_path_groupmean_pic = os.path.join(plotdata_path, groupmean_pic_name.replace(' ', '_'))
                os.makedirs(plotdata_path_groupmean_pic, exist_ok=True)
                
                ax,im = plot_matrix_pcolormesh(
                    flag_sharp_contour= flag_sharp_contour,
                        plotdata_save_folder_path = plotdata_path_groupmean_pic,
                        flag_colorbar = 1,
                        flag_return_image= 1,
                         x_coords= times_use,
                         y_coords= freqs,
                        # x0   = x0, 
                        # x_end= x_end, 
                        # y0   = y0, 
                        # y_end= y_end,  
                        data= data_ft, 
                        mask=mask_ft, 
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
                        hlines = hlines,
                        title = plot_content, 
                        square_fig=False, 
                        dpi=300,
                        log_scale_y = True if flag_y_logscale else False);

                
                fig = ax.figure
                fig.suptitle(groupmean_pic_name, y = 1.021,fontsize=14)

                #%
                fig.savefig(fig_name_jpg,  bbox_inches='tight',dpi=300)#
                fig.savefig( fig_name_jpg.replace('jpg','pdf'),  bbox_inches='tight',dpi=300)#  
                           


                # --------- plot of only sig freq range -----------
                if plot_prefix == 'sig':
                    fig_name_jpg = os.path.join(jpg_path,f'detail_{groupmean_pic_name}.jpg')
                    if flag_no_newsave and os.path.exists(fig_name_jpg) and os.path.exists(fig_name_jpg.replace('jpg','pdf')):
                        print(f'{fig_name_jpg} already exists, skip plotting detail plot')
                        del fig_name_jpg
                        continue

                    sig_freqs = np.array(freqs)[mask_ft.any(axis=1)]  # Frequencies with ANY sig time point

                    # hlines_array = np.sort(np.array(hlines))
                    # # np.searchsorted finds where to INSERT sig_freqs.min() to keep array sorted
                    # # subtract 1 to get the element BEFORE the insertion point
                    # # max(0, ...) ensures don't get negative indices
                    # idx_low = max(0, np.searchsorted(hlines_array, sig_freqs.min()) - 1)
                    # idx_high = min(len(hlines_array) - 1, np.searchsorted(hlines_array, sig_freqs.max()))
                    # covering_hlines = [hlines_array[idx_low], hlines_array[idx_high]]

                    # # Get intermediate hlines (more concise condition)
                    # hlines = hlines_array[idx_low+1:idx_high] if idx_high - idx_low > 1 else None


                    freq_range = sig_freqs
                    if hlines:
                        covering_hlines = np.array(hlines)
                        covering_hlines = np.array(hlines)[(np.array(hlines) >= freq_range[0]) & (np.array(hlines) <= freq_range[1])]
                    else:
                        covering_hlines = None

                    # Frequency index selection
                    freq_use_index = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[-1]))[0]

                    ax,im = plot_matrix_pcolormesh(
                        flag_sharp_contour= flag_sharp_contour,
                            plotdata_save_folder_path = plotdata_path_groupmean_pic,
                            flag_colorbar = 1,
                            flag_return_image= 1,
                            x_coords= times_load[cond_tidx[0]-1:cond_tidx[-1]+2],
                            y_coords= freq_range,
                            # x0   = times_load[cond_tidx[0]-1], 
                            # x_end= times_load[cond_tidx[-1]+1], 
                            # y0   = freq_range[0], 
                            # y_end= freq_range[-1],  
                            data= data_ft[freq_use_index,:][:,[cond_tidx[0]-1] + cond_tidx +[cond_tidx[-1]+1]], 
                            mask= mask_ft[freq_use_index,:][:,[cond_tidx[0]-1] + cond_tidx +[cond_tidx[-1]+1]], 
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
                            hlines = covering_hlines,
                            title = plot_content, 
                            square_fig=False, 
                            dpi=300,
                            log_scale_y = True if flag_y_logscale else False);


                    fig = ax.figure
                    fig.suptitle(f'detail {groupmean_pic_name}', y = 1.021,fontsize=14)
                    #%
                    
                    fig.savefig(fig_name_jpg,  bbox_inches='tight',dpi=300)#     
                    fig.savefig(fig_name_jpg.replace('jpg','pdf'), bbox_inches='tight',dpi=300)#  
                                   


            print('------------','finish plot group result' ,
                        '\n',config_setting,
                        '\n',datetime.now(),
                        )

    # %%
