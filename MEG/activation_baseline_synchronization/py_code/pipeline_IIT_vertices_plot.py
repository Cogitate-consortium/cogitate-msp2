'''
Group-level (IIT) plotting for the selected VertROIs (from pipeline_IIT_vertices_select.py).

Part 1: build/load a per-subject VertROI table (which subjects have vertices in
each VertROI and how many), saved to _label_info/ses-v2-vertROI-sub.csv, and
print summary stats (% subjects with vertices, mean/std/max n vertices).

Part 2: for each VertROI, load the cbpt result (significant vertices) + the
pre-saved source power, and extract the two conditions' log-power over those
significant vertices, per subject.

Part 3: plot the group-mean log-power time course (vertices reduced by RMS) with
within-subject CI for cond1 vs cond2, plus a per-participant subplot.

Config folder: /config_files/pipeline_IIT_vertices_plot/
    vertFFface.json
    vertFFobje.json
  each lists band_names / saved_roi_list_names / selectVert_configfiles.
'''

#%% import
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# for show time
from datetime import datetime
import numpy as np
import pandas as pd

# for save data
import pickle

# for function parameter use path
from pathlib import Path

from EXP1_help_functions import (BidsPath)

from _help_functions import (general_param,
                             get_file_name,
                             path_allana,
                             bids_root,
                                 read_configfile,
                                 read_labels_exp2,# for load VertROI labels
                                    get_trl_indices,# for select condition trials
                                 )

from _help_functions import get_sublist,Project_Dir
from _help_functions import (compute_mean_exp2,
                                 convert_to_ci_1D,
                                 apply_lowpass_filter,
                                 colordict)
from cog_plot import (
                      plot_time_series_sigline,
                      )
from _help_module import (_slug,_plot_subject_subplots)

import mne_bids


def list_filetype(directory, extension=".json"):
    try:
        json_files = [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(extension)]
        return json_files
    except FileNotFoundError:
        return f"Error: {directory} is not a valid directory."

#%% setting test 
flag_test = False
#%% setting path    
vertices_avg_type = "rms"  
individual_ana = "pipeline_IIT_vertices_select"
group_ana = 'pipeline_IIT_vertices_plot'
group_configfolder = 'pipeline_IIT_vertices_plot'
folder_config   = os.path.join(Project_Dir,'config_files',group_configfolder)



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

pdf_dir = Path(os.path.join(path_group_plot, 'pdf'))
jpg_dir = Path(os.path.join(path_group_plot, 'jpg'))
data_dir = Path(os.path.join(path_group_plot, 'plotdata'))

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(jpg_dir, exist_ok=True)  
os.makedirs(data_dir, exist_ok=True)  

# Get the different configfile
config_files = list_filetype(folder_config, extension=".json")
print('--------',len(config_files), 'configfile')


# for group_configfile in config_files:
group_configfile = config_files  # for test
# Read the configfile file:
param_group = read_configfile(group_configfile)
config_filename = get_file_name(group_configfile)

# Unpack the general information
roi_params = general_param['roi_params_dict']

#%% create a dataframe for the VertROI information of each subject
# first load the sublist who have good behavior data
checksub_behavior = os.path.join(bids_root, 'derivatives', 'qcs', 'ses-v2-analysis-subs.csv')
df_checksub_bh = pd.read_csv(checksub_behavior)

df_checksub_bh ['allcondi_false'] = df_checksub_bh.apply(lambda x: 
    all(x[col]==False for col in df_checksub_bh.columns if '_min_' in col),axis = 1)
max_sub = list(np.sort(df_checksub_bh[df_checksub_bh['allcondi_false']==False]['sub_code'].unique()))
print('behavior check max sub number',len(max_sub))

# create a dataframe to save the VertROI information of each subject
roi_vert_params = general_param['VertROI_params_dict']
deriv_root_VertROI_info  = os.path.join(path_allana, individual_ana,'_label_info')   
os.makedirs(deriv_root_VertROI_info, exist_ok=True) 

checksub_vertROI =  os.path.join(deriv_root_VertROI_info,'ses-v2-vertROI-sub.csv')   

if not os.path.exists(checksub_vertROI):
    df_check = pd.DataFrame()
    for subidx,subject in enumerate(max_sub):
        roi_dict = {}
        for roi in roi_vert_params.keys():
            lb_use = read_labels_exp2(
                bids_paths = BidsPath(subject_id = subject, 
                                        visit_id   = 'V2'), 
                parc       = roi_vert_params[roi]['parc'], 
                labels_list= roi_vert_params[roi]['labels_list'],
                rois_list  = roi_vert_params[roi]['rois_list'], 
                combine_labels_list=True, 
                merge_hemi=True
                )  

            roi_dict[roi_vert_params[roi]['label_name']] = len(lb_use)>0
            if len(lb_use)>0:
                roi_dict[f'{roi_vert_params[roi]['label_name']}_nvert'] = len(lb_use[roi])
            else:
                roi_dict[f'{roi_vert_params[roi]['label_name']}_nvert'] = 0
                
            
        a = pd.DataFrame([roi_dict])
        a.insert(0,'subject',subject)
        df_check = pd.concat([df_check,a])
    df_check.to_csv(checksub_vertROI,index = False)   
else :
    df_check= pd.read_csv(checksub_vertROI)
    
    
# %% show the VertROI information 
df_check_long = df_check.melt(
    id_vars = 'subject',
    value_vars=[x for x in df_check.columns if x!='subject']
)
df_check_long['vert_type'] = df_check_long.apply(lambda x:
    x['variable'].split('_')[0], axis = 1)

df_check_long['measure_type'] =df_check_long.apply(lambda x:
  'nvert'  if 'nvert'  in x['variable'] else 'notempty', axis = 1)

df_check_2d = df_check_long.pivot(
    index = ['subject','vert_type'],
    columns='measure_type',
    values='value'
).reset_index()
print('-------- total subject',df_check.subject.nunique())
print('--------percentage of subjects having vertices\n',df_check_2d.groupby('vert_type')['notempty'].value_counts(normalize=True))
print('--------number of subjects having vertices\n',df_check_2d.groupby('vert_type')['notempty'].value_counts())
print('--------mean number of vertices\n',df_check_2d[df_check_2d.notempty ==1].groupby('vert_type')['nvert'].mean())
print('--------std number of vertices\n',df_check_2d[df_check_2d.notempty ==1].groupby('vert_type')['nvert'].std())
print('--------min number of vertices\n',df_check_2d[df_check_2d.notempty ==1].groupby('vert_type')['nvert'].max())


#%% load the cbpt results and extract the power of the selected vertices   
VertROI_logpower_dict = {}
for individual_configfile_name in param_group['selectVert_configfiles']:
    individual_configfile = os.path.join(Project_Dir,'config_files',
                                         individual_ana,
                                individual_configfile_name )
    para_selectVert = read_configfile(individual_configfile)
    sublist = get_sublist(individual_configfile)
    if flag_test:
        sublist = sublist[0:3]
    print('-------- load individual configfile',individual_configfile_name)
    
    # Unpack the Select Vert parameters
    band_name = para_selectVert['band_name']
    roi = para_selectVert['roi']
    saved_roi_list_name = para_selectVert['saved_roi_list_name'] # 'xx_highergamma' or 'xx_loweralpha'

    cond1_name    = para_selectVert['cond1_name']
    cond2_name    = para_selectVert['cond2_name']
    cond1_content = para_selectVert['cond1_content']
    cond2_content = para_selectVert['cond2_content']
    cond1_tw = para_selectVert['cond1_tw']
    cond2_tw = para_selectVert['cond2_tw']
    compare_trltypes_select_columns =para_selectVert['compare_trltypes_select_columns']
    epoch_data = general_param['epoch_data_dict'][para_selectVert['epoch_data_name']]
    cbpt_setting = general_param['cbpt_params_dict'][para_selectVert['cbpt_params_name']]

    p_cluster_forming= cbpt_setting['p_cluster_forming']
    out_type = cbpt_setting['out_type']
    n_permutations =  cbpt_setting['n_permutations'] 
    
    #% get the VertROI vertices and extract power of the two conditions
    VertROI_logpower_dict[band_name] = {}
    for subi,subject in enumerate(sublist)  :

        # ---------- read cbpt results
        deriv_root = os.path.join(path_allana, individual_ana)
        deriv_root_label  = os.path.join(deriv_root,'_label')    
        bids_path = mne_bids.BIDSPath(
                root=deriv_root, 
                subject= subject, 
                session= epoch_data['exp_id'],  
                datatype='meg',  
                task=epoch_data['task_id'],
                suffix=f"{saved_roi_list_name}",
                extension='.pkl',
                check=False)
        dir_analyse = os.path.dirname( bids_path.fpath)
            

        savefile_name = bids_path.fpath
        blankfile_name = str(bids_path.fpath).replace('.pkl','_nosig.pkl')

        if os.path.exists(blankfile_name):
            print(f"{subi+1} of {len(sublist)} ------------ No detected VertROI for " ,subject,
                config_filename,
                '\n',datetime.now(),  )
            VertROI_logpower_dict[band_name][subject]={}
        else:
            print(f"{subi+1} of {len(sublist)} ------------ load VertROI for " ,subject,
                  config_filename,
                  '\n',datetime.now(),  )
            # ---------------- get comparison type based on time windows and content
            time_window_is_different = (cond1_tw != cond2_tw)
            content2_is_subset_of_content1 = all(item in cond1_content for item in cond2_content)
            flag_withintrl_cmp = time_window_is_different and content2_is_subset_of_content1

            time_window_is_same = (cond1_tw == cond2_tw)
            content_is_not_subset = not content2_is_subset_of_content1 
            flag_betwtrl_cmp = time_window_is_same and content_is_not_subset
            assert np.logical_or(flag_withintrl_cmp, flag_betwtrl_cmp) ,'error:flag not right'
            

            # ---------------- load presaved power get roi_vidx_lr 
            # load presaved power 
            deriv_root_presaved_power = os.path.join(
                path_allana, 
                "pipeline_presaved_power")
            bids_path = mne_bids.BIDSPath(
                    root=deriv_root_presaved_power, 
                    subject= subject, 
                    session= epoch_data['exp_id'],  
                    datatype='meg',  
                    task=epoch_data['task_id'],
                    suffix=f'{band_name}_power',
                    extension='.pkl',
                    check=False)
            pklfile_name = bids_path.fpath
            with open(pklfile_name, 'rb') as pickle_file_load:
                loaded_data = pickle.load(pickle_file_load)
            print(loaded_data.keys())
        
            times = loaded_data['times']
            pow_tlvt = loaded_data['pow_tlvt']
            
            # get the name of label used in label_dict: 'all' for multiple labels; or the only label name if one label
            lb_list = read_labels_exp2(
                        bids_paths = BidsPath(subject_id = subject, 
                                            visit_id   = epoch_data['exp_id']), 
                        parc       = roi_params[roi]['parc'], 
                        labels_list= roi_params[roi]['labels_list'],
                        rois_list  = roi_params[roi]['rois_list'], 
                        combine_labels_list=True, 
                        merge_hemi=True
                        )  
            if len(lb_list) == 1:
                label_name_load = list(lb_list.keys())[0]
            else:
                label_name_load = [x for x in lb_list.keys() if 'all' in x][0]
            
            # based on loaded power results get the roi vertices index from loaded power
            roi_vidx_lr = loaded_data ['label_dict'][roi]['vidx_dic_allowoverlap'][label_name_load]


            # ----------------  get VertROI vertices info
            with open(savefile_name, 'rb') as pickle_file_load:
                cluster_stats = pickle.load(pickle_file_load)
            # Extract cluster information
            T_obs, clusters, cluster_p_values, H0 = cluster_stats
            good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
            print('------------ load ',len(good_cluster_inds),'good_cluster' ,subject,
            ' ',config_filename,
            ' ',datetime.now(),
            )
            
            sig_vidx_lr = []
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                # unpack cluster information, get unique indices
                if flag_withintrl_cmp:
                    space_inds = clusters[clu_idx]
                if flag_betwtrl_cmp:   
                    time_inds, space_inds = clusters[clu_idx]
                # get unique vertex indices
                v_inds = np.unique(space_inds)
                
                # get index for each cluster
                cluster_vidx_lr = [x for idx,x in enumerate(roi_vidx_lr) if idx in v_inds]
                sig_vidx_lr = sig_vidx_lr + cluster_vidx_lr
                
            sig_vidx_lr = list(set(sig_vidx_lr))

            # ---------------- extract power of the two conditions
            df_epoch= loaded_data ['epochs_metadata'].copy().fillna('no').reset_index(drop = True)
            
            cond1_tlidx = get_trl_indices(df_epoch,compare_trltypes_select_columns, cond1_content)
            cond2_tlidx = get_trl_indices(df_epoch,compare_trltypes_select_columns, cond2_content)

            cond1_tlvt = pow_tlvt[cond1_tlidx,:,:][:,sig_vidx_lr,:]
            cond2_tlvt = pow_tlvt[cond2_tlidx,:,:][:,sig_vidx_lr,:]
            
            VertROI_logpower_dict[band_name][subject]={
                cond1_name:np.log10(cond1_tlvt).mean(axis=0),
                cond2_name:np.log10(cond2_tlvt).mean(axis=0),
            }

#%%   3 plot the power of the selected vertices
#% plot setting
flag_preset_ylim = 0
ylim_erf = (-0.1,0.45)
ylim_erf_seplabel = (-0.1,0.5)
ylim_erfdif = (-0.1,0.1)
xtick_interval =0.5

if 'baseline' in group_configfolder:
    times_use_w = [-0.9,1.01]
else:
    times_use_w = [-0.51,1.01]
    
times_use_idx = [idx for idx,t in enumerate(loaded_data['times']) if (t<= times_use_w[1])&(t>= times_use_w[0])]
times_use_plot = loaded_data['times'][times_use_idx] 
linewidth_list =[2,2,4]
linestyle_list=['-' ,'-','--']

vlines           = [0] + [float(x) for x in cond1_tw if x!= None ]
vlines_colors    = ['k'] + ['k','k']
vlines_linestyle =  ['-'] +['--','--']
vlines_linewidth = [1] + [1,1]#%%    
fs = loaded_data['sfreq']

for band_name,saved_roi_list_name, selectVert_configfile in zip(
                                            param_group['band_names'],
                                            param_group['saved_roi_list_names'],
                                            param_group['selectVert_configfiles']):
    if  all([len(v)== 0 for v in VertROI_logpower_dict[band_name].values()]):
        print('no subject have VertROI for ', band_name, saved_roi_list_name)
        continue
    # --- reduce vertices -> t
    if vertices_avg_type.lower() == "rms":
        # Define the reduction function as RMS
        reduction_func = lambda data: np.sqrt(np.mean(data ** 2, axis=0))
        ylabel_reduce = "RMS"
    elif vertices_avg_type.lower() == "mean":
        # Define the reduction function as MEAN
        reduction_func = lambda data: data.mean(axis=0)
        ylabel_reduce = "Mean"
    else:
        raise ValueError("vertices_avg_type must be 'rms' or 'mean'.")
    
    # Use the defined function within the list comprehension
    cond_names = [cond1_name,cond2_name]
    subcondt = np.array(
        [
            # Outer list element (Participant)
            [
                # Inner list element (Condition) -- each element is the reduced time series
                reduction_func(VertROI_logpower_dict[band_name][participant][cond]) # vt

                # Inner loop: Conditions
                for cond in cond_names
            ]
            
            # Outer loop: Participants
            for participant in VertROI_logpower_dict[band_name].keys() if len(VertROI_logpower_dict[band_name][participant]) > 0
        ]
    )
    
    title = f"LogPower V{vertices_avg_type} {saved_roi_list_name} {subcondt.shape[0]}Sub"
    # --- group mean & CI
    condt, condt_ci_raw = compute_mean_exp2(
        subcondt, 
        axis_cond=1, 
        axis_sub=0, 
        zscore=False, design="within"
    )
    condt_ci = convert_to_ci_1D(condt, condt_ci_raw)

            
    # --- low-pass + slice to requested window ---
    condt_filt = apply_lowpass_filter(condt, fs=fs)                 # (Cond, T)
    condt_ci_filt = apply_lowpass_filter(condt_ci, fs=fs)           # (Cond, T, 2)
    plat_condt = condt_filt[:, times_use_idx]                               
    plat_condt_ci = condt_ci_filt[:, times_use_idx, :]

    # --- plot ---
    conds_name = [cond1_name, 
                cond2_name]
    conds_colors = [colordict[cond1_name], 
                    colordict[cond2_name]]
    ylabel = f"{band_name}Power ({ylabel_reduce})"
    ylim = ylim_erf if flag_preset_ylim else None
    window_list = None
    ax = plot_time_series_sigline(
        plotdata_save_folder_path=os.path.join(data_dir, _slug(title)),
        data=plat_condt,
        err=None,
        ci_1D=plat_condt_ci,
        t0  = times_use_plot[0],
        tend= times_use_plot[-1],
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
        sig_hatchedpatterns=window_list,
        sig_hatchedpattern_dataidx=[0, 1],
        dpi=300,
    )

    # legend & save
    h, l = ax.get_legend_handles_labels()
    fig = ax.figure
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)
    fig.savefig(jpg_dir / f"{_slug(title)}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{_slug(title)}.pdf", bbox_inches="tight", dpi=300)

    # --- optional: plot each participant (Cond × Sub × T) ---
    flag_plot_each_participant=1
    if flag_plot_each_participant:
        title_each = f"Each Participant {title}"

        condsubt_use = np.transpose(subcondt, (1, 0, 2))[:, :, times_use_idx]  # (Cond, Sub, T)
        fig_each, ax_each = _plot_subject_subplots(
            condsubt_use=condsubt_use,
            times_use=times_use_plot,
            title=title_each,
            cond1_content_name=cond1_name,
            cond2_content_name=cond2_name,
            palette=conds_colors,
            y_suptitle = 1.2,
            vlines=vlines,
            vlines_colors=vlines_colors,
            vlines_linestyle=vlines_linestyle,
            vlines_linewidth=vlines_linewidth,
        )
        fig_each.savefig(jpg_dir / f"{_slug(title_each)}.jpg", bbox_inches="tight", dpi=300)
        fig_each.savefig(pdf_dir / f"{_slug(title_each)}.pdf", bbox_inches="tight", dpi=300)

# %%

   