'''
Individual-level (IIT): select the "qualified" vertices in an ROI via a
cluster-based permutation test on source band power, and save them as a VertROI
label for the subject.

Inputs (per subject + config): pre-saved source info (inverse + example stc +
src adjacency) from pipeline_presave_sourceinfo, and source band power from
pipeline_presaved_power (pow_tlvt = trial x vertex x time).

Two modes, auto-detected from the config's time windows / contents:
- within-trial (active vs baseline window, same trials): permutation_cluster_1samp_test
  on log10(cond1/cond2) power over ROI vertices -> "responsive" vertices.
- between-trial (two trial types, same window): permutation_cluster_test
  (Welch df, ttest_ind_nop) over time x vertices -> content-selective vertices.

If significant clusters are found: save cluster_stats + create/save the VertROI
label (create_save_label_from_vert, named saved_roi_list_name); otherwise save a
*_nosig.pkl blank. The VertROI name is unique per config, so it is used directly
as the label name.

Config folder: /config_files/pipeline_IIT_vertices_select/
    vertFFface_alpha/gamma.json       (face-selective)
    vertFFobje_alpha/gamma.json       (object-selective)
    vertFFstiVbla_alpha/gamma.json    (stimulus vs blank)
    vertV1V2actVbas_alpha/gamma.json  (V1/V2 active vs baseline)
  each sets roi, band_name, saved_roi_list_name, cond1/cond2_content + _tw, tail, cbpt_params_name.
'''

#%% import
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime
# for statistic testing
import scipy
# for save data
import pickle
from EXP1_help_functions import (BidsPath)

from _help_functions import (general_param,
                             path_allana,
                                 turn_sub_code_to_num,
                                 read_configfile,
                                 read_labels_exp2,
                                    get_trl_indices,# for select condition trials
                                    create_save_label_from_vert,# create and save label from vertices
                                    welch_df,# specific for between-trial comparison
                                    ttest_ind_nop# specific for between-trial comparison
                                 )

import numpy as np

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
    args.subject = 'SA127'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_IIT_vertices_select" / "vertV1V2actVbas_gamma.json"

else:
    parser = argparse.ArgumentParser(
        description="Implements analysis of source erf for experiment2")
    parser.add_argument('--subject', type=str, default=None,
                    help="Name of the subject")
    parser.add_argument('--configfile', type=str, default=None,
                        help="configfile file for analysis parameters (file name + path)")
    args = parser.parse_args()

#%% load configure and setting path
subject = args.subject
configfile  = args.configfile

# get the analyse name and configfile name to create folder to save
config_filename_with_ext = os.path.basename(configfile)
config_filename, _ = os.path.splitext(config_filename_with_ext)
print('--------- start ',config_filename)

# script_name = os.path.basename(os.path.dirname(configfile))
script_name = os.path.basename(__file__).replace('.py','')
         

# Read the configfile file:
param = read_configfile(configfile)
    
epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
cbpt_setting = general_param['cbpt_params_dict'][param['cbpt_params_name']]
roi_params = general_param['roi_params_dict']


band_name = param['band_name']
roi = param['roi']
saved_roi_list_name = param['saved_roi_list_name'] # 'xx_highergamma' or 'xx_loweralpha'

# Unpack the parameters
cond1_content = param['cond1_content']
cond2_content = param['cond2_content']
cond1_tw = param['cond1_tw']
cond2_tw = param['cond2_tw']
compare_trltypes_select_columns =param['compare_trltypes_select_columns']
tail = param['tail']  

p_cluster_forming= cbpt_setting['p_cluster_forming']
out_type = cbpt_setting['out_type']
n_permutations = cbpt_setting['n_permutations'] # 1000 if flag_test else 



# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          "test")
else:
    deriv_root = os.path.join(path_allana, script_name)
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
os.makedirs(os.path.dirname(savefile_name), exist_ok=True) 
os.makedirs(deriv_root_label, exist_ok=True) 

#%% get the results
if (os.path.exists(savefile_name)) or  (os.path.exists(blankfile_name))  :
    print('------------','exist subject' ,subject,
            config_filename,
            '\n',datetime.now(),
    )
    if os.path.exists(savefile_name):
        try:
            with open(savefile_name, 'rb') as pickle_file_load:
                loaded_data = pickle.load(pickle_file_load)
            print('------------','Successfully loaded' ,subject,
                config_filename,
                '\n',datetime.now(),
                    )    
        except Exception :
            # Print the error for debugging purposes (optional)
            print('------------','Error loading' ,savefile_name,
                config_filename,
                '\n',datetime.now(),
                    )
            pass
else:

    print('------------','start subject' ,subject,
            config_filename,
            '\n',datetime.now(),
            )
    
    # ----------------  load example source info for getting vertices index
    deriv_root_sourceinfo = os.path.join(
        path_allana, 
        "pipeline_presave_sourceinfo")
    # filename of stc example
    bids_path_stc = mne_bids.BIDSPath(
            root=deriv_root_sourceinfo, 
            subject= subject, 
            session= epoch_data['exp_id'],  
            datatype='meg',  
            task=epoch_data['task_id'],
            suffix='example_stc',
            check=False)
    stc_filename = bids_path_stc.fpath 
    stc_exp = mne.read_source_estimate(stc_filename)

    # filename of inv
    bids_path_inv = mne_bids.BIDSPath(
            root=deriv_root_sourceinfo, 
            subject= subject, 
            session= epoch_data['exp_id'],  
            datatype='meg',  
            task=epoch_data['task_id'],
            suffix='inv',
            extension='.fif',
            check=False)
    inv_filename = bids_path_inv.fpath
    inv = mne.minimum_norm.read_inverse_operator(inv_filename)
    
    #  from inv get src to get adjacency matrix
    src = inv['src']
    adjacency_src_inv = mne.spatial_src_adjacency(src)
    
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
        print('------------','there is only one label',label_name_load,'\n',datetime.now())
    else:
        label_name_load = [x for x in lb_list.keys() if 'all' in x][0]
    
    # ---------------- load presaved power 
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
    print('------------','Successfully loaded power' ,subject,
        config_filename,
        '\n',datetime.now(),
            )
   
    print(loaded_data.keys())
    times = loaded_data['times']
    pow_tlvt = loaded_data['pow_tlvt']
    
    # ------- construct the roi adjacency matrix
    # based on loaded power results get the roi vertices index from loaded power
    roi_vidx_lr = loaded_data ['label_dict'][roi]['vidx_dic_allowoverlap'][label_name_load]
    # based on roi vertices index from loaded power to get adjacency matrix for roi
    adjacency_roi = adjacency_src_inv.tocsr()[roi_vidx_lr][:, roi_vidx_lr].tocoo()

    # ------- construct the two conditions
    df_epoch= loaded_data ['epochs_metadata'].copy().fillna('no').reset_index(drop = True)
    
    cond1_tlidx = get_trl_indices(df_epoch,compare_trltypes_select_columns, cond1_content)
    cond2_tlidx = get_trl_indices(df_epoch,compare_trltypes_select_columns, cond2_content)

    cond1_tidx = [idx for idx,x in enumerate( times) if (x>=cond1_tw[0])&(x<=cond1_tw[1])]
    cond2_tidx = [idx for idx,x in enumerate( times) if (x>=cond2_tw[0])&(x<=cond2_tw[1])]
    
    cond1_tlvt = pow_tlvt[cond1_tlidx,:,:][:,roi_vidx_lr,:][:,:,cond1_tidx]
    cond2_tlvt = pow_tlvt[cond2_tlidx,:,:][:,roi_vidx_lr,:][:,:,cond2_tidx]
    
    # ---------------- Decide the statistical comparison type based on time windows and content
    # 1. Flag for WITHIN-TRIAL comparison: 
    #- Time windows must be DIFFERENT.
    time_window_is_different = (cond1_tw != cond2_tw)
    #- Content 2 is a subset of Content 1 (e.g., comparing a baseline window to an active window within the same trials)
    content2_is_subset_of_content1 = all(item in cond1_content for item in cond2_content)

    flag_withintrl_cmp = time_window_is_different and content2_is_subset_of_content1

    # 2. Flag for BETWEEN-TRIAL comparison:
    #- Time windows must be the SAME.
    time_window_is_same = (cond1_tw == cond2_tw)
    #- Content is NOT a subset (i.e., comparing two different trial types at the same time window)
    content_is_not_subset = not content2_is_subset_of_content1 

    flag_betwtrl_cmp = time_window_is_same and content_is_not_subset
    print('flag_withintrl_cmp',flag_withintrl_cmp,
          '\nflag_betwtrl_cmp',flag_betwtrl_cmp)
    assert np.logical_or(flag_withintrl_cmp, flag_betwtrl_cmp) ,'error:flag not right'
    

    # ----- with-in trial compare
    # responsiveness activity tw VS baseline tW
    # for each trial, get the mean power of activity tw, mean power of baseline tw
    # paird group is assumed: one group - other group
    # vertices as the dimension to find clusters
    if flag_withintrl_cmp:
        logdifpow_tlv = np.log10(
                    np.mean(cond1_tlvt, axis = 2)  /\
                    np.mean(cond2_tlvt, axis = 2) 
                    )
        print('------------',f'start with-in trial cluster test {n_permutations} ' ,subject,
            config_filename,
            '\n',datetime.now(),
            )
        # setting for permutation_cluster_1samp_test(rather than permutation_cluster_test)
        X_observ_clulsterdim =  logdifpow_tlv
        v_adjacency = adjacency_roi
        n_observations = logdifpow_tlv.shape[0] 
        print('------------','n_permutations ',n_permutations,'input ',logdifpow_tlv.shape[0])
        
        
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
            
        cluster_stats= mne.stats.permutation_cluster_1samp_test(
            X = X_observ_clulsterdim, 
            threshold     = t_thresh, 
            out_type      = out_type, 
            n_permutations= n_permutations, 
            tail          = tail,    # tail is 0, the statistic is thresholded on both sides of the distribution.
            stat_fun      = None,    # None (the default), uses mne.stats.ttest_1samp_no_p which comparing the result against 0
            adjacency     = v_adjacency, #If None, a regular lattice adjacency is assumed, connecting each location to its neighbor(s) along the last dimension of X (or the last two dimensions if X is 2D). 
            n_jobs        = None, 
            seed          = np.random.default_rng(seed=turn_sub_code_to_num(subject)) , 
            max_step      =1, 
            exclude       =None, 
            step_down_p   =0, 
            t_power       =1, 
            check_disjoint=False, 
            buffer_size=1000, 
            verbose= False
                        )
        T_obs, clusters, cluster_p_values, H0 = cluster_stats
        good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
        print(f'good_cluster_inds (<({p_cluster_forming}) ',len(good_cluster_inds) )
        print(cluster_p_values)
        if len(good_cluster_inds) == 0:
            #-------- 2 with-in trial save cluster_stats data
            with open(blankfile_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
                
            print('------------','no good_cluster save blank' ,subject,
            config_filename,
            '\n',datetime.now(),
            )        
        else:
            #-------- 2 with_in_trl save cluster_stats data
            with open(savefile_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
                
            print('------------ save stat of ',len(good_cluster_inds),'good_cluster' ,subject,
            config_filename,
            '\n',datetime.now(),
            )
            
            # ----------------  save significant vertices as label
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
            
            stc = stc_exp.copy()
            stc.data = stc.data[:, :1]
            new_data = np.zeros_like(stc.data)
            new_data[sig_vidx_lr, 0] = 1 
            stc.data = new_data
            flag_sigl,flag_sigr,sig_vidx_l,sig_vidx_r \
                = create_save_label_from_vert(
                stc             = stc,
                src             = src,
                sig_vidx_lr     = sig_vidx_lr,
                flag_get_label = 1,
                labelaveDir          = deriv_root_label,
                groupdata_name       = saved_roi_list_name,
                subdata_name         = f"{saved_roi_list_name} {subject}",
                subject              = subject,
            )
            print('------------ save label at ',deriv_root_label, 
                  subject,
            config_filename,
            '\n',datetime.now(),
            )
    # ----- between-trial comp
    # since trl number of 2 condtions are different 
    # use permutation_cluster_test, rather than permutation_cluster_1samp_test, to find condition-level difference 
    # withsubjects paird group is assumed: we see two epochs group as the paired condition 
    # time and vertices as the dimension to find clusters
        
    if flag_betwtrl_cmp :
        times_use = times[cond1_tidx]
        logpow_tltv_1 = np.transpose(
                        np.log10(cond1_tlvt)  ,(0,2,1))
        logpow_tltv_2 = np.transpose(
                        np.log10(cond2_tlvt) ,(0,2,1))
        
        cond1_tltv = logpow_tltv_1
        cond2_tltv = logpow_tltv_2
        print('------------','start between-trial cluster test' ,subject,
            config_filename,
            '\n',datetime.now(),
            )
        # setting for permutation_cluster_test
        X_conlists = [cond1_tltv, cond2_tltv]
        
        tv_adjacency = mne.stats.combine_adjacency(len(times_use), adjacency_roi)
        n_observations = cond1_tltv.shape[0] +  cond2_tltv.shape[0]
        print('------------','n_permutations ',n_permutations,'list of two input ',cond1_tltv.shape)

        df = welch_df(cond1_tltv,cond2_tltv)
        
        if tail == 1:    # we want to test 1 tail a>b
            t_thresh = scipy.stats.t.ppf(1 - p_cluster_forming, df)      # one-tailed, upper critical value
        elif tail == -1: # we want to test 1 tail a<b
            t_thresh = scipy.stats.t.ppf(p_cluster_forming, df)          # one-tailed, lower critical value
        elif  tail == 0: # we want to test 2 tails a!=b
            t_thresh = scipy.stats.t.ppf(1 - p_cluster_forming / 2, df)  # two-tailed, t distribution

        cluster_stats = mne.stats.permutation_cluster_test(
            X = X_conlists,
            out_type       = out_type,
            n_permutations = n_permutations,
            threshold      = t_thresh,
            tail           = tail,
            adjacency      = tv_adjacency,#None,# False,# #If None, lattice adjacency is assumed,connecting each location to its neighbor(s) along last dime of X (or the last two dimensions if X is 2D). 
            seed           = np.random.default_rng(seed=turn_sub_code_to_num(subject)),
            stat_fun       = ttest_ind_nop,#mne.stats.f_oneway,#ttest_ind_nop,#defaut is mne.stats.f_oneway. should be ttest_ind_nop, not ttest_rel_nop,but trial number is not same
            max_step       = 1, 
            n_jobs         = None, 
            exclude        = None, 
            step_down_p    = 0, 
            t_power        = 1, 
            check_disjoint = False, 
            buffer_size=1000, 
            verbose=None
                    )

        T_obs, clusters, cluster_p_values, H0 = cluster_stats
        good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
        print('good_cluster_inds ',len(good_cluster_inds))
        print(cluster_p_values)
        
        if len(good_cluster_inds) == 0:
            #-------- 2 betwtr save cluster_stats data
            with open(blankfile_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
                
            print('------------','no good_cluster save blank' ,subject,
            config_filename,
            '\n',datetime.now(),
            )
        
        else:
            ##-------- 2 betwtr save cluster_stats data
            with open(savefile_name, 'wb') as pickle_file:
                pickle.dump(cluster_stats, pickle_file)
                
            print('------------ save stat of ',len(good_cluster_inds),'good_cluster' 
                  ,subject,
            config_filename,
            '\n',datetime.now(),
            )
            # ----------------  save significant vertices as label
            
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
            
            stc = stc_exp.copy()
            stc.data = stc.data[:, :1]
            new_data = np.zeros_like(stc.data)
            new_data[sig_vidx_lr, 0] = 1 
            stc.data = new_data
            
            flag_sigl,flag_sigr,sig_vidx_l,sig_vidx_r \
                = create_save_label_from_vert(
                stc             = stc,
                src             = src,
                sig_vidx_lr     = sig_vidx_lr,
                flag_get_label = 1,
                labelaveDir          = deriv_root_label,
                groupdata_name       = saved_roi_list_name,
                subdata_name         = f"{saved_roi_list_name} {subject}",
                subject              = subject,
            )
            print('------------ save label at ',deriv_root_label, 
                  subject,
            config_filename,
            '\n',datetime.now(),
            )
#%%