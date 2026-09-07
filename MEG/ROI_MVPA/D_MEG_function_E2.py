# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:43:32 2022

@author: Ling Liu  ling.liu@pku.edu.cn
=================================
Functions for MEG decoding, special for experiment 2
=================================
"""

import os
import os.path as op
import joblib
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import shutil
import argparse
from mne import read_source_estimate


from mne.decoding import (Vectorizer, SlidingEstimator, cross_val_multiscore, get_coef)
# import a linear classifier from mne.decoding
from mne.decoding import LinearModel
from mne.decoding import GeneralizingEstimator
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold

from skimage.measure import block_reduce

from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects


#from config import no_eeg_sbj
#from config import site_id, subject_id, file_names, visit_id, data_path, out_path
from config import l_freq, h_freq, sfreq
from config import (bids_root, tmin, tmax)

from rsa_helper_functions import equate_offset




# set the path for decoding analysis
def set_path_ROI_MVPA(bids_root,subject_id, visit_id, analysis_name):
    ### I   Set subject information
    # sub and visit info
    sub_info = 'sub-' + subject_id + '_ses-' + visit_id
    print(sub_info)

    ### II  Set the Input Data Path
    # 1 Set path to the data root path
    fpath_root = op.join(bids_root, "derivatives") #data_path
    # fpath_root = '/Volumes/Cogitate/HPC'
    #fpath_root = '/home/user/S10/Cogitate/HPC'
    # fpath_root = 'Z:\HPC'

    # 2 Set path to preprocessed sensor (xxx_epo.fif)
    fpath_epo = op.join(fpath_root, "preprocessing",
                        f"sub-{subject_id}", f"ses-{visit_id}", "meg")
    #fpath_epo = op.join(fpath_root, 'epo')
    # /sub-SB085_ses-V1_task-dur_epo.fif'
    # fname_epo = op.join(out_path,
    #                     file_names[0][0:13] + 'ALL_epo.fif')

    # 2 Set path to the preprocessed source model data
    fpath_fw = op.join(fpath_root,'forward', f"sub-{subject_id}", "ses-" + visit_id, "meg")

    # 3 Set path to the freesufer subjects_dir for source analysis
    fpath_fs=op.join(fpath_root, "fs")
    # subjects_dir = r'/home/user/S10/Cogitate/HPC/fs'


    ### III  Set the Output Data Path
    # Set path to decoding derivatives
    mvpa_deriv_root = op.join(fpath_root, "decoding")
    if not op.exists(mvpa_deriv_root):
        os.makedirs(mvpa_deriv_root)
        
    
    # Set path to the ROI MVPA output(1) data, 2) figures, 3) codes)
    roi_deriv_root = op.join(mvpa_deriv_root, "roi_mvpa_e2", analysis_name)
    if not op.exists(roi_deriv_root):
        os.makedirs(roi_deriv_root)
    # 1) output_data
    roi_data_root = op.join(roi_deriv_root,
                            f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                            "data")
    if not op.exists(roi_data_root):
        os.makedirs(roi_data_root)

    # 2) output_figure
    roi_figure_root = op.join(roi_deriv_root,
                              f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                              "figures")
    if not op.exists(roi_figure_root):
        os.makedirs(roi_figure_root)

    # 3) output_code
    roi_code_root = op.join(roi_deriv_root,
                            f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                            "codes")
    if not op.exists(roi_code_root):
        os.makedirs(roi_code_root)

    return sub_info,fpath_epo,fpath_fw,fpath_fs, roi_data_root,roi_figure_root, roi_code_root

# functions for use both spatial and temporal feature as the decoding feature
def STdata(Xraw):
    #spatial + temporal decoding
    # temporal feature window
    #Xraw=epochs_cd.get_data()
    twd=5  # how many time points will used as temporal feature
    Xtemp=[];
    for twd_index in range(twd):
        if twd_index==0:
            #Xtemp1=np.append(Xraw[:,:,:1],Xraw[:,:,:-1],axis=2)
            Xtemp = Xraw
        else:
            Xtemp1=np.append(Xraw[:,:,:twd_index],Xraw[:,:,:-twd_index],axis=2)
            Xtemp=np.append(Xtemp,Xtemp1,axis=1)
    
    return Xtemp

# sliding windows (twd,) for MEG data
def ATdata(Xraw,nbin):

    #Xraw=epochs_cd.get_data()
    twd=nbin  # how many time points will be used as sliding windows
    [t1,t2,t3]=Xraw.shape
    Xtemp=np.zeros([nbin,t1,t2,t3]);
    for twd_index in range(twd):
        if twd_index==0:
            #Xtemp1=np.append(Xraw[:,:,:1],Xraw[:,:,:-1],axis=2)
            #Xtemp = np.expand_dims(Xraw, axis=0)
            Xtemp[twd_index,:,:,:] = Xraw
        else:
            Xtemp1=np.append(Xraw[:,:,:twd_index],Xraw[:,:,:-twd_index],axis=2)
            Xtemp[twd_index,:,:,:] = Xtemp1
    
    Xnew=np.mean(Xtemp,axis=0)
    
    return Xnew


def sensor_data_for_ROI_MVPA_AT(fpath_epo,sub_info):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-replay_epo.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')
    
    fname_rest=op.join(fpath_epo,sub_info + '_task-rest_epo.fif')
    epochs_rest = mne.read_epochs(fname_rest,
                             preload=True,
                             verbose=True).pick('meg')

    
        
    #condition_Stim=['Face','Object']
    #condition_Trial=['Target','Non-Target']
    
    
    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs = epochs.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    epochs_rs.crop(tmin=-0.5, tmax=1,include_tmax=True, verbose=None)
    
    epochs_rest_rs=epochs_rest.copy().resample(sfreq, n_jobs=-1)
    epochs_rest_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.25
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)

    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')
    rank_rest = mne.compute_rank(epochs_rest_rs, tol=1e-6, tol_kind='relative')
    if rank_rest['meg']<rank['meg']:
        rank=rank_rest
    

    baseline_cov = mne.compute_covariance(epochs_rest_rs, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=1, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    return epochs_rs, rank, common_cov



def sensor_data_for_ROI_MVPA_dAT(fpath_epo,sub_info):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-vg_epo.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')
    
    fname_rest=op.join(fpath_epo,sub_info + '_task-rest_epo.fif')
    epochs_rest = mne.read_epochs(fname_rest,
                             preload=True,
                             verbose=True).pick('meg')

    epochs_p = epochs['Trial_type in {}'.format(['Probe'])]
    condition_Stim=['Face','Object']
    epochs = epochs_p['Stimuli_type in {}'.format(condition_Stim)]
        
    #condition_Stim=['Face','Object']
    #condition_Trial=['Seen','Unseen']
    
    
    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs = epochs.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    epochs_rs.crop(tmin=-0.5, tmax=1,include_tmax=True, verbose=None)
    
    epochs_rest_rs=epochs_rest.copy().resample(sfreq, n_jobs=-1)
    epochs_rest_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.25
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)

    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')
    rank_rest = mne.compute_rank(epochs_rest_rs, tol=1e-6, tol_kind='relative')
    if rank_rest['meg']<rank['meg']:
        rank=rank_rest
    

    baseline_cov = mne.compute_covariance(epochs_rest_rs, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=1, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    return epochs_rs, rank, common_cov



def source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label,task_info):

    # projecting sensor-space data to source space
    # the path of forward solution
    fname_fwd = op.join(fpath_fw, sub_info + '_task-'+task_info+"_surface_fwd.fif")

    fwd = mne.read_forward_solution(fname_fwd)
    
    #make inverse operator
    # Make inverse operator
   
    inv = mne.minimum_norm.make_inverse_operator(epochs_rs.info, fwd, common_cov,
                                                 loose=.2,depth=.8,fixed=False,
                                                 rank=rank,use_cps=True)  # cov= baseline + active, compute rank, same as the LCMV
    
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(epochs_rs, inv, 1. / lambda2, 'dSPM', pick_ori="normal", label=surf_label)

    return stcs

def get_lables(fpath_fs,subject_id,ROI_list):
    # prepare the label for extract data
    
        
    PFC13_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                   'G&S_cingul-Mid-Post', 'G_front_middle',
                    'S_front_inf', 'S_front_sup',
                    'Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical',
                    'G_front_inf-Opercular','G_front_inf-Orbital','G_front_inf-Triangul',
                    'S_front_middle','G_front_sup'
                    ]
    
    P2F_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                   'G&S_cingul-Mid-Post', 'G_front_middle',
                    'S_front_inf', 
                    'S_intrapariet&P_trans','S_postcentral',
                    'G_postcentral','S_central','G_precentral'
                    ]
    
    IIT_ROI_list_path=r'/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/decoding/roi_mvpa_e2/'
    IIT_ROI_list_name=op.join(IIT_ROI_list_path,'iit_roilist_'+subject_id+'.npy')
    power_roi=np.load(IIT_ROI_list_name,allow_pickle=True)
    power_roi=power_roi.item()
    roi_ts_list=[x for x in power_roi.values()]
    IIT_ts_list=roi_ts_list[0]+roi_ts_list[1]
    
    
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels_parc_sub = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)
        PFC13_ts_list= [x.replace('&', '_and_') for x in PFC13_ts_list]
        IIT_ts_list= [x.replace('&', '_and_') for x in IIT_ts_list]
        P2F_ts_list= [x.replace('&', '_and_') for x in P2F_ts_list]
        
        
        
    else:
        labels_parc_sub = mne.read_labels_from_annot(subject=f"sub-{subject_id}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)
        
    #IITPFC=IIT_ts_list+PFC13_ts_list
        
    IIT_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name in IIT_ts_list:
            IIT_ts_index.append(ii)
            
    for ni, n_label in enumerate(IIT_ts_index):
        IIT_label=[x for x in labels_parc_sub if x.name==labels_parc_sub[n_label].name][0]
        if ni ==0:
            rIIT_label = IIT_label
        elif ni ==1:
            lIIT_label = IIT_label
        elif ni %2 ==0:
            rIIT_label = rIIT_label + IIT_label
        else:
            lIIT_label = lIIT_label + IIT_label
            
            
    PFC_ts_index=[]
    for ii in range(len(labels_parc_sub)):
        label_name=[]
        label_name=labels_parc_sub[ii].name
        #print(label_name)
        if label_name[:-3] in PFC13_ts_list:
            PFC_ts_index.append(ii)
            
    P2F_ts_index=[]
    for ii in range(len(labels_parc_sub)):
        label_name=[]
        label_name=labels_parc_sub[ii].name
        #print(label_name)
        if label_name[:-3] in P2F_ts_list:
            P2F_ts_index.append(ii)
      
    IITPFC_ts_index=[]
    IITPFC_ts_index=IIT_ts_index+PFC_ts_index
        
    # IITPFC_ts_index = []
    # for ii in range(len(labels_parc_sub)):
    #     label_name = []
    #     label_name = labels_parc_sub[ii].name
    #     if label_name in IITPFC_ts_index:
    #         IITPFC_ts_index.append(ii)
            
    for ni, n_label in enumerate(IITPFC_ts_index):
        IITPFC_label=[x for x in labels_parc_sub if x.name==labels_parc_sub[n_label].name][0]
        if ni ==0:
            rIITPFC_label = IITPFC_label
        elif ni ==1:
            lIITPFC_label = IITPFC_label
        elif ni %2 ==0:
            rIITPFC_label = rIITPFC_label + IITPFC_label
        else:
            lIITPFC_label = lIITPFC_label + IITPFC_label
    
    for i in range(13):
        locals()[f'PFC13_index_{i+1}']=[]
        for ii in range(len(labels_parc_sub)):
            label_name = []
            label_name = labels_parc_sub[ii].name
            if label_name[:-3] in PFC13_ts_list[i]:
                locals()[f'PFC13_index_{i+1}'].append(ii)
                
        for ni, n_label in enumerate(locals()[f'PFC13_index_{i+1}']):
            PFC13_temp_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
            if ni == 0:
                locals()[f'rPFC13_ROI_{i+1}'] = PFC13_temp_label
            elif ni == 1:
                locals()[f'lPFC13_ROI_{i+1}'] = PFC13_temp_label
            elif ni % 2 == 0:
                locals()[f'rPFC13_ROI_{i+1}'] = locals()[f'rPFC13_ROI_{i+1}'] + PFC13_temp_label  # , hemi="both"
            else:
                locals()[f'lPFC13_ROI_{i+1}'] = locals()[f'lPFC13_ROI_{i+1}'] + PFC13_temp_label
                
    for i in range(10):
        locals()[f'P2F_index_{i+1}']=[]
        for ii in range(len(labels_parc_sub)):
            label_name = []
            label_name = labels_parc_sub[ii].name
            if label_name[:-3] in P2F_ts_list[i]:
                locals()[f'P2F_index_{i+1}'].append(ii)
                
        for ni, n_label in enumerate(locals()[f'P2F_index_{i+1}']):
            P2F_temp_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
            if ni == 0:
                locals()[f'rP2F_ROI_{i+1}'] = P2F_temp_label
            elif ni == 1:
                locals()[f'lP2F_ROI_{i+1}'] = P2F_temp_label
            elif ni % 2 == 0:
                locals()[f'rP2F_ROI_{i+1}'] = locals()[f'rP2F_ROI_{i+1}'] + P2F_temp_label  # , hemi="both"
            else:
                locals()[f'lP2F_ROI_{i+1}'] = locals()[f'lP2F_ROI_{i+1}'] + P2F_temp_label
       
    if ROI_list=='IIT':
        surf_label_list=[rIIT_label+lIIT_label,rIITPFC_label +lIITPFC_label]
        ROI_Name=['IIT','IITPFC']
    elif ROI_list=='PFC_subROI':
        surf_label_list=[]
        ROI_Name=[]
        for i in range(13):
            temp=[locals()[f'rPFC13_ROI_{i+1}']+locals()[f'lPFC13_ROI_{i+1}']]
            surf_label_list=surf_label_list+temp
            temp_name=['PFC13_ROI_'+str(i+1)]
            ROI_Name=ROI_Name+temp_name
            #ROI_Name=ROI_Name+[locals()[f'PFC13_ROI_{i+1}']]
    elif ROI_list=='P2F_subROI':
        surf_label_list=[]
        ROI_Name=[]
        for i in range(10):
            temp=[locals()[f'rP2F_ROI_{i+1}']+locals()[f'lP2F_ROI_{i+1}']]
            surf_label_list=surf_label_list+temp
            temp_name=['P2F_ROI_'+str(i+1)]
            ROI_Name=ROI_Name+temp_name
            #ROI_Name=ROI_Name+[locals()[f'PFC13_ROI_{i+1}']]

    return(surf_label_list, ROI_Name)
    

def sub_ROI_for_ROI_MVPA(fpath_fs,subject_id,analysis_name):
    
    # prepare the label for extract data
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels_parc_sub = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)
    else:
        labels_parc_sub = mne.read_labels_from_annot(subject=f"sub-{subject_id}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)

    
    # replace "&" and "_and_" for indisual MRI or fsaverage
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        #ROI info, could change ###TODO: the final defined ROI
        # GNW_ts_list = ['G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant',
        #                'G_and_S_cingul-Mid-Post', 'G_front_middle',
        #                 'S_front_inf', 'S_front_sup',
        #                 ]
        
        GNW_ts_list = ['G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant',
                       'G_and_S_cingul-Mid-Post', 'G_front_middle',
                        'S_front_inf', 'S_front_sup',
                        'Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical',
                        'G_front_inf-Opercular','G_front_inf-Orbital','G_front_inf-Triangul',
                        'S_front_middle','G_front_sup'
                        ]
        
        PFC6_ts_list = ['G_and_S_cingul-Ant', 'G_front_middle',
                        'Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical',
                        'G_front_inf-obital','G_front_inf-Triangul',
                        'S_front_middle'
                        ]
        
        PFC_ts_list = ['G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant',
                       'G_and_S_cingul-Mid-Post', 'G_front_middle', 'S_front_sup',
                        ] #'S_front_inf' # remove S_front_inf, since this GNW ROI is also in the extented IIT ROI list.
        
        IIT_ts_list = ['G_cuneus',
                       'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Lingual',
                       'Pole_occipital', 'S_calcarine',
                       'S_oc_sup_and_transversal']
        
        MT_ts_list = ['S_central','S_postcentral']
        
        
        
    else:
        #ROI info, could change ###TODO: the final defined ROI
        # GNW_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
        #                'G&S_cingul-Mid-Post', 'G_front_middle',
        #                 'S_front_inf', 'S_front_sup',
        #                 ]
        GNW_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                       'G&S_cingul-Mid-Post', 'G_front_middle',
                        'S_front_inf', 'S_front_sup',
                        'Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical',
                        'G_front_inf-Opercular','G_front_inf-obital','G_front_inf-Triangul',
                        'S_front_middle','G_front_sup'
                        ]
        
        PFC6_ts_list = ['G&S_cingul-Ant', 'G_front_middle',
                        'Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical',
                        'G_front_inf-obital','G_front_inf-Triangul',
                        'S_front_middle'
                        ]
        
        PFC_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                       'G&S_cingul-Mid-Post', 'G_front_middle', 'S_front_sup',
                        ] #'S_front_inf' # remove S_front_inf, since this GNW ROI is also in the extented IIT ROI list.
        
        
        IIT_ts_list = ['G_cuneus',
                       'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Lingual',
                       'Pole_occipital', 'S_calcarine',
                       'S_oc_sup&transversal']
        
        #MT_ts_list = ['S_central','S_postcentral']

        MT_ts_list = ['S_central']
        
        

    GNW_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in GNW_ts_list:
            GNW_ts_index.append(ii)
    
    # PFC13_ts_index = []
    # for ii in range(len(labels_parc_sub)):
    #     label_name = []
    #     label_name = labels_parc_sub[ii].name
    #     if label_name[:-3] in PFC13_ts_list:
    #         PFC13_ts_index.append(ii)
            
    PFC6_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in PFC6_ts_list:
            PFC6_ts_index.append(ii)
            
    PFC_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in PFC_ts_list:
            PFC_ts_index.append(ii)
        
            

    IIT_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in IIT_ts_list:
            IIT_ts_index.append(ii)
            
    MT_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in MT_ts_list:
            MT_ts_index.append(ii)
            
    

    for ni, n_label in enumerate(GNW_ts_index):
        GNW_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rGNW_label = GNW_label
        elif ni == 1:
            lGNW_label = GNW_label
        elif ni % 2 == 0:
            rGNW_label = rGNW_label + GNW_label  # , hemi="both"
        else:
            lGNW_label = lGNW_label + GNW_label
            
    # for ni, n_label in enumerate(PFC13_ts_index):
    #     PFC13_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
    #     if ni == 0:
    #         rPFC13_label = PFC13_label
    #     elif ni == 1:
    #         lPFC13_label = PFC13_label
    #     elif ni % 2 == 0:
    #         rPFC13_label = rPFC13_label + PFC13_label  # , hemi="both"
    #     else:
    #         lPFC13_label = lPFC13_label + PFC13_label
            
            
    for ni, n_label in enumerate(PFC6_ts_index):
        PFC6_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rPFC6_label = PFC6_label
        elif ni == 1:
            lPFC6_label = PFC6_label
        elif ni % 2 == 0:
            rPFC6_label = rPFC6_label + PFC6_label  # , hemi="both"
        else:
            lPFC6_label = lPFC6_label + PFC6_label
            
    for ni, n_label in enumerate(PFC_ts_index):
        PFC_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rPFC_label = PFC_label
        elif ni == 1:
            lPFC_label = PFC_label
        elif ni % 2 == 0:
            rPFC_label = rPFC_label + PFC_label  # , hemi="both"
        else:
            lPFC_label = lPFC_label + PFC_label
            
    for ni, n_label in enumerate(IIT_ts_index):
        IIT_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rIIT_label = IIT_label
        elif ni == 1:
            lIIT_label = IIT_label
        elif ni % 2 == 0:
            rIIT_label = rIIT_label + IIT_label  # , hemi="both"
        else:
            lIIT_label = lIIT_label + IIT_label
            
    for ni, n_label in enumerate(MT_ts_index):
        MT_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rMT_label = MT_label
        elif ni == 1:
            lMT_label = MT_label
        elif ni % 2 == 0:
            rMT_label = rMT_label + MT_label  # , hemi="both"
        else:
            lMT_label = lMT_label + MT_label
            
    



            
    if analysis_name=='Cat' or analysis_name=='Ori' or analysis_name=='Cat_offset_control':
        surf_label_list = [rGNW_label+lGNW_label, rIIT_label+lIIT_label,rGNW_label+lGNW_label+rIIT_label+lIIT_label]
        ROI_Name = ['GNW', 'IIT','FP']
        
    elif analysis_name=='Cat_MT_control':
        surf_label_list = [rMT_label+lMT_label]
        ROI_Name = ['MT']
        
    
    elif analysis_name=='Cat_PFC' or analysis_name=='Ori_PFC':
        surf_label_list = [rPFC_label+lPFC_label, rIIT_label+lIIT_label,rPFC_label+lPFC_label+rIIT_label+lIIT_label]
        ROI_Name = ['PFC', 'IIT','IITPFC']
        
    # elif analysis_name=='AT_PFC13_control' or analysis_name=='dAT_PFC13':
    #     surf_label_list = [rPFC13_label+lPFC13_label]
    #     ROI_Name = ['PFC13']
        
    elif analysis_name=='AT_PFC6_control':
        surf_label_list = [rPFC6_label+lPFC6_label]
        ROI_Name = ['PFC6']
    
    else:
        surf_label_list = [rGNW_label+lGNW_label, rIIT_label+lIIT_label]
        ROI_Name = ['GNW']

    return surf_label_list, ROI_Name

def sub_ROI_for_ROI_MVPA_subROI(fpath_fs,subject_id,analysis_name):
    # prepare the label for extract data
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels_parc_sub = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)
    else:
        labels_parc_sub = mne.read_labels_from_annot(subject=f"sub-{subject_id}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)

    
    # replace "&" and "_and_" for indisual MRI or fsaverage
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        
        F1_ts_list=['G_and_S_cingul-Ant']
        F2_ts_list=['G_and_S_cingul-Mid-Ant']
        F3_ts_list=['G_and_S_cingul-Mid-Post']
        F4_ts_list=['G_front_middle']
        F5_ts_list=['S_front_inf']
        F6_ts_list=['S_front_sup']
        F7_ts_list=['Lat_Fis-ant-Horizont']
        F8_ts_list=['Lat_Fis-ant-Vertical']
        F9_ts_list=['G_front_inf-Opercular']
        F10_ts_list=['G_front_inf-Orbital']
        F11_ts_list=['G_front_inf-Triangul']
        F12_ts_list=['S_front_middle']
        F13_ts_list=['G_front_sup']
        
        P1_ts_list=['S_intrapariet_and_P_trans']
        P2_ts_list=['S_postcentral']
        P3_ts_list=['G_postcentral']
        P4_ts_list=['S_central']
        P5_ts_list=['G_precentral']
        P6_ts_list=['S_precentral-inf-part']
        
        
    else:
        
        F1_ts_list=['G&S_cingul-Ant']
        F2_ts_list=['G&S_cingul-Mid-Ant']
        F3_ts_list=['G&S_cingul-Mid-Post']
        F4_ts_list=['G_front_middle']
        F5_ts_list=['S_front_inf']
        F6_ts_list=['S_front_sup']
        F7_ts_list=['Lat_Fis-ant-Horizont']
        F8_ts_list=['Lat_Fis-ant-Vertical']
        F9_ts_list=['G_front_inf-Opercular']
        F10_ts_list=['G_front_inf-Orbital']
        F11_ts_list=['G_front_inf-Triangul']
        F12_ts_list=['S_front_middle']
        F13_ts_list=['G_front_sup']
        
    
        P1_ts_list=['S_intrapariet&P_trans']
        P2_ts_list=['S_postcentral']
        P3_ts_list=['G_postcentral']
        P4_ts_list=['S_central']
        P5_ts_list=['G_precentral']
        P6_ts_list=['S_precentral-inf-part']
        

    
            
    F1_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F1_ts_list:
            F1_ts_index.append(ii)
    F2_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F2_ts_list:
            F2_ts_index.append(ii)
    F3_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F3_ts_list:
            F3_ts_index.append(ii)
    F4_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F4_ts_list:
            F4_ts_index.append(ii)
            
    F5_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F5_ts_list:
            F5_ts_index.append(ii)
    
    F6_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F6_ts_list:
            F6_ts_index.append(ii)
    
    F7_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F7_ts_list:
            F7_ts_index.append(ii)
            
    F8_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F8_ts_list:
            F8_ts_index.append(ii)
    F9_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F9_ts_list:
            F9_ts_index.append(ii)
    F10_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F10_ts_list:
            F10_ts_index.append(ii)
            
    F11_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F11_ts_list:
            F11_ts_index.append(ii)
    F12_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F12_ts_list:
            F12_ts_index.append(ii)
    
    F13_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F13_ts_list:
            F13_ts_index.append(ii)
            
            
            
    P1_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P1_ts_list:
            P1_ts_index.append(ii)
    P2_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P2_ts_list:
            P2_ts_index.append(ii)
    P3_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P3_ts_list:
            P3_ts_index.append(ii)
    P4_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P4_ts_list:
            P4_ts_index.append(ii)
            
    P5_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P5_ts_list:
            P5_ts_index.append(ii)
    P6_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P6_ts_list:
            P6_ts_index.append(ii)

    
            
    for ni, n_label in enumerate(F1_ts_index):
        F1_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF1_label = F1_label
        elif ni == 1:
            lF1_label = F1_label
        elif ni % 2 == 0:
            rF1_label = rF1_label + F1_label  # , hemi="both"
        else:
            lF1_label = lF1_label + F1_label
    
    for ni, n_label in enumerate(F2_ts_index):
        F2_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF2_label = F2_label
        elif ni == 1:
            lF2_label = F2_label
        elif ni % 2 == 0:
            rF2_label = rF2_label + F2_label  # , hemi="both"
        else:
            lF2_label = lF2_label + F2_label
            
    for ni, n_label in enumerate(F3_ts_index):
        F3_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF3_label = F3_label
        elif ni == 1:
            lF3_label = F3_label
        elif ni % 2 == 0:
            rF3_label = rF3_label + F3_label  # , hemi="both"
        else:
            lF3_label = lF3_label + F3_label
            
    for ni, n_label in enumerate(F4_ts_index):
        F4_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF4_label = F4_label
        elif ni == 1:
            lF4_label = F4_label
        elif ni % 2 == 0:
            rF4_label = rF4_label + F4_label  # , hemi="both"
        else:
            lF4_label = lF4_label + F4_label
            
    for ni, n_label in enumerate(F5_ts_index):
        F5_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF5_label = F5_label
        elif ni == 1:
            lF5_label = F5_label
        elif ni % 2 == 0:
            rF5_label = rF5_label + F5_label  # , hemi="both"
        else:
            lF5_label = lF5_label + F5_label
            
    for ni, n_label in enumerate(F6_ts_index):
        F6_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF6_label = F6_label
        elif ni == 1:
            lF6_label = F6_label
        elif ni % 2 == 0:
            rF6_label = rF6_label + F6_label  # , hemi="both"
        else:
            lF6_label = lF6_label + F6_label
            
    for ni, n_label in enumerate(F7_ts_index):
        F7_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF7_label = F7_label
        elif ni == 1:
            lF7_label = F7_label
        elif ni % 2 == 0:
            rF7_label = rF7_label + F7_label  # , hemi="both"
        else:
            lF7_label = lF7_label + F7_label
            
    # for ni, n_label in enumerate(F7_ts_index):
    #     F7_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
    #     if ni == 0:
    #         rF7_label = F7_label
    #     elif ni == 1:
    #         lF7_label = F7_label
    #     elif ni % 2 == 0:
    #         rF7_label = rF7_label + F7_label  # , hemi="both"
    #     else:
    #         lF7_label = lF7_label + F7_label
    
    for ni, n_label in enumerate(F8_ts_index):
        F8_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF8_label = F8_label
        elif ni == 1:
            lF8_label = F8_label
        elif ni % 2 == 0:
            rF8_label = rF8_label + F8_label  # , hemi="both"
        else:
            lF8_label = lF8_label + F8_label
            
    for ni, n_label in enumerate(F9_ts_index):
        F9_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF9_label = F9_label
        elif ni == 1:
            lF9_label = F9_label
        elif ni % 2 == 0:
            rF9_label = rF9_label + F9_label  # , hemi="both"
        else:
            lF9_label = lF9_label + F9_label
            
    for ni, n_label in enumerate(F10_ts_index):
        F10_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF10_label = F10_label
        elif ni == 1:
            lF10_label = F10_label
        elif ni % 2 == 0:
            rF10_label = rF10_label + F10_label  # , hemi="both"
        else:
            lF10_label = lF10_label + F10_label
            
    for ni, n_label in enumerate(F11_ts_index):
        F11_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF11_label = F11_label
        elif ni == 1:
            lF11_label = F11_label
        elif ni % 2 == 0:
            rF11_label = rF11_label + F11_label  # , hemi="both"
        else:
            lF11_label = lF11_label + F11_label
            
    for ni, n_label in enumerate(F12_ts_index):
        F12_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF12_label = F12_label
        elif ni == 1:
            lF12_label = F12_label
        elif ni % 2 == 0:
            rF12_label = rF12_label + F12_label  # , hemi="both"
        else:
            lF12_label = lF12_label + F12_label
            
    for ni, n_label in enumerate(F13_ts_index):
        F13_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF13_label = F13_label
        elif ni == 1:
            lF13_label = F13_label
        elif ni % 2 == 0:
            rF13_label = rF13_label + F13_label  # , hemi="both"
        else:
            lF13_label = lF13_label + F13_label
            
    for ni, n_label in enumerate(P1_ts_index):
        P1_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP1_label = P1_label
        elif ni == 1:
            lP1_label = P1_label
        elif ni % 2 == 0:
            rP1_label = rP1_label + P1_label  # , hemi="both"
        else:
            lP1_label = lP1_label + P1_label
    
    for ni, n_label in enumerate(P2_ts_index):
        P2_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP2_label = P2_label
        elif ni == 1:
            lP2_label = P2_label
        elif ni % 2 == 0:
            rP2_label = rP2_label + P2_label  # , hemi="both"
        else:
            lP2_label = lP2_label + P2_label
            
    for ni, n_label in enumerate(P3_ts_index):
        P3_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP3_label = P3_label
        elif ni == 1:
            lP3_label = P3_label
        elif ni % 2 == 0:
            rP3_label = rP3_label + P3_label  # , hemi="both"
        else:
            lP3_label = lP3_label + P3_label
            
    for ni, n_label in enumerate(P4_ts_index):
        P4_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP4_label = P4_label
        elif ni == 1:
            lP4_label = P4_label
        elif ni % 2 == 0:
            rP4_label = rP4_label + P4_label  # , hemi="both"
        else:
            lP4_label = lP4_label + P4_label
            
    for ni, n_label in enumerate(P5_ts_index):
        P5_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP5_label = P5_label
        elif ni == 1:
            lP5_label = P5_label
        elif ni % 2 == 0:
            rP5_label = rP5_label + P5_label  # , hemi="both"
        else:
            lP5_label = lP5_label + P5_label
            
    for ni, n_label in enumerate(P6_ts_index):
        P6_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP6_label = P6_label
        elif ni == 1:
            lP6_label = P6_label
        elif ni % 2 == 0:
            rP6_label = rP6_label + P6_label  # , hemi="both"
        else:
            lP6_label = lP6_label + P6_label



            
    if  analysis_name=='AT_subF_control':
        surf_label_list = [rF1_label+lF1_label,rF2_label+lF2_label,rF3_label+lF3_label,
                           rF4_label+lF4_label,rF5_label+lF5_label,rF6_label+lF6_label,rF7_label+lF7_label,
                           rF8_label+lF8_label,rF9_label+lF9_label,
                           rF10_label+lF10_label,rF11_label+lF11_label,rF12_label+lF12_label,
                           rF13_label+lF13_label]
        ROI_Name = ['F1','F2','F3','F4','F5','F6','F7',
                    'F8','F9','F10','F11','F12','F13']
        
    elif analysis_name=='Cat_subP_control':
        surf_label_list = [rP1_label+lP1_label,rP2_label+lP2_label,rP3_label+lP3_label,
                           rP4_label+lP4_label,rP5_label+lP5_label,rP6_label+lP6_label]
        ROI_Name = ['P1','P2','P3','P4','P5','P6']
    

    return surf_label_list, ROI_Name
