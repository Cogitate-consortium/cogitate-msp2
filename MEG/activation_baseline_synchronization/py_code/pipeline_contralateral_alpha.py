'''

baseline analysis for contralateral alpha power
This analysis answer: Is contralateral alpha different between cond1 and cond2 (irrespective of hemifield)

configure file: 
config_files/pipeline_contralateral_alpha/dAT_probe_stim_noft_nobc_deci5_nocut_cla_POS.json

Since we need power of longer time windows, we could not directly use the pre-saved power data.
Here we re-calculate the power for the contra-lateral alpha analysis.

To compare contralateral alpha power between seen and unseen trials in experiment 2.

Individual analysis (this script):
For each trial and each label in the ROI, compute the time-resolved power in the
alpha band (band_map[band_name], Morlet) and average over the band frequencies.
The power of ALL trials is saved together (tlvt_alphapower_by_label) with the epoch
metadata and the left/right hemisphere vertex indices (vidx_dict); trials are split
by hemifield (LVF/RVF) and response (seen/unseen) later in the group script.

Group analysis (pipeline_contralateral_alpha_group.py):
per condition, average alpha power over contralateral vs ipsilateral vertices,
where for left-location trials contra = right hemisphere and for right-location
trials contra = left hemisphere:
  contra = 1/2 * (P_leftVF_rightH + P_rightVF_leftH)
  ipsi   = 1/2 * (P_leftVF_leftH  + P_rightVF_rightH)
  
The measure that is tested is the contralateral power (contra_power), selected via
the group config's contralateral_calculation_method (contra_power_baseline.json).
'''
#%% import
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse 

# for show time
from datetime import datetime

# for save data
import pickle
from EXP1_help_functions import (BidsPath)

from _help_functions import (general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 read_configfile,
                                 get_fw_inv,
                                 get_stcs_allsrc,
                                 read_labels_exp2,
                                get_label_vertidx,
                                
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
    args.subject = 'SA121'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_contralateral_alpha" / "dAT_probe_stim_noft_nobc_deci5_nocut_cla_POS.json"

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
    
epoch_setting = general_param['epoch_setting_dict'][param['epoch_setting_name']]
epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
roi_params = general_param['roi_params_dict']
band_map = general_param['band_map']
band_name = param['band_name']

roi = param['roi']
                                   
info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                           f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                         info.epoch_info )

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param['epoch_data_name']}_contralateral_{band_name}_power_{roi}",
        extension='.pkl',
        check=False)
dir_analyse = os.path.dirname( bids_path.fpath)
os.makedirs(dir_analyse, exist_ok=True)      

pklfile_name = bids_path.fpath

#%% get the results
if  os.path.exists(pklfile_name) :
    print('------------','exist subject' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )
    
    try:
        with open(pklfile_name, 'rb') as pickle_file_load:
            loaded_data = pickle.load(pickle_file_load)
        print('------------','Successfully loaded' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )    
    except Exception :
        # Print the error for debugging purposes (optional)
        print('------------','Error loading' ,pklfile_name,
            '\n',config_filename,
            '\n',datetime.now(),
                )
        pass
else:

    print('------------','start subject' ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
            )
      

    eps_use_result = get_eps_use(subject,
                                debug= True if flag_test else False, 
                                ** epoch_data, 
                                ** epoch_setting)
    
    # Read forward model
    fw_inv_result  = get_fw_inv(
                        subject,
                        **eps_use_result,
                        **epoch_data,
                        **epoch_setting,
                        **general_param['source_method_setting'])
    
    # get stc for all epoch
    stcs_results = get_stcs_allsrc (
                        subject = subject,
                        **epoch_setting,
                        **general_param['source_method_setting'],
                        **fw_inv_result,
                        **eps_use_result)
    
    stcs_epoch      =  stcs_results['stcs_epoch']
    epochs_metadata =  stcs_results['stcs_metadata']
    src = fw_inv_result['inverse']['src']

    # get label info
    label_dict = {}
    roilist_all = {}
    for roi in roi_params.keys():
        roi
        lb_use = read_labels_exp2(
            bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   =  epoch_data['exp_id']), 
            parc       = roi_params[roi]['parc'], 
            labels_list= roi_params[roi]['labels_list'],
            rois_list  = roi_params[roi]['rois_list'], 
            combine_labels_list=True, 
            merge_hemi=True
            )  
        if lb_use:
            print('read label for',roi, list(lb_use.keys()))
            ## get vertno index
            label_vertidx_dict=\
                        get_label_vertidx(lb_use =lb_use,
                                            src    = fw_inv_result ['src'],
                                            flag_return_dict =1)
            label_dict[roi] = label_vertidx_dict
            roi_params[roi]['sublabel_list'] = list( lb_use.keys())
            roilist_all.update({roi:lb_use})
        else:
            print('no label for',roi)
            label_dict[roi] = {}
            roi_params[roi]['sublabel_list'] =[]
    print(roi_params[roi]['sublabel_list'])

    # all roi should have label
    assert len(roi_params[roi]) == len(roi_params[roi].keys()),'some roi have no label'
    

    #%  calculate the power for each label
    tlvt_alphapower_by_label = {}
    vidx_dict = {}
    for label_name, label in roilist_all[roi].items():
        # ---------- get the left and right hemi vertices index ----------
        label_vidx_lr = label_dict[roi]['vidx_dic_allowoverlap'][label_name]
        label_vidx_l = [x for x in label_vidx_lr if x < len(src[0]['vertno'])]
        label_vidx_r = [x for x in label_vidx_lr if x >= len(src[0]['vertno'])]
        vidx_dict[label_name] = {
            'both':label_vidx_lr,
            'left':label_vidx_l,
            'right':label_vidx_r}

        # ---------- validate the label_dict ----------
        # without vert ROI it could be easy to check src
        label_src = np.sum(
                    [label.lh.restrict(src),
                    label.rh.restrict(src)])
                
        assert label_dict[roi] ['v_l_dic'][label_name] == list(label_src.lh.vertices),'label_src not match v_l_dic'
        assert label_dict[roi] ['v_r_dic'][label_name] == list(label_src.rh.vertices),'label_src not match v_r_dic'
        # ---------- get the label_src ----------
        # if label of two hemi, combine two hemi and check the label dict 
        if ('rh' in label.name) and ('lh' in label.name):
            
            label_src = np.sum(
                        [label.lh.restrict(src),
                            label.rh.restrict(src)])
            print('------------ start ' ,roi,
                    'left',len(label_src.lh.vertices),
                    'right',len(label_src.rh.vertices),
                    ' ',label_name,
                        datetime.now(),)
            
            # if label.subject =='fsaverage':
            if ( src[0]['subject_his_id'] =='fsaverage') or (label.subject =='fsaverage'):
                label_src.subject = 'fsaverage'
            elif subject in label.subject:
                label_src.subject = 'sub-' + subject
            else:
                ValueError
                
            assert label_dict[roi] ['v_l_dic'][label_name] == list(label_src.lh.vertices),'label_src not match v_l_dic'
            assert label_dict[roi] ['v_r_dic'][label_name] == list(label_src.rh.vertices),'label_src not match v_r_dic'
            assert len(label_vidx_l) == len(label_src.lh.vertices),'label_src.lh.vertices not match label_vidx_l'
            assert len(label_vidx_r) == len(label_src.rh.vertices),'label_src.rh.vertices not match label_vidx_r'
        
        # if label only have one hemi, check the label_dict 
        else:
            label_src = label.restrict(src)
            print('start ' ,roi,label_name,' one hemi ',len(label_src.vertices))
            # if label.subject =='fsaverage':
            if  (src[0]['subject_his_id'] =='fsaverage' ) or (label.subject =='fsaverage'):
                label_src.subject = 'fsaverage'
            elif subject in label.subject:
                label_src.subject = 'sub-' + subject
            else:
                ValueError
            assert label_dict[roi] ['v_dic'][label_name] == list(label_src.vertices),'label_src not match v_l_dic'

        # ---------- validate the label_src being the same with the label_vidx_lr ----------
        aa = stcs_epoch[0].in_label(label_src).data
        bb = stcs_epoch[0].data[label_vidx_lr,:]
        assert np.allclose(aa, bb, rtol=1e-7, atol=1e-12) 
    
        tlvt_label = np.array([stc.in_label(label_src).data for idx,stc in  enumerate(stcs_epoch)])
        
        
        # ---------- get the power ----------
        print('tlvt_label', tlvt_label.shape,datetime.now())
                
        pow_tlvft_morlet =  mne.time_frequency.tfr_array_morlet(
            tlvt_label, 
            sfreq         = stcs_results['sfreq'] ,
            freqs         = band_map[band_name]['freqs'], 
            n_cycles      = band_map[band_name]['n_cycles'], 
            decim         = band_map[band_name]['decim'],
            use_fft=True, 
            output='power',
            n_jobs = 1,
            verbose=True)
        tlvt_alphapower_by_label[label_name] = np.mean(pow_tlvft_morlet, axis=2) 

        
    #  save data
    save_data = {
                'tlvt_alphapower_by_label':tlvt_alphapower_by_label,
                "vidx_dict":vidx_dict,
                'sfreq':stcs_results['sfreq'],
                'epochs_metadata':epochs_metadata,
                'times':stcs_results['times'],
                'label_dict':label_dict,
                'band_map':band_map,
                'roi_params':roi_params,
                'general_param_config':general_param,
                'param_config':param,
                } 

    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)
    print('------------','finish save '  ,subject,
            '\n',config_filename,
            '\n',datetime.now(),
    )
# %%
