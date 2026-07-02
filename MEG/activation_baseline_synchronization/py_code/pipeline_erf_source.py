'''
Individual-level: compute source-space ERF (condition averages) for every ROI /
label of a subject, for the cond1 vs cond2 comparison.

Per subject + config:
  - build source estimates for all epochs (get_stcs_allsrc),
  - for each ROI in roi_params, read its label(s) and, per label, extract the
    label vertex time courses and run compute_all_combinations with reducers
    (trial: mean & rms; vertex: rms) -> t_dict,
  - collect into t_dict_by_roi[roi][label_name].
Saves t_dict_by_roi + label_dict + roi_params + metadata as
<epoch_data_name>_<compare_trltypes>.pkl under path_allana/<script_name>/<epoch_info>/.
Skips if the file already exists.

Runs per label (not per whole ROI then extract) because vertices/labels are
saved per label, which is clearer.

Consumed by pipeline_erf_source_group.py (+ _group_plot / _group_checkleackage).

Config folder: /config_files/pipeline_erf_source/
    dAT_probe_face_ft_bc_deci1_nocut.json
    dAT_probe_obje_ft_bc_deci1_nocut.json
    dAT_probe_stim_ft_bc_deci1_nocut.json
  each sets epoch_setting_name, epoch_data_name, compare_trltypes,
  compare_trltypes_select_column, cond1/cond2_content_name + _content_list.
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
                                 turn_sub_code_to_num,
                                 read_configfile,
                                 get_fw_inv,
                                 get_stcs_allsrc,
                                 read_labels_exp2,
                                 get_label_vertidx
                                 )
from _help_module import (compute_all_combinations,SourceROIEpochs
                              )
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
    args.subject = 'SA108'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_erf_source" / "dAT_probe_face_ft_bc_deci1_nocut.json"

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


info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info)

bids_path = mne_bids.BIDSPath(
        root=deriv_root, 
        subject= subject, 
        session= epoch_data['exp_id'],  
        datatype='meg',  
        task=epoch_data['task_id'],
        suffix=f"{param['epoch_data_name']}_{param['compare_trltypes']}",
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
        print(loaded_data.keys())
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

    fw_inv_result  = get_fw_inv(
                        subject,
                        **eps_use_result,
                        **epoch_data,
                        **epoch_setting,
                        **general_param['source_method_setting'])
    src = fw_inv_result['src']
    
    # get stc for all epoch
    stcs_results = get_stcs_allsrc (
                        subject,
                        **eps_use_result,
                        **epoch_setting,
                        **general_param['source_method_setting'],
                        **fw_inv_result,
                        )


    stcs_epoch      =  stcs_results['stcs_epoch']
    epochs_metadata =  stcs_results['stcs_metadata']
    
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
    # assert len(roi_params[roi]) == len(roi_params[roi].keys()),'some roi have no label'
    
    
    #%
    t_dict_by_roi = {}
    for roi in roi_params.keys():
        print('------------ start roi',roi, datetime.now())  
        t_dict_by_roi[roi] = {}
        # roilist_all.keys() only contain roi that have label
        if roi not in roilist_all.keys():
            continue

        for label_name, label in roilist_all[roi].items():
            # ---------- validation ----------
            # without vert ROI it could be easy to check src
            label_src = np.sum(
                        [label.lh.restrict(src),
                        label.rh.restrict(src)])
                    
            assert label_dict[roi] ['v_l_dic'][label_name] == list(label_src.lh.vertices),'label_src not match v_l_dic'
            assert label_dict[roi] ['v_r_dic'][label_name] == list(label_src.rh.vertices),'label_src not match v_r_dic'
            vidx = label_dict[roi]['vidx_dic_allowoverlap'][label_name]
            
            # if label of two hemi, combine two hemi and check the label dict 
            if ('rh' in label.name) and ('lh' in label.name):
                
                label_src = np.sum(
                            [label.lh.restrict(src),
                                label.rh.restrict(src)])
                print('------------ start ' ,roi,label_name,'left ',len(label_src.lh.vertices),
                        ' right ',len(label_src.rh.vertices),
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

            
            aa = stcs_epoch[0].in_label(label_src).data
            bb = stcs_epoch[0].data[vidx,:]
            assert np.allclose(aa, bb, rtol=1e-7, atol=1e-12) 
        
            tlvt_label = np.array([stc.in_label(label_src).data for idx,stc in  enumerate(stcs_epoch)])
        
            label_epochs = SourceROIEpochs(tlvt=tlvt_label)


            t_dict = compute_all_combinations(
                epochs= label_epochs,
                epochs_metadata=epochs_metadata,
                order = ("trials", "vertices",),   # 
                epochdataa_dims_list= ["trials", "vertices", "times"],
                trial_type_col= param['compare_trltypes_select_column'],
                cond1_name= param['cond1_content_name'],
                cond2_name= param['cond2_content_name'],
                cond1_values= param['cond1_content_list'],
                cond2_values= param['cond2_content_list'],
                vertex_idx=None ,     
                n_subsamples= 20 if flag_test else general_param['Nsample'] ,
                seed= turn_sub_code_to_num(subject)   ,
                reducers = (("mean", "rms"), ("rms",))   ,          
            )
            
            t_dict_by_roi[roi][label_name] = t_dict
            if 'nepoch_df'  in locals():
                assert all(nepoch_df ==pd.DataFrame([t_dict[list(t_dict.keys())[0]]['counts']])),'error: times not match'
            else:
                nepoch_df = pd.DataFrame([t_dict[list(t_dict.keys())[0]]['counts']])
        
    #  save data
    save_data = {
                't_dict_by_roi':t_dict_by_roi,
                'param_config':param,
                'general_param_config':general_param,
                'nepoch_df':nepoch_df,
                'sfreq':eps_use_result['sfreq'],
                'epochs_metadata':epochs_metadata,
                'times'   :eps_use_result ['times'],
                'label_dict':label_dict,
                'roi_params':roi_params,
                } 

    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)

    print('------','save' ,
          '\n',pklfile_name,
        '\n',subject,
        '\n',config_filename,
        '\n',datetime.now(),
            )


#%%