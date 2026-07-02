'''
Individual-level synchronization analysis: IIT
extract single-trial ROI time courses using PCA (pca_flip),
for the synchronization analysis (IIT nodes: category-selective & V1/V2 vertices).

VertROIs: FF_face_select_vertices, FF_obje_select_vertices,
          V1V2_responsive_activewin_vs_baseline_vertices.

The PCA time course is computed HERE: the source
estimates of the vertices in each VertROI label are reduced with
mne.extract_label_time_course(mode='pca_flip'), 
giving the single-trial time course (tlt: n_trials x n_times),
saved per VertROI for later PPC analysis.

Config folder: /config_files/pipeline_syn_timecourse_PCA_VertROI/
    noft_nobc_deci1_nocut_AT_nontarget.json
    noft_nobc_deci1_nocut_AT_target.json
    noft_nobc_deci1_nocut_allseen.json
    noft_nobc_deci1_nocut_dAT_probeface.json
    noft_nobc_deci1_nocut_dAT_probeobje.json
    noft_nobc_deci1_nocut_dAT_probeseen.json
each config sets the epoch setting + epoch data (one or multiple) and the VertROIs to extract.
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

from _help_functions import (get_stcs_allsrc,
                             general_param,
                             path_allana,
                                 get_eps_use,
                                 get_eps_use_info,
                                 read_configfile,
                                 get_fw_inv,
                                 read_labels_exp2,
                                 get_label_vertidx
                                 )

import mne_bids
import mne
import numpy as np
import pandas as pd

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
    args.subject = 'SA118'# 'SA121'
    args.configfile = base / "config_files" / "pipeline_syn_timecourse_PCA_VertROI" / "noft_nobc_deci1_nocut_allseen.json"

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
# !!!!! specific for VertROI
roi_params = general_param['VertROI_params_dict']
sub_no_T1 = general_param['sub_no_T1']


info =get_eps_use_info(**epoch_setting) 


# create folder with analyse name
if flag_test :
    deriv_root = os.path.join(path_allana, script_name,
                          f"test_{info.epoch_info}")
else:
    deriv_root = os.path.join(path_allana, script_name,
                          info.epoch_info)
    

if "epoch_data_name" in param.keys():
    epoch_data = general_param['epoch_data_dict'][param['epoch_data_name']]
    epoch_data_list = [epoch_data]
    epoch_data_name_list = [param['epoch_data_name']]

elif 'epoch_data_name_list' in param.keys():
    epoch_data_list = [general_param['epoch_data_dict'][name] for name in param['epoch_data_name_list']]
    epoch_data_name_list = param['epoch_data_name_list']
    epoch_data_name_list_name = param['epoch_data_name_list_name']
    assert(len(np.unique([general_param['epoch_data_dict'][name]['exp_id'] for name in param['epoch_data_name_list']])) ==1),\
        "All epoch_data in epoch_data_name_list should have the same exp_id"

    exp_id = general_param['epoch_data_dict'][param['epoch_data_name_list'][0]]['exp_id']

else:
    raise ValueError('No epoch_data_name or epoch_data_name_list in configfile')

for VertROI in param['VertROIs']:
    # !!! special for this analysis to get roi Used in VertROIs
    roi = VertROI
    VertROI_type = roi_params[roi]['label_name']

    
    if 'epoch_data_name_list' in param.keys():
        tlt_alldataset = []
        sfreq_alldataset = []
        stcs_metadata_alldataset = pd.DataFrame()
        times_alldataset = []
        label_vertidx_dict_alldataset = []

        bids_path_all = mne_bids.BIDSPath(
                root=deriv_root, 
                subject= subject, 
                session= exp_id,  
                datatype='meg',  
                suffix=f"{epoch_data_name_list_name}_{VertROI_type}",
                extension='.pkl',
                check=False)
        os.makedirs(os.path.dirname( bids_path_all.fpath), exist_ok=True)      
        pklfile_name_all = bids_path_all.fpath

        if  os.path.exists(pklfile_name_all) :
            print('------------',epoch_data_name,'exist subject' ,subject,
                    '\n',config_filename,
                    '\n',datetime.now(),
            )
            with open(pklfile_name_all, 'rb') as pickle_file_load:
                loaded_data_all = pickle.load(pickle_file_load)
            print('------------',epoch_data_name,'Successfully loaded' ,subject,
                '\n',config_filename,
                '\n',datetime.now(),
                    )
            continue
        
    for epoch_data,epoch_data_name in zip(epoch_data_list,
                                          epoch_data_name_list):
        bids_path = mne_bids.BIDSPath(
                root=deriv_root, 
                subject= subject, 
                session= epoch_data['exp_id'],  
                datatype='meg',  
                task=epoch_data['task_id'],
                suffix=f"{epoch_data_name}_{VertROI_type}",
                extension='.pkl',
                check=False)

        os.makedirs(os.path.dirname( bids_path.fpath), exist_ok=True)      
        pklfile_name = bids_path.fpath

        #% get the results
        if  os.path.exists(pklfile_name) :
            print('------------',epoch_data_name,'exist subject' ,subject,
                    '\n',config_filename,
                    '\n',datetime.now(),
            )
            
            
            with open(pklfile_name, 'rb') as pickle_file_load:
                loaded_data = pickle.load(pickle_file_load)
            print(loaded_data.keys())
            print('------------',epoch_data_name,'Successfully loaded' ,subject,
                '\n',config_filename,
                '\n',datetime.now(),
                    )    
            
            if 'epoch_data_name_list' in param.keys():
                # append to alldataset
                tlt_alldataset.append(loaded_data["tlt"])
               
                sfreq_alldataset.append(loaded_data["sfreq"])

                times_alldataset.append(loaded_data["times"])

                label_vertidx_dict_alldataset.append(loaded_data["label_vertidx_dict"])

                # reset index and add epoch_data_name column
                aa = loaded_data['stcs_metadata'].reset_index(drop=True)
                aa.insert(0, 'epoch_data_name', epoch_data_name)
                stcs_metadata_alldataset = pd.concat([stcs_metadata_alldataset, aa], 
                                                     ignore_index=True)

                del loaded_data
        else:
            print('------------',epoch_data_name,'start subject' ,subject,
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
            
            # get label info
            lb_use = read_labels_exp2(
                bids_paths = BidsPath(subject_id = subject, 
                                        visit_id   =  epoch_data['exp_id']), 
                parc       = roi_params[roi]['parc'], 
                labels_list= roi_params[roi]['labels_list'],
                rois_list  = roi_params[roi]['rois_list'], 
                combine_labels_list=True, 
                merge_hemi=True
                ) 
            # !!!!! specific for VertROI
            assert len(lb_use) ==1, f'error: {len(lb_use)} label in VertROI, should be 1'
            
            label_all = [v for k,v in lb_use.items()][0] 
            label_all_rename = label_all
            if subject in sub_no_T1:
                print('no T1, use fsaverage')
                label_all_rename.subject = 'fsaverage'
            elif label_all.subject and not label_all.subject.startswith("sub-"):
                label_all_rename.subject = f"sub-{label_all.subject}"
            else:
                raise ValueError('error in label subject name')
            print('label_all_rename.subject\n',label_all_rename.subject)

        
            
            ## get vertno index
            label_vertidx_dict=\
                        get_label_vertidx(lb_use =lb_use,
                                        src = fw_inv_result ['src'],
                                            flag_return_dict =1)

            #  ---- get stc of all
            stcs_results = get_stcs_allsrc (
                                subject = subject,
                                **epoch_setting,
                                **general_param['source_method_setting'],
                                **fw_inv_result,
                                **eps_use_result)

            t_trllist =  mne.extract_label_time_course(
                                stcs   = stcs_results['stcs_epoch'], 
                                labels = label_all_rename, 
                                src    = fw_inv_result['src'], 
                                mode   ='pca_flip', 
                                allow_empty=False, 
                                return_generator=False,
                                mri_resolution=True, 
                                verbose=True)
            tlt = np.array([np.squeeze(x) for x in t_trllist])# shape: n_trials * n_timepoints
            print('tlt shape', tlt.shape)

            assert tlt.shape == (len(stcs_results['stcs_metadata']), len(stcs_results['times'])) ,\
                "Mismatch in shape of tlt and stcs_results"

            if 'epoch_data_name_list' in param.keys():
                # append to alldataset
                tlt_alldataset.append(tlt)
               
                sfreq_alldataset.append(stcs_results['sfreq'])

                times_alldataset.append(stcs_results['times'])

                label_vertidx_dict_alldataset.append(label_vertidx_dict)

                # reset index and add epoch_data_name column
                aa = stcs_results['stcs_metadata'].reset_index(drop=True)
                aa.insert(0, 'epoch_data_name', epoch_data_name)
                stcs_metadata_alldataset = pd.concat([stcs_metadata_alldataset, aa], ignore_index=True)

                del tlt, stcs_results, label_vertidx_dict

            else:
                # save trial* time for each ROI, 
                # here trial contains trials of all conditions that will be used in future syn analysis
                save_data = {
                            'tlt':tlt,
                            'param_config':param,
                            'general_param_config':general_param,
                            'sfreq':stcs_results['sfreq'],
                            'stcs_metadata':stcs_results['stcs_metadata'],
                            'times'   : stcs_results ['times'],
                            'label_vertidx_dict':label_vertidx_dict,
                            'roi_params':roi_params,
                            } 

                with open(pklfile_name, 'wb') as pickle_file:
                    pickle.dump(save_data, pickle_file)
                

                print('------------',epoch_data_name,'save' ,
                    '\n',pklfile_name,
                    '\n',subject,
                    '\n',config_filename,
                    '\n',datetime.now(),
                        )
                
    if 'epoch_data_name_list' in param.keys():
        # check the consistency of sfreq, times,label_vertidx_dict
        assert len(sfreq_alldataset)==len(epoch_data_name_list), "length of sfreq_alldataset not equal to epoch_data_name_list"
        assert(len(set(sfreq_alldataset))==1), "inconsistent sfreq across epoch_data"
        sfreq = sfreq_alldataset[0]
        for t1, t2 in zip(times_alldataset[:-1], times_alldataset[1:]):
            assert(np.array_equal(t1, t2)), "inconsistent times across epoch_data"
        times = times_alldataset[0]
        for d1, d2 in zip(label_vertidx_dict_alldataset[:-1], label_vertidx_dict_alldataset[1:]):
            assert(np.array_equal(d1, d2)), "inconsistent label_vertidx_dict across epoch_data"
        label_vertidx_dict = label_vertidx_dict_alldataset[0]

        save_data = {
                    'tlt':np.concatenate(tlt_alldataset, axis=0),
                    'param_config':param,
                    'general_param_config':general_param,
                    'roi_params':roi_params,
                    'sfreq':sfreq,
                    'times'   : times,
                    'label_vertidx_dict':label_vertidx_dict,
                    'stcs_metadata':stcs_metadata_alldataset,
                    } 

        with open(pklfile_name_all, 'wb') as pickle_file:
            pickle.dump(save_data, pickle_file)

        print('------','final all datasets save' ,
            '\n',pklfile_name_all,
            '\n',subject,
            '\n',config_filename,
            '\n',datetime.now(),
                )


#%%