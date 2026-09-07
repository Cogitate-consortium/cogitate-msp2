"""
====================
D08. Group analysis for decoding pattern
====================



"""

import os.path as op
import os
import argparse

import pickle

from config import bids_root
from sublist_exp2 import sub_list

parser = argparse.ArgumentParser()
parser.add_argument('--visit',
                    type=str,
                    default='V2',
                    help='visit_id (e.g. "V2")')
# parser.add_argument('--cT', type=str, nargs='*', default=['500ms','1000ms','1500ms'],
#                     help='condition in Time duration:  [500ms],[1000ms],[1500ms]')
# parser.add_argument('--cC', type=str, nargs='*', default=['FO'],
#                     help='selected decoding category, FO for face and object, LF for letter and false')
# parser.add_argument('--cD',type=str,nargs='*', default=['Irrelevant', 'Relevant non-target'],
#                     help='selected decoding Task, Relevant non Target or Irrelevant condition')
parser.add_argument('--space',
                    type=str,
                    default='surface',
                    help='source space ("surface" or "volume")')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--analysis',
                    type=str,
                    default='AT',
                    help='the name for anlaysis, e.g. Cat or Ori or GAT_Cat')
parser.add_argument('--nF',
                    type=int,
                    default=30,
                    help='number of feature selected for source decoding')
parser.add_argument('--nT',
                    type=int,
                    default=5,
                    help='number of trial averaged for source decoding')
parser.add_argument('--nPCA',
                    type=float,
                    default=0.95,
                    help='percentile of PCA selected for source decoding')


opt = parser.parse_args()

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path
#analysis_name=opt.analysis
analysis_name='ET_WCD'


opt = parser.parse_args()


# if analysis_name=='Cat' or analysis_name=='Ori':
#     if methods_name=='T_all':
#         con_T=['500ms','1000ms','1500ms']
#     else:
#         con_T = methods_name[0]
    

select_F = opt.nF
n_trials = opt.nT
nPCA = opt.nPCA


visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path



    
if analysis_name=='ET_WCD':
    decoding_path=op.join(bids_root,'derivatives','decoding','dAT_mvpa')
else:
    decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa_e2')

data_path=op.join(decoding_path,analysis_name)

# Set path to group analysis derivatives
group_deriv_root = op.join(data_path, "group")
if not op.exists(group_deriv_root):
    os.makedirs(group_deriv_root)

# evokeds_group = []
# stc_group = []


#sb_list=[ 'SA111','SA148', 'SB040', 'SB069','SB081'] #'SA111

sb_list=sub_list[analysis_name]

group_data=dict()
for i, sbn in enumerate(sb_list):
    # if 'SB' in sbn:
    # sub and visit info
    sub_info = 'sub-' + sbn + '_ses-' + visit_id
    
    sub_data_root = op.join(data_path,
                            f"sub-{sbn}", f"ses-{visit_id}", "meg",
                            "data")
    # if analysis_name=='Cat':
    #     pkl_name = "_ROIs_data_Cat"
    # elif analysis_name=='Ori':
    #     pkl_name = "_ROIs_data_Ori"
    # elif analysis_name=='GAT_Cat':
    #     pkl_name = "_ROIs_data_GAT_Cat"
    # elif analysis_name=='GAT_Ori':
    #     pkl_name = "_ROIs_data_GAT_Cat"
    # elif analysis_name=='RSA_Cat':
    #     pkl_name = "_ROIs_RSA_Cat"
    # elif analysis_name=='RSA_Ori':
    #     pkl_name = "_ROIs_RSA_Ori"
    # elif analysis_name=='RSA_ID':
    #     pkl_name = "_ROIs_RSA_ID"
      
    rsa_data=dict()
    if analysis_name == "AT_cmb_stf_full" :
        fname_data=op.join(sub_data_root, sub_info + '_'  +"Cat_ROIs_data_AT" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
        
    elif analysis_name == "AT_cmb_stf_full_loc" :
        fname_data=op.join(sub_data_root, sub_info + '_'  +"Loc_ROIs_data_AT" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
        
    elif analysis_name == "AT_loc" or analysis_name=='dAT_loc_full'  or analysis_name=='dAT_subP2F_loc_full' or analysis_name=='dAT_subPFC_loc_full' or analysis_name=='GAT_PFC_dAT_loc_full':
        fname_data=op.join(sub_data_root, sub_info + '_'  +"Loc_ROIs_data" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
        
    elif analysis_name == 'GAT_AT' or analysis_name=='CCD_IIT_full' or analysis_name=='CCD_subPFC_full':
        fname_data=op.join(sub_data_root, sub_info + '_'  +"CCD_ROIs_data" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
        
    elif analysis_name == 'WCD_full' or analysis_name == 'WCD_subPFC_full':
        fname_data=op.join(sub_data_root, sub_info + '_'  +"WCD_ROIs_data" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data    
    
    elif analysis_name == 'CTCCD_PFC_full' or analysis_name == 'CTCCD_subPFC_full':
        fname_data=op.join(sub_data_root, sub_info + '_'  +"CTCCD_ROIs_data" + '.pickle')
        
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
    
    elif analysis_name == 'ET_WCD':
        fname_data=op.join(sub_data_root, sub_info + '_vg_dAT_ET_data_acc.pickle')
        fr=open(fname_data,'rb')
        et_data=pickle.load(fr)
        group_data[sbn]=et_data
        
        
    
        
    # elif analysis_name == "Ori_PFC":
    #     fname_data=op.join(sub_data_root, sub_info + '_' + task_info +'_IITPFC_data_Ori.pickle')
    #     fr=open(fname_data,'rb')
    #     roi_data=pickle.load(fr)
    #     group_data[sbn]=roi_data
            
    else:       
        fname_data=op.join(sub_data_root, sub_info + '_'  +"Cat_ROIs_data" + '.pickle')
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data

fname_data=op.join(group_deriv_root, "data_group_" + analysis_name +
                   '.pickle')
fw = open(fname_data,'wb')
pickle.dump(group_data,fw)
fw.close()


