
"""
====================
D01. Decoding for MEG on sensor
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  
classifier: SVM (linear)
feature: spatial pattern (S)

feature selection methods test

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

from skimage.measure import block_reduce

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold



from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects


#from config import no_eeg_sbj
#from config import site_id, subject_id, file_names, visit_id, data_path, out_path
from config import l_freq, h_freq, sfreq
from config import (bids_root, tmin, tmax)

from D_MEG_function_E2 import set_path_ROI_MVPA, ATdata,sensor_data_for_ROI_MVPA_dAT
from D_MEG_function_E2 import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA,get_lables

####if need pop-up figures
# %matplotlib qt5
#mpl.use('Qt5Agg')

parser=argparse.ArgumentParser()
parser.add_argument('--sub',type=str,default='SA101',help='subject_id')
parser.add_argument('--visit',
                    type=str,
                    default='V2',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--space',
                    type=str,
                    default='surface',
                    help='source space ("surface" or "volume")')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--out_fw',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward',
                    help='Path to the forward (derivative) directory')
# parser.add_argument('--nF',
#                     type=int,
#                     default=30,
#                     help='number of feature selected for source decoding')
# parser.add_argument('--nT',
#                     type=int,
#                     default=5,
#                     help='number of trial averaged for source decoding')
# parser.add_argument('--nPCA',
#                     type=float,
#                     default=0.95,
#                     help='percentile of PCA selected for source decoding')
# parser.add_argument('--coreg_path',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/coreg',
#                     help='Path to the coreg (derivative) directory')


opt = parser.parse_args()

#select_F = 30
#n_trials = 3
#nPCA = 0.95


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub
#subject_id = 'SA148'

#visit_id = opt.visit
#visit_id = 'V2'



#space = opt.space
#subjects_dir = opt.fs_path


    # Now we define a function to decoding condition for one subject
    # Category, train on condition A, test on condition A
# set the path for decoding analysis
def set_path_dAT_MVPA(bids_root,subject_id, visit_id, analysis_name):
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
    roi_deriv_root = op.join(mvpa_deriv_root, "dAT_mvpa", analysis_name)
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

def Category_WCD(epochs_rs,et_data,
                 #select_F,
                 #n_trials,
 #                nPCA,
                 ch_index,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        #StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        #SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    sliding = SlidingEstimator(clf, scoring=score_methods, n_jobs=-1)


    print(' Creating evoked datasets')
    
    condition_Stim=['Face','Object']
    condition_Trial=['Seen','Unseen']

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Stimuli_type'] == condition_Stim[0]] = 1  # face
    temp[epochs_rs.metadata['Stimuli_type'] == condition_Stim[1]] = 2 # object

    y = temp
    X=et_data.get_data()

    # cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
    # #         # Find indices of Irrelevant trials
    # cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]
    
    subsamp_scores=np.zeros([100,X.shape[2]])
    
    con_index_seen=np.where(epochs_rs.metadata['Response'] == condition_Trial[0])[0]
    con_index_unseen=np.where(epochs_rs.metadata['Response'] == condition_Trial[1])[0]
    
    T_seen=con_index_seen.size
    T_unseen=con_index_unseen.size
    
    Trial_min=min(T_seen,T_unseen)

    wcd=dict()
    for condi in range(2):
        for subs_per in range(100):
            if condi==0:
                T_seen_stim_index=np.array(range(0,T_seen,1))
                np.random.shuffle(T_seen_stim_index)
                group_x=X[con_index_seen[:Trial_min]]
                group_y=y[con_index_seen[:Trial_min]]
            elif condi==1:
                T_unseen_stim_index=np.array(range(0,T_unseen,1))
                np.random.shuffle(T_unseen_stim_index)
                group_x=X[con_index_unseen[:Trial_min]]
                group_y=y[con_index_unseen[:Trial_min]]
                
                
                
            # con_index=np.where(epochs_rs.metadata['Response'] == condition_Trial[condi])[0]
            # group_x=X[con_index]
            # group_y=y[con_index]
            
            #scores_per=np.zeros([100,group_x.shape[2]])
            #for num_per in range(100):
            # do the average trial
            new_x = []
            new_y = []
            for label in range(2):
                # Extract the data:
                data = group_x[np.where(group_y == label+1)]
                data = np.take(data, np.random.permutation(data.shape[0]), axis=0)
                n_psu=data.shape[0]//3
                avg_x = block_reduce(data, block_size=tuple([n_psu, *[1] * len(data.shape[1:])]),
                                     func=np.nanmean, cval=np.nan)
                #block_size
                #array_like or int
                #Array containing down-sampling integer factor along each axis. Default block_size is 2.
                
                # funccallable
                # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.
                
                # cvalfloat
                # Constant padding value if image is not perfectly divisible by the block size.
                
                # Now generating the labels and group:
                new_x.append(avg_x)
                new_y += [label] * avg_x.shape[0]

            new_x = np.concatenate((new_x[0],new_x[1]),axis=0)
            new_y = np.array(new_y)
            
            # average temporal feature (5 point average)
            new_x=ATdata(new_x,nbin=5)
            
            scores= cross_val_multiscore(sliding, X=new_x, y=new_y, cv=3, n_jobs=1)
            #scores_per[num_per,:]=np.mean(scores, axis=0)
            
            subsamp_scores[subs_per,:]=np.mean(scores, axis=0) 
                
            
        wcd[condition_Trial[condi]]=np.mean(subsamp_scores, axis=0)       
            
        
        
    # wcd=dict()
    # scores_a= cross_val_multiscore(sliding, X=X[cond_a], y=y[cond_a], cv=5, n_jobs=1)
    # wcd[conditions_D[0]]=np.mean(scores_a, axis=0)
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=1)
    # wcd[conditions_D[1]] = np.mean(scores_b, axis=0)

    # pattern = dict()
    # pattern['IR'] = coef_a
    # pattern['RE'] = coef_b

    
    fig, ax = plt.subplots(1)
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Ti_name in wcd.items():
        ax.plot(t, gaussian_filter1d(Ti_name,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title(f'WCD_Cat_ {ch_index}')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    #mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig)

    return wcd

def Location_WCD(epochs_rs,et_data,
                 #select_F,
                 #n_trials,
 #                nPCA,
                 ch_index,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        #StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        #SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    sliding = SlidingEstimator(clf, scoring=score_methods, n_jobs=-1)


    print(' Creating evoked datasets')
    
    
    condition_Trial=['Seen','Unseen']

    condition_Stim=['Upper Right','Lower Right','Upper Left','Lower Left']
    

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Location'] == condition_Stim[0]] = 1  # right
    temp[epochs_rs.metadata['Location'] == condition_Stim[1]] = 1 # right
    temp[epochs_rs.metadata['Location'] == condition_Stim[2]] = 2  # left
    temp[epochs_rs.metadata['Location'] == condition_Stim[3]] = 2 # left


    y = temp
    X=et_data.get_data()

    # cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
    # #         # Find indices of Irrelevant trials
    # cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]
    
    subsamp_scores=np.zeros([100,X.shape[2]])
    
    con_index_seen=np.where(epochs_rs.metadata['Response'] == condition_Trial[0])[0]
    con_index_unseen=np.where(epochs_rs.metadata['Response'] == condition_Trial[1])[0]
    
    T_seen=con_index_seen.size
    T_unseen=con_index_unseen.size
    
    Trial_min=min(T_seen,T_unseen)

    wcd=dict()
    for condi in range(2):
        for subs_per in range(100):
            if condi==0:
                T_seen_stim_index=np.array(range(0,T_seen,1))
                np.random.shuffle(T_seen_stim_index)
                group_x=X[con_index_seen[:Trial_min]]
                group_y=y[con_index_seen[:Trial_min]]
            elif condi==1:
                T_unseen_stim_index=np.array(range(0,T_unseen,1))
                np.random.shuffle(T_unseen_stim_index)
                group_x=X[con_index_unseen[:Trial_min]]
                group_y=y[con_index_unseen[:Trial_min]]
                
                
                
            # con_index=np.where(epochs_rs.metadata['Response'] == condition_Trial[condi])[0]
            # group_x=X[con_index]
            # group_y=y[con_index]
            
            #scores_per=np.zeros([100,group_x.shape[2]])
            
            #for num_per in range(100):
                # do the average trial
            new_x = []
            new_y = []
            for label in range(2):
                # Extract the data:
                data = group_x[np.where(group_y == label+1)]
                data = np.take(data, np.random.permutation(data.shape[0]), axis=0)
                n_psu=data.shape[0]//3
                avg_x = block_reduce(data, block_size=tuple([n_psu, *[1] * len(data.shape[1:])]),
                                     func=np.nanmean, cval=np.nan)
                #block_size
                #array_like or int
                #Array containing down-sampling integer factor along each axis. Default block_size is 2.
                
                # funccallable
                # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.
                
                # cvalfloat
                # Constant padding value if image is not perfectly divisible by the block size.
                
                # Now generating the labels and group:
                new_x.append(avg_x)
                new_y += [label] * avg_x.shape[0]

            new_x = np.concatenate((new_x[0],new_x[1]),axis=0)
            new_y = np.array(new_y)
                
            # average temporal feature (5 point average)
            new_x=ATdata(new_x,nbin=5)
                
            scores= cross_val_multiscore(sliding, X=new_x, y=new_y, cv=3, n_jobs=1)
            #scores_per[num_per,:]=np.mean(scores, axis=0)
            
            subsamp_scores[subs_per,:]=np.mean(scores, axis=0) 
                
            
        wcd[condition_Trial[condi]]=np.mean(subsamp_scores, axis=0)       
            
        
        
    # wcd=dict()
    # scores_a= cross_val_multiscore(sliding, X=X[cond_a], y=y[cond_a], cv=5, n_jobs=1)
    # wcd[conditions_D[0]]=np.mean(scores_a, axis=0)
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=1)
    # wcd[conditions_D[1]] = np.mean(scores_b, axis=0)

    # pattern = dict()
    # pattern['IR'] = coef_a
    # pattern['RE'] = coef_b

    
    fig, ax = plt.subplots(1)
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Ti_name in wcd.items():
        ax.plot(t, gaussian_filter1d(Ti_name,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title(f'WCD_LOC_ {ch_index}')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    #mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig)

    return wcd


# =============================================================================
# RUN
# =============================================================================


# run roi decoding analysis

if __name__ == "__main__":
    
    #opt INFO
    
    #subject_id = 'SA114'
    #
    visit_id = 'V2'
    # space = 'surface'
    #

    # analysis info
    
    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']
    
    
    analysis_name='ET_WCD'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    dAT_data_root, dAT_figure_root, dAT_code_root = set_path_dAT_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    

    task='vg'
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-'+ task+ '_epo_ET.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True)
    
    epochs = epochs['Trial_type in {}'.format(['Probe'])]
    epochs.resample(100).crop(-0.5,1).apply_baseline(baseline=(-0.5,-0.25))
    
    #mark eyelink info channel, in order of :
    #1: Left X postion (LX),2: Left Y position (LY),3:Left Puiple size (LP)
    #4: Right X postion (RX),5: Right Y position (RY),6:Right Puiple size (RP)
    if subject_id[:2]=='SB':
        ET_index=['MISC002','MISC003','MISC007','MISC008','MISC009','MISC010']
        ET_index2=['MISC002','MISC003','MISC008','MISC009']
        ET_6ch = epochs.pick(ET_index)
        ET_4ch = epochs.pick(ET_index2)
    elif subject_id[:2]=='SA':
        ET_index=['MISC001','MISC002','MISC003']
        ET_index2=['MISC001','MISC002']
        ET_6ch = epochs.pick(ET_index)
        ET_4ch = epochs.pick(ET_index2)
        
    score_methods=make_scorer(accuracy_score)
    score_methods='roc_auc'
    ch_index=['4ch','6ch']
    ET_dAT=dict()
    #Category decoding
    for ch in range(2):
        fname_fig_acc = op.join(dAT_figure_root, 
                            sub_info  
                            + "_acc_dAT_Cat_" +ch_index[ch] + '.png')
        c_index='Cat_'+ch_index[ch]
        temp_acc = Category_WCD(epochs,ET_4ch,ch_index[ch],score_methods,fname_fig_acc)
        ET_dAT[c_index]=temp_acc
    
    #Location decoding
    for ch in range(2):
        fname_fig_acc = op.join(dAT_figure_root, 
                            sub_info  
                            + "_acc_dAT_Loc_" +ch_index[ch] + '.png')
        c_index='Loc_'+ch_index[ch]
        temp_acc = Location_WCD(epochs,ET_4ch,ch_index[ch],score_methods,fname_fig_acc)
        ET_dAT[c_index]=temp_acc
    

    
    


    fname_data=op.join(dAT_data_root, sub_info + '_' + task +"_dAT_ET_data_acc" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(ET_dAT,fw)
    fw.close()


# Save code
#    shutil.copy(__file__, roi_code_root)
