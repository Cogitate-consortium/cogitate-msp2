
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

from D_MEG_function import set_path_ROI_MVPA, ATdata,sensor_data_for_ROI_MVPA
from D_MEG_function import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA

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

# select_F = 30
# n_trials = 3
#nPCA = 0.95


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub
#subject_id = 'SB006'

visit_id = 'V2'



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

def Cat_CT(epochs1,epochs2,score_methods,fname_fig_ccd,fname_fig_ctccd):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
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

    sliding_CCD = SlidingEstimator(clf, scoring=score_methods, n_jobs=-1)
    sliding_CTCCD = GeneralizingEstimator(clf, scoring=score_methods, n_jobs=-1)

    print(' Creating evoked datasets')
    
    
    condition_Stim=['Face','Object']
    #condition_Trial=['Filler','Probe']
    #condition_Trial=['Seen','Unseen']
    # #1) Select Category
    #     condition_Stim=['Face','Object','Blank']
    #     #condition_Trial=['Target','Non-Target']
        
    
    
    # for condition in condition_Trial:
    #     # Select epochs by condition
    #     if condition== 'Non-Target':
    #         cond_name='Ntarget'
    #     else:
    #         cond_name='Target'
    
    epochT1 = epochs1
    epochT1.filter(0.1,40).resample(100).crop(-0.5,1).apply_baseline(baseline=(-0.5,-0.25))
        
        
    temp1 = epochT1.events[:, 2]
    temp1[epochT1.metadata['Stimuli_type'] == condition_Stim[0]] = 1  # face
    temp1[epochT1.metadata['Stimuli_type'] == condition_Stim[1]] = 2 # object

    y1 = temp1
    X1=epochT1.pick_types(meg=True).get_data()
    
    epochT2 = epochs2
    epochT2.filter(0.1,40).resample(100).crop(-0.5,1).apply_baseline(baseline=(-0.5,-0.25))
        
        
    temp2 = epochT2.events[:, 2]
    temp2[epochT2.metadata['Stimuli_type'] == condition_Stim[0]] = 1  # face
    temp2[epochT2.metadata['Stimuli_type'] == condition_Stim[1]] = 2 # object

    y2 = temp2
    X2=epochT2.pick_types(meg=True).get_data()
    # cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
    # #         # Find indices of Irrelevant trials
    # cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]

    CT_CCD=dict()
    CT_CTCCD=dict()
    group_xa=X1
    group_ya=y1
    group_xb=X2
    group_yb=y2
    
    scores_ab_per_ccd=np.zeros([100,X1.shape[2]])
    scores_ab_per_ctccd=np.zeros([100,X1.shape[2],X1.shape[2]])
    scores_ba_per_ccd=np.zeros([100,X2.shape[2]])
    scores_ba_per_ctccd=np.zeros([100,X2.shape[2],X2.shape[2]])
    for num_per in range(100):
        # do the average trial
        new_xa = []
        new_ya = []
        new_xb = []
        new_yb = []
        for label in range(2):
            # Extract the data:
            data_a = group_xa[np.where(group_ya == label+1)]
            data_a = np.take(data_a, np.random.permutation(data_a.shape[0]), axis=0)
            n_psu_a=data_a.shape[0]//3
            avg_xa = block_reduce(data_a, block_size=tuple([n_psu_a, *[1] * len(data_a.shape[1:])]),
                                 func=np.nanmean, cval=np.nan)
            #block_size
            #array_like or int
            #Array containing down-sampling integer factor along each axis. Default block_size is 2.
            
            # funccallable
            # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.
            
            # cvalfloat
            # Constant padding value if image is not perfectly divisible by the block size.
            
            # Now generating the labels and group:
            new_xa.append(avg_xa)
            new_ya += [label] * avg_xa.shape[0]
            
            # Extract the data:
            data_b = group_xb[np.where(group_yb == label+1)]
            data_b = np.take(data_b, np.random.permutation(data_b.shape[0]), axis=0)
            n_psu_b=data_b.shape[0]//10
            avg_xb = block_reduce(data_b, block_size=tuple([n_psu_b, *[1] * len(data_b.shape[1:])]),
                                 func=np.nanmean, cval=np.nan)
            #block_size
            #array_like or int
            #Array containing down-sampling integer factor along each axis. Default block_size is 2.
            
            # funccallable
            # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.
            
            # cvalfloat
            # Constant padding value if image is not perfectly divisible by the block size.
            
            # Now generating the labels and group:
            new_xb.append(avg_xb)
            new_yb += [label] * avg_xb.shape[0]

        new_xa = np.concatenate((new_xa[0],new_xa[1]),axis=0)
        new_ya = np.array(new_ya)
        
        # average temporal feature (5 point average)
        new_xa=ATdata(new_xa)
        
        new_xb = np.concatenate((new_xb[0],new_xb[1]),axis=0)
        new_yb = np.array(new_yb)
        
        # average temporal feature (5 point average)
        new_xb=ATdata(new_xb)
        
        # First: train condition a (cond_a) and Test on condition b (cond_b) cross condition decoding
        # Fit
        sliding_CCD.fit(X=new_xa, y=new_ya)
        # Test
        scores_ab_CCD = sliding_CCD.score(X=new_xb, y=new_yb)


        scores_ab_per_ccd[num_per,:]=scores_ab_CCD
        
        # Then: train condition b (cond_b) and Test on condition a (cond_a) cross condition decoding
        # Fit
        sliding_CCD.fit(X=new_xb, y=new_yb)
        # Test
        scores_ba_CCD = sliding_CCD.score(X=new_xa, y=new_ya)


        scores_ba_per_ccd[num_per,:]=scores_ba_CCD

        # First: train condition a (cond_a) and Test on condition b (cond_b) cross condition decoding
        # Fit
        sliding_CTCCD.fit(X=new_xa, y=new_ya)
        # Test
        scores_ab_ctccd = sliding_CTCCD.score(X=new_xb, y=new_yb)


        scores_ab_per_ctccd[num_per,:,:]=scores_ab_ctccd
        
        # Then: train condition b (cond_b) and Test on condition a (cond_a) cross condition decoding
        # Fit
        sliding_CTCCD.fit(X=new_xb, y=new_yb)
        # Test
        scores_ba_ctccd = sliding_CTCCD.score(X=new_xa, y=new_ya)


        scores_ba_per_ctccd[num_per,:,:]=scores_ba_ctccd
    

    
    # ccd['IR'] = np.mean(scores_a, axis=0)
    # ccd['RE'] = np.mean(scores_b, axis=0)
    CT_CCD['A2B'] = np.mean(scores_ab_per_ccd, axis=0)
    CT_CCD['B2A'] = np.mean(scores_ba_per_ccd, axis=0)
    CT_CTCCD['A2B'] = np.mean(scores_ab_per_ctccd, axis=0)
    CT_CTCCD['B2A'] = np.mean(scores_ba_per_ctccd, axis=0)
            
        
        
    # wcd=dict()
    # scores_a= cross_val_multiscore(sliding, X=X[cond_a], y=y[cond_a], cv=5, n_jobs=1)
    # wcd[conditions_D[0]]=np.mean(scores_a, axis=0)
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=1)
    # wcd[conditions_D[1]] = np.mean(scores_b, axis=0)

    # pattern = dict()
    # pattern['IR'] = coef_a
    # pattern['RE'] = coef_b

    #CCD
    fig, ax = plt.subplots(1)
    t = 1e3 * epochT1.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Ti_name in CT_CCD.items():
        ax.plot(t, gaussian_filter1d(Ti_name,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title('Sensor_CCD_cat')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    #mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig_ccd)
    
    #CTCCD
    fig, axes = plt.subplots(1, 2,figsize=(10,10),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    fig.suptitle('CTCCD')
    
    t = 1e3 * epochT1.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    cmap = mpl.cm.jet
    vmin = 0.5
    vmax = 0.7
    bounds = np.linspace(vmin, vmax, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # diff setting
    vmind = -0.15
    vmaxd = 0.15
    boundsd = np.linspace(vmind, vmaxd, 11)
    normd = mpl.colors.BoundaryNorm(boundsd, cmap.N)
    #plot
    im = axes[0].imshow(gaussian_filter(CT_CTCCD['A2B'],sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                   extent=epochT1.times[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Testing Time (s)')
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('CTCCD_A2B')
    axes[0].axvline(0, color='k')
    axes[0].axhline(0, color='k')
    axes[0].axline((0, 0), slope=1, color='k')
    plt.colorbar(im, ax=axes[0],fraction=0.03, pad=0.05)

    im = axes[1].imshow(gaussian_filter(CT_CTCCD['B2A'], sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                    extent=epochT1.times[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Testing Time (s)')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('CTCCD_B2A')
    axes[1].axvline(0, color='k')
    axes[1].axhline(0, color='k')
    axes[1].axline((0,0), slope=1, color='k')
    plt.colorbar(im, ax=axes[1],fraction=0.03, pad=0.05)

    # Save figure

    

    fig.savefig(fname_fig_ctccd)

    return CT_CCD,CT_CTCCD



# =============================================================================
# RUN
# =============================================================================


# run roi decoding analysis

if __name__ == "__main__":
    
    #opt INFO
    
    # subject_id = 'SB085'
    #
    # visit_id = 'V1'
    # space = 'surface'
    #

    # analysis info
    
    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']
    
    
    analysis_name='Sensor_CrossTask'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    dAT_data_root, dAT_figure_root, dAT_code_root = set_path_dAT_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    fname_fig_ccd = op.join(dAT_figure_root, 
                            sub_info  
                            + "_acc_sensor_cat_ccd" + '.png')
    
    fname_fig_ctccd = op.join(dAT_figure_root, 
                            sub_info  
                            + "_acc_sensor_cat_ctccd" + '.png')
    

    task1='replay'
    task2='vg'
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo1=op.join(fpath_epo,sub_info + '_task-'+ task1+ '_epo.fif')
    epochs1 = mne.read_epochs(fname_epo1,
                             preload=True,
                             verbose=True)
    
    fname_epo2=op.join(fpath_epo,sub_info + '_task-'+ task2+ '_epo.fif')
    epochs2 = mne.read_epochs(fname_epo2,
                             preload=True,
                             verbose=True)
    
    epochs2 = epochs2['Response in {}'.format(['Seen'])]
    score_methods=make_scorer(accuracy_score)
    
    CT_ccd,CT_ctccd= Cat_CT(epochs1,epochs2,score_methods,fname_fig_ccd,fname_fig_ctccd)
   
        

    
    sensor_CrossTalk=dict()
    
    
    sensor_CrossTalk['ccd']=CT_ccd
    sensor_CrossTalk['ctccd']=CT_ctccd


    fname_data=op.join(dAT_data_root, sub_info +"_sensor_CrossTask_Cat_acc" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(sensor_CrossTalk,fw)
    fw.close()


# Save code
#    shutil.copy(__file__, roi_code_root)
