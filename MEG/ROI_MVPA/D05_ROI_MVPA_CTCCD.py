
"""
====================
D10. Decoding for MEG on source space of ROI
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CTCCD: Cross Time Cross Condition Decoding
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

from D_MEG_function_E2 import set_path_ROI_MVPA, ATdata,sensor_data_for_ROI_MVPA_dAT,sensor_data_for_ROI_MVPA_AT
from D_MEG_function_E2 import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA,get_lables

####if need pop-up figures
# %matplotlib qt5
#mpl.use('Qt5Agg')

parser=argparse.ArgumentParser()
parser.add_argument('--sub',type=str,default='SB006',help='subject_id')
parser.add_argument('--visit',
                    type=str,
                    default='V2',
                    help='visit_id (e.g. "V2")')
# parser.add_argument('--cT',type=str,nargs='*', default=['500ms','1000ms','1500ms'], help='condition in Time duration')
# parser.add_argument('--cC',type=str,nargs='*', default=['FO'],
#                     help='selected decoding category, FO for face and object, LF for letter and false,'
#                          'F for face ,O for object, L for letter, FA for false')
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
parser.add_argument('--out_fw',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward',
                    help='Path to the forward (derivative) directory')
parser.add_argument('--nF',
                    type=int,
                    default=30,
                    help='number of feature selected for source decoding')
parser.add_argument('--nT',
                    type=int,
                    default=3,
                    help='number of trial averaged for source decoding')
parser.add_argument('--nPCA',
                    type=float,
                    default=0.95,
                    help='percentile of PCA selected for source decoding')
# parser.add_argument('--coreg_path',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/coreg',
#                     help='Path to the coreg (derivative) directory')


opt = parser.parse_args()
# con_C = opt.cC
# con_D = opt.cD
# con_T = opt.cT
select_F = opt.nF
n_trials = opt.nT
nPCA = opt.nPCA


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path


    # Now we define a function to decoding condition for one subject
    # Category_CCD, train on condition A, test on condition B
def Category_CTCCD(epochs_rs1,stcs1,epochs_rs2,stcs2,index_name,conditions_C,select_F,n_trials,roi_name,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        #StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        #SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #PCA(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    sliding = GeneralizingEstimator(clf, scoring=score_methods, n_jobs=-1)


    print(' Creating evoked datasets')
    
    if len(conditions_C)==2:
        

        temp1 = epochs_rs1.events[:, 2]
        temp1[epochs_rs1.metadata[index_name] == conditions_C[0]] = 1  # face
        temp1[epochs_rs1.metadata[index_name] == conditions_C[1]] = 2 # object
    
        y1 = temp1
        X1 = np.array([stc.data for stc in stcs1])
        
        temp2 = epochs_rs2.events[:,2]
        temp2[epochs_rs2.metadata[index_name] == conditions_C[0]] = 1
        temp2[epochs_rs2.metadata[index_name] == conditions_C[1]] = 2
        
        y2 = temp2
        X2 = np.array([stc.data for stc in stcs2])
    
    elif len(conditions_C)==4:
        temp1 = epochs_rs1.events[:, 2]
        temp1[epochs_rs1.metadata[index_name] == conditions_C[0]] = 1  # right
        temp1[epochs_rs1.metadata[index_name] == conditions_C[1]] = 1 # right
        temp1[epochs_rs1.metadata[index_name] == conditions_C[2]] = 2  # left
        temp1[epochs_rs1.metadata[index_name] == conditions_C[3]] = 2 # left

        y1 = temp1
        X1=np.array([stc.data for stc in stcs1])
        
        temp2 = epochs_rs2.events[:, 2]
        temp2[epochs_rs2.metadata[index_name] == conditions_C[0]] = 1  # right
        temp2[epochs_rs2.metadata[index_name] == conditions_C[1]] = 1 # right
        temp2[epochs_rs2.metadata[index_name] == conditions_C[2]] = 2  # left
        temp2[epochs_rs2.metadata[index_name] == conditions_C[3]] = 2 # left

        y2 = temp2
        X2=np.array([stc.data for stc in stcs2])
    
    
    
    
    

    

    

    # # Run cross-validated decoding analyses:
    # scores_a = cross_val_multiscore(sliding,X=X[cond_a], y=y[cond_a],cv=5,n_jobs=-1)
    # # Run cross-validated decoding analyses:
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=-1)
    
    ctccd=dict()
    

    group_xa=X1
    group_ya=y1
    group_xb=X2
    group_yb=y2
    
    scores_ab_per=np.zeros([100,group_xa.shape[2],group_xa.shape[2]])
    scores_ba_per=np.zeros([100,group_xb.shape[2],group_xb.shape[2]])
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
            n_psu_b=data_b.shape[0]//3
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
        new_xa=ATdata(new_xa,nbin=5)
        
        new_xb = np.concatenate((new_xb[0],new_xb[1]),axis=0)
        new_yb = np.array(new_yb)
        
        # average temporal feature (5 point average)
        new_xb=ATdata(new_xb,nbin=5)
        
        # First: train condition a (cond_a) and Test on condition b (cond_b) cross condition decoding
        # Fit
        sliding.fit(X=new_xa, y=new_ya)
        # Test
        scores_ab = sliding.score(X=new_xb, y=new_yb)


        scores_ab_per[num_per,:,:]=scores_ab
        
        # Then: train condition b (cond_b) and Test on condition a (cond_a) cross condition decoding
        # Fit
        sliding.fit(X=new_xb, y=new_yb)
        # Test
        scores_ba = sliding.score(X=new_xa, y=new_ya)


        scores_ba_per[num_per,:,:]=scores_ba

    

    
    # ccd['IR'] = np.mean(scores_a, axis=0)
    # ccd['RE'] = np.mean(scores_b, axis=0)
    ctccd['A2B'] = np.mean(scores_ab_per, axis=0)
    ctccd['B2A'] = np.mean(scores_ba_per, axis=0)


    
    fig, axes = plt.subplots(1, 2,figsize=(10,3),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    fig.suptitle('CTWCD')
    
    t = 1e3 * epochs_rs1.times
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
    im = axes[0].imshow(gaussian_filter(ctccd['A2B'],sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                   extent=epochs_rs1.times[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Testing Time (s)')
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('Train Seen Test Seen')
    axes[0].axvline(0, color='k')
    axes[0].axhline(0, color='k')
    axes[0].axline((0, 0), slope=1, color='k')
    plt.colorbar(im, ax=axes[0],fraction=0.03, pad=0.05)

    im = axes[1].imshow(gaussian_filter(ctccd['B2A'], sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                   extent=epochs_rs1.times[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Testing Time (s)')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('Train Unseen Test Unseen')
    axes[1].axvline(0, color='k')
    axes[1].axhline(0, color='k')
    axes[1].axline((0,0), slope=1, color='k')
    plt.colorbar(im, ax=axes[1],fraction=0.03, pad=0.05)
    
    fig.savefig(fname_fig)

    return ctccd




# =============================================================================
# RUN
# =============================================================================


# run roi decoding analysis

if __name__ == "__main__":
    
    #opt INFO
    
    #subject_id = 'SA111'
    #
    visit_id = 'V2'
    space = 'surface'
    #

    # analysis info
    
    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']
    
    
    analysis_name='CTCCD_PFC_full'
    task1_info='replay'
    task2_info='vg'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    roi_data_root, roi_figure_root, roi_code_root = set_path_ROI_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    # 2 Get Sub ROI
    if analysis_name=='CTCCD_PFC_full':
        surf_label_list, ROI_Name = sub_ROI_for_ROI_MVPA(fpath_fs, subject_id,analysis_name)
    elif analysis_name=='CTCCD_IIT_full':
        Roi_list='IIT'
        surf_label_list,ROI_Name = get_lables(fpath_fs, subject_id,Roi_list)
    elif analysis_name=='CTCCD_subPFC_full':
        Roi_list='PFC_subROI'
        surf_label_list,ROI_Name = get_lables(fpath_fs,subject_id,Roi_list)
        
    # 3 prepare the sensor data
    epochs_rs1, \
    rank1, common_cov1= sensor_data_for_ROI_MVPA_AT(fpath_epo,sub_info)
    
    if analysis_name=='CTCCD_IIT_full':
    
        epochs_rs1= epochs_rs1['Trial_type in {}'.format(['Non-Target'])]
    
    epochs_rs2, \
    rank2, common_cov2= sensor_data_for_ROI_MVPA_dAT(fpath_epo,sub_info)
    
    epochs_rs2= epochs_rs2['Response in {}'.format(['Seen'])]
    
    
    
    
    
    
    

    
    
    roi_ctccd_cat = dict()
    roi_ctccd_loc = dict()
    
    

    for nroi, roi_name in enumerate(ROI_Name):

        # 4 Get Source Data for each ROI
        stcs1 = []
        stcs1 = source_data_for_ROI_MVPA(epochs_rs1, fpath_fw, rank1, common_cov1, sub_info, surf_label_list[nroi],task1_info)
        
        stcs2 = []
        stcs2 = source_data_for_ROI_MVPA(epochs_rs2, fpath_fw, rank2, common_cov2, sub_info, surf_label_list[nroi],task2_info)
        
        
        


        ### WCD
        
        
        

        
        score_methods=make_scorer(accuracy_score)
        
        #Category
        index_name='Stimuli_type'
        conditions_C=['Face','Object']
        
        fname_fig_cat = op.join(roi_figure_root, 
                            sub_info  + 'Cat' + '_' + roi_name + "_acc_CCD" + '.png')
        
        ctccd_acc_cat= Category_CTCCD(epochs_rs1,stcs1,epochs_rs2,stcs2,
                              index_name,conditions_C,
                              select_F,n_trials,roi_name,
                              score_methods,fname_fig_cat)

        roi_ctccd_cat[roi_name] = ctccd_acc_cat
        
        
        #Location
        index_name='Location'
        conditions_C=['Upper Right','Lower Right','Upper Left','Lower Left']
        
        fname_fig_loc = op.join(roi_figure_root, 
                            sub_info  + 'Loc' + '_' + roi_name + "_acc_CCD" + '.png')
        
        ctccd_acc_loc= Category_CTCCD(epochs_rs1,stcs1,epochs_rs2,stcs2,
                              index_name,conditions_C,
                              select_F,n_trials,roi_name,
                              score_methods,fname_fig_loc)

        roi_ctccd_loc[roi_name] = ctccd_acc_loc
        


        
    roi_data=dict()
    
    
    roi_data['ctccd_acc_cat']=roi_ctccd_cat
    roi_data['ctccd_acc_loc']=roi_ctccd_loc
    


    fname_data=op.join(roi_data_root, sub_info + '_'  +"CTCCD_ROIs_data" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(roi_data,fw)
    fw.close()


# Save code
#    shutil.copy(__file__, roi_code_root)
