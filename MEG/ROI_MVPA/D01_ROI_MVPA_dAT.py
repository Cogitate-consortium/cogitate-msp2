
"""
====================
D10. Decoding for MEG on source space of ROI
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CCD: Cross Condition Decoding
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
parser.add_argument('--sub',type=str,default='SA111',help='subject_id')
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

def Category_WCD(epochs_rs,stcs,
                 select_F,
                 n_trials,
 #                nPCA,
                 roi_name,score_methods,fname_fig):
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
    X=np.array([stc.data for stc in stcs])

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
    ax.set_title(f'WCD_ {roi_name}')
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
    
    #subject_id = 'SB08d1'
    #
    visit_id = 'V2'
    space = 'surface'
    #

    # analysis info
    
    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']
    
    
    analysis_name='dAT_full'
    task_info='vg'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    roi_data_root, roi_figure_root, roi_code_root = set_path_ROI_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    # 2 Get Sub ROI
    if analysis_name == 'dAT_full':
        surf_label_list, ROI_Name = sub_ROI_for_ROI_MVPA(fpath_fs, subject_id,analysis_name)
    elif analysis_name == 'dAT_subPFC_full':
        Roi_list='PFC_subROI'
        surf_label_list, ROI_Name = get_lables(fpath_fs,subject_id,Roi_list)

    # 3 prepare the sensor data # time comsumming
    fname_sensordata=op.join(roi_data_root,sub_info + '_dAT_sensorpare.pickle')
    if not os.path.isfile(fname_sensordata):
        
        epochs_rs, \
        rank, common_cov= sensor_data_for_ROI_MVPA_dAT(fpath_epo,sub_info)
        
        sensordict={}
        sensordict={'epochs':epochs_rs,'cov':common_cov,'rank':rank}
        
        fw = open(fname_sensordata,'wb')
        
        pickle.dump(sensordict,fw)
        fw.close()
        del sensordict
    else:
        fr=open(fname_sensordata,'rb')
        sensordata=pickle.load(fr)
        fr.close()
        epochs_rs=sensordata['epochs']
        common_cov=sensordata['cov']
        rank=sensordata['rank']
        del sensordata
    
    
    

    
    #roi_ccd_auc = dict()
    roi_wcd_acc = dict()
    #roi_wcd_auc = dict()
    

    for nroi, roi_name in enumerate(ROI_Name):

        # 4 Get Source Data for each ROI
        stcs = []
        stcs = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[nroi],task_info)
        
        
        


        ### WCD
        
        
        fname_fig_acc = op.join(roi_figure_root, 
                            sub_info  + 'Cat' + '_' + roi_name + "_acc_WCD" + '.png')

        
        score_methods=make_scorer(accuracy_score)
        score_methods='roc_auc'
        wcd_acc= Category_WCD(epochs_rs, stcs,
                              select_F,
                              n_trials,
                              # nPCA,
                              roi_name, score_methods,
                              fname_fig_acc)

        roi_wcd_acc[roi_name] = wcd_acc
        


        
    roi_data=dict()
    
    
    roi_data['wcd_acc']=roi_wcd_acc
    


    fname_data=op.join(roi_data_root, sub_info + '_'  +"Cat_ROIs_data" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(roi_data,fw)
    fw.close()


# Save code
#    shutil.copy(__file__, roi_code_root)
