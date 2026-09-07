"""
===============
A01. Evoked data
===============

Compute averaged evoked responses.

@author: Ling Liu  ling.liu@pku.edu.cn

"""

import os.path as op
import os
# import sys
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import mne

from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import PIL
import pickle


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style='ticks')

from mne.stats import fdr_correction
from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_cluster_test

from bayes_factor_fun import bayes_ttest, bayes_binomtest, sim_decoding_binomial, sim_decoding_binomial_2d, kendall_bf_from_data

import matplotlib.patheffects as path_effects

import matplotlib.colors as mcolors

from scipy import stats as stats
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

from sklearn.model_selection import LeaveOneOut

#from config import subject_id, file_names, out_path
#from config import no_eeg_sbj
from config import (bids_root, tmin, tmax)
from config import plot_param
from plotters import plot_matrix, mm2inch, plot_time_series
import matplotlib as mpl
from sublist_exp2 import sub_list

# get the parameters dictionary
param = plot_param
colors=param['colors']
fig_size = param["figure_size_mm"]
plt.rc('font', size=8)  # controls default text size
plt.rc('axes', labelsize=20)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('xtick.major', width=2, size=4)
plt.rc('ytick.major', width=2, size=4)
plt.rc('legend', fontsize=18)
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

# Color parameters:
cmap = "RdYlBu_r"
# #color_blind_palette = sns.color_palette("col
#sub_list=['SA111','SA148','SB040','SB069','SB071']

visit_id='V2'
task='vg'
analysis_name='ET_WCD'

sub_list_a=sub_list[analysis_name]
sub_list=sub_list_a

# # 1 Set path to the data root path
# if task=='replay':
#     analysis_name='AT'
    
# elif task=='vg':
#     analysis_name='dAT'
    

decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa_e2')

data_path=op.join(decoding_path,analysis_name)    
    





### III  Set the Output Data Path
# Set path to decoding derivatives
Stat_deriv_root = op.join(data_path,'group')
if not op.exists(Stat_deriv_root):
    os.makedirs(Stat_deriv_root)
    

# 1) output_data
Stat_data_root= op.join(Stat_deriv_root,"data")
if not op.exists(Stat_data_root):
    os.makedirs(Stat_data_root)

# 2) output_figure
Stat_figure_root = op.join(Stat_deriv_root,"figures")
if not op.exists(Stat_figure_root):
    os.makedirs(Stat_figure_root)
    
 
    

def g2gdat(roi_g,time_point,sig1,sig2):
    roi_g_acc=np.mean(roi_g[:,:,:],axis=1)
    roi_g_ci=1.96*stats.sem(roi_g[:,:,30:251],axis=1)
    roi_g_dat=np.vstack((time_point,roi_g_acc,roi_g_ci,sig1,sig2))
    return roi_g_dat

def dat2g(dat,roi_name,cond_name,time_index,decoding_name,sub_list):
    roi_g=np.zeros([2,len(sub_list),time_index])
    for ci, cond in enumerate(cond_name):
        group_gc=np.zeros([len(sub_list),time_index])
        for i, sbn in enumerate(sub_list):
            group_gc[i,:]=dat[sbn][decoding_name][roi_name][cond]
           
        
        roi_g[ci,:,:]=group_gc*100
        
    return roi_g

def dat2g_ET(dat,roi_name,cond_name,time_index,sub_list):
    roi_g=np.zeros([2,len(sub_list),time_index])
    for ci, cond in enumerate(cond_name):
        group_gc=np.zeros([len(sub_list),time_index])
        for i, sbn in enumerate(sub_list):
            group_gc[i,:]=dat[sbn][roi_name][cond]
           
        
        roi_g[ci,:,:]=group_gc*100
        
    return roi_g

def dat2g_roi(dat,roi_name,cond_name,time_index,decoding_name,sub_list):
    roi_g=np.zeros([2,len(sub_list),time_index])
    for ci, cond in enumerate(roi_name):
        group_gc=np.zeros([len(sub_list),time_index])
        for i, sbn in enumerate(sub_list):
            group_gc[i,:]=dat[sbn][decoding_name][cond][cond_name]
           
        
        roi_g[ci,:,:]=group_gc*100
        
    return roi_g

def dat2gat(dat,roi_name,cond_name,decoding_name,sub_list):
    roi_gat=np.zeros([2,len(sub_list),151,151])
    for ci, cond in enumerate(cond_name):
        roi_gat_gc=np.zeros([len(sub_list),151,151])
        for i, sbn in enumerate(sub_list):
            roi_gat_gc[i,:,:]=dat[sbn][decoding_name][roi_name][cond]
           
        
        roi_gat[ci,:,:,:]=roi_gat_gc*100
        
    return roi_gat   

def df2csv(np_data,task_index,csv_fname):
    columns_index=['Time',
                   'ACC (' + task_index[0] + ')',
                   'CI (' + task_index[0] + ')',
                   'sig (' + task_index[0] + ')']
    df = pd.DataFrame(np_data.T, columns=columns_index)
    df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')


def gc2df(gc_mean,test_win_on,test_win_off,task_index,chance_index,time_point,sub_list):

    df1 = pd.DataFrame(gc_mean[0,:,:], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T1, pval1 = stats.ttest_1samp(gc_mean[0,:,test_win_on:test_win_off], chance_index)
    
    df2 = pd.DataFrame(gc_mean[1,:,:], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='Task',value=task_index[1])
    
    T2, pval2 = stats.ttest_1samp(gc_mean[1,:,test_win_on:test_win_off], chance_index)
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,T1,pval1,T2,pval2

def df_plot(ts_df,T1,pval1,T2,pval2,time_point,test_win_on,roi_name,task_index,chance_index,y_index,fname_fig):
    if roi_name=='GNW':
       window=[0.3,0.5,0.5,0.3]
    elif roi_name=='IIT':
        window=[0.3,1.5,1.5,0.3]    
    elif roi_name=='MT':
        window=[0.25,0.5,0.5,0.25]    
    elif roi_name=='FP':
        window=[0.3,1.5,1.5,0.3]    
    #plot with sns
    
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    reject_fdr1, pval_fdr1 = fdr_correction(pval1, alpha=0.05, method='indep')
    temp=reject_fdr1.nonzero()
    sig1=np.full(time_point.shape,np.nan)
    if len(temp[0])>=1:
        threshold_fdr1 = np.min(np.abs(T1)[reject_fdr1])
        T11=np.concatenate((np.zeros((test_win_on,)),T1))
        clusters1 = np.where(T11 > threshold_fdr1)[0]
        if len(clusters1)>1:
            clusters1 = clusters1[clusters1 > test_win_on]
            #times = range(0, 500, 10)
            plt.plot(time_point[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[task_index[0]])
            sig1[clusters1]=1
    
    reject_fdr2, pval_fdr2 = fdr_correction(pval2, alpha=0.05, method='indep')
    temp=reject_fdr2.nonzero()
    sig2=np.full(time_point.shape,np.nan)
    if len(temp[0])>=1:
        threshold_fdr2 = np.min(np.abs(T2)[reject_fdr2])
        T22=np.concatenate((np.zeros((test_win_on,)),T2))
        clusters2 = np.where(T22 > threshold_fdr2)[0]
        if len(clusters2)>1:
            clusters2 = clusters2[clusters2 > test_win_on]
            #times = range(0, 500, 10)
            plt.plot(time_point[clusters2], np.zeros(clusters2.shape) + 30, 'o', linewidth=3,color=colors[task_index[1]])
            sig2[clusters2]=1
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1])
    plt.ylim([15,80])
    plt.xticks([-0.5,0,0.5,1.0])
    plt.yticks([20,40,60,80])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1, sig2

def df_plot_cluster(ts_df,C1_stat,C2_stat,time_point,test_win_on,test_win_off,roi_name,task,chance_index,y_index,fname_fig):
    
    window=[0.3,0.5,0.5,0.3]
    if task=='replay':
        cond_name=['Target','Non-Target']
    elif task=='vg':
        cond_name=['Seen','Unseen']
    elif task=='ccd_acc_cat' or task=='ccd_acc_loc':
            cond_name=['A2B','B2A']
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on):(test_win_off)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[cond_name[0]])
                sig1[clusters1]=i
            
    temp2=C2_stat['cluster']
    temp_p2=C2_stat['cluster_p']
    sig2=np.full(time_point.shape,np.nan)
    if len(temp2)>=1:
        for i in range(len(temp2)):
            if temp_p2[i]<0.05:# plot the cluster which  p < 0.05
                clusters2=temp2[i][0]
                plt.plot(time_index[clusters2], np.zeros(clusters2.shape) + 30, 'o', linewidth=3,color=colors[cond_name[1]])
                sig2[clusters2]=i
    
    
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1])
    plt.ylim([15,80])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1, sig2


def df_plot_cluster_diff(ts_df,C1_stat,C2_stat,C3_stat,time_point,test_win_on,test_win_off,roi_name,task,chance_index,y_index,fname_fig):
    
    window=[0.3,0.5,0.5,0.3]
    # if task=='replay':
    #     cond_name=['Target','Non-Target']
    # elif task=='vg':
    cond_name=['Seen','Unseen']
    # elif task=='ccd_acc_cat' or task=='ccd_acc_loc':
    #         cond_name=['A2B','B2A']
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=True)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    # leg = g._legend
    # leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on):(test_win_off)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 30, 'o', linewidth=3,color=colors[cond_name[0]])
                sig1[clusters1]=i
            
    temp2=C2_stat['cluster']
    temp_p2=C2_stat['cluster_p']
    sig2=np.full(time_point.shape,np.nan)
    if len(temp2)>=1:
        for i in range(len(temp2)):
            if temp_p2[i]<0.05:# plot the cluster which  p < 0.05
                clusters2=temp2[i][0]
                plt.plot(time_index[clusters2], np.zeros(clusters2.shape) + 25, 'o', linewidth=3,color=colors[cond_name[1]])
                sig2[clusters2]=i
                
    temp3=C3_stat['cluster']
    temp_p3=C3_stat['cluster_p']
    sig3=np.full(time_point.shape,np.nan)
    if len(temp3)>=1:
        for i in range(len(temp3)):
            if temp_p3[i]<0.05:# plot the cluster which  p < 0.05
                clusters3=temp3[i][0]
                x_start, x_end = time_index[clusters3][0], time_index[clusters3][-1:]
                condition = (time_point >= x_start) & (time_point <= x_end)
                #plt.plot(time_index[clusters3], np.zeros(clusters3.shape) + 20, 'o', linewidth=3,color='black')
                plt.fill_between(time_point,C3_stat['cond1'],C3_stat['cond2'],where=condition,interpolate=True,color='red',alpha=0.3)
                sig3[clusters3]=i
    
    
   
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1])
    plt.ylim([15,80])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1, sig2,sig3



def df_plot_cluster_cmb(ts_df,C1_stat,time_point,test_win_on,test_win_off,roi_name,task,chance_index,y_index,fname_fig):
    
    #window=[0.3,0.5,0.5,0.3]
    
    # if task=='replay':
    #     cond_name=['Target','Non-Target']
    # elif task=='vg':
    #     cond_name=['Seen','Unseen']
    cond_name=task
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on):(test_win_off)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[cond_name[0]])
                sig1[clusters1]=i
            
    
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1.0])
    plt.ylim([15,85])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1

def df_plot_cluster_seen(ts_df,C1_stat,time_point,test_win_on,test_win_off,roi_name,task,chance_index,y_index,fname_fig):
    
    #window=[0.3,0.5,0.5,0.3]
    
    # if task=='replay':
    #     cond_name=['Target','Non-Target']
    # elif task=='vg':
    #     cond_name=['Seen','Unseen']
    cond_name=task
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on):(test_win_off)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[cond_name[0]])
                sig1[clusters1]=i
            
    
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1.5])
    plt.ylim([15,100])
    plt.xticks([-0.5,0,0.5,1,1.5])
    plt.yticks([20,40,60,80,100])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1

def df_plot_ROI_cluster(ts_df,C1_stat,time_point,test_win_on,test_win_off,chance_index,y_index,fname_fig):
    
    #window=[0.3,1.5,1.5,0.3]
    #print(fname_fig)
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('talk',rc=talk_rc,font_scale=1)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='ROI',aspect=2,palette=colors,legend=True)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #sns.move_legend(g, "upper left", bbox_to_anchor=(.72, .8), frameon=False)
    #leg = g._legend
    # leg.remove()
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-5, 'o', linewidth=3,color=colors['IIT'])
                sig1[clusters1]=i
    
    
            
    #plt.fill(window,[40,40,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1.0])
    plt.ylim([15,80])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1

def df_plot_cluster_GAT(gc_mean,C1_stat,C2_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    
    talk_rc={'lines.linewidth':1,'lines.markersize':1}
    sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    fig, axes = plt.subplots(1, 1,figsize=(12,10),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.subplots_adjust(left=0.2)
    
    
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    #cmap = mpl.cm.RdYlBu_r
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                     np.vstack((plt.cm.Blues_r(np.linspace(0, 1, 220) ),
                                                                plt.cm.Blues_r( np.linspace(1, 1, 36) ), 
                                                                plt.cm.Reds( np.linspace(0, 0, 36) ),
                                                                plt.cm.Reds( np.linspace(0, 1, 220) ) ) ) )
    vmin = 20
    vmax = 80
    # bounds = np.linspace(vmin, vmax, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #plot
    GAT_avg=np.mean(gc_mean[0,:,25:151,25:151],axis=0)
    GAT_avg_plot = np.nan * np.ones_like(GAT_avg)
    for c, p_val in zip(C1_stat['cluster'], C1_stat['cluster_p']):
        if p_val <= 0.05:
            GAT_avg_plot[c] = GAT_avg[c]
    
    im = axes.imshow(gaussian_filter(GAT_avg,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,alpha=0.9,
                    extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.contour(GAT_avg_plot > 0, GAT_avg_plot > 0, colors="black", linewidths=1.5, origin="lower",extent=t[[0, -1, 0, -1]])
    im = axes.imshow(GAT_avg_plot, origin='lower', cmap=cmap,aspect='auto',
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.set_xticks([-0.5,0,0.5,1])
    axes.set_yticks([-0.5,0,0.5,1])
    axes.set_title(task_index[0])
    axes.axvline(0, color='k',linestyle='--')
    axes.axhline(0, color='k',linestyle='--')
    axes.axline((0, 0), slope=1, color='k',linestyle='--')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    
    cb = axes.figure.axes[-1]
    m = axes.figure.axes[-2]
    pos = m.get_position().bounds
    cb.set_position([pos[2]+pos[0]+0.01, pos[1], 0.1, pos[3]])
    
    
    
    fname_fig_1=op.join(fname_fig+'_'+task_index[0]+'.svg')
    
    fig.savefig(fname_fig_1,format="svg", transparent=True, dpi=300)
    
    
    talk_rc={'lines.linewidth':1,'lines.markersize':1}
    sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    fig, axes = plt.subplots(1, 1,figsize=(12,10),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.subplots_adjust(left=0.2)
    
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    #cmap = mpl.cm.RdYlBu_r
    
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                     np.vstack((plt.cm.Blues_r(np.linspace(0, 1, 220) ),
                                                                plt.cm.Blues_r( np.linspace(1, 1, 36) ), 
                                                                plt.cm.Reds( np.linspace(0, 0, 36) ),
                                                                plt.cm.Reds( np.linspace(0, 1, 220) ) ) ) )
    vmin = 20
    vmax = 80
    # bounds = np.linspace(vmin, vmax, 11)
    
    GAT2_avg=np.mean(gc_mean[1,:,25:151,25:151],axis=0)
    GAT2_avg_plot = np.nan * np.ones_like(GAT2_avg)
    for c2, p2_val in zip(C2_stat['cluster'], C2_stat['cluster_p']):
        if p2_val <= 0.05:
            GAT2_avg_plot[c2] = GAT2_avg[c2]
    im = axes.imshow(gaussian_filter(GAT2_avg,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,alpha=0.9,
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.contour(GAT2_avg_plot > 0, GAT2_avg_plot > 0, colors="black", linewidths=1.5, origin="lower",extent=t[[0, -1, 0, -1]])
    im = axes.imshow(GAT2_avg_plot, origin='lower', cmap=cmap,aspect='auto',
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.set_xticks([-0.5,0,0.5,1])
    axes.set_yticks([-0.5,0,0.5,1])
    axes.set_title(task_index[1])
    axes.axvline(0, color='k',linestyle='--')
    axes.axhline(0, color='k',linestyle='--')
    axes.axline((0,0), slope=1, color='k',linestyle='--')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    plt.tight_layout()
    cb = axes.figure.axes[-1]
    m = axes.figure.axes[-2]
    pos = m.get_position().bounds
    cb.set_position([pos[2]+pos[0]+0.01, pos[1], 0.1, pos[3]])
    
    fname_fig_2=op.join(fname_fig+'_'+task_index[1]+'.svg')
    
    fig.savefig(fname_fig_2,format="svg", transparent=True, dpi=300)
    
def df_plot_subROI_cluster(ts_df,time_point,test_win_on,test_win_off,task_index,chance_index,y_index,fname_fig):
    
    #window=[0.3,0.5,0.5,0.3]
    
    
    #plot with sns
    
    talk_rc={'lines.linewidth':2,'lines.markersize':4}
    sns.set_context('talk',rc=talk_rc,font_scale=2)
    
    
    
    
    g = sns.relplot(x="Times(s)", y="Accuracy(%)", kind="line", data=ts_df,col='ROI',hue='ROI',aspect=4,palette=colors,col_wrap=5,legend=False)
    g.map(plt.axhline, y=50, color='k', linestyle='-', label='chance')
    g.map(plt.axvline, x=0, color='k', linestyle='-', label='onset')
    g.map(plt.axvline, x=0.5, color='gray', linestyle='--')
    
  
 
    
    g.fig.set_size_inches(mm2inch(fig_size[0])*5,mm2inch(fig_size[0])*2)
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
   
    
    #g.map( plt.axvline(1, color='gray', linestyle='--'))
    #g.map(plt.axvline(1.5, color='gray', linestyle='--'))
    
    
    # temp=C1_stat['cluster']
    # temp_p=C1_stat['cluster_p']
    # sig1=np.full(time_point.shape,np.nan)
    # time_index=time_point[(test_win_on-30):(test_win_off-30)]
    # if len(temp)>=1:
    #     for i in range(len(temp)):
    #         if temp_p[i]<0.05:# plot the cluster which  p < 0.05
    #             clusters1=temp[i][0]
    #             plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-4, 'o', linewidth=3,color=colors[task_index[0]])
    #             sig1[clusters1]=i
    
    
            
    #plt.fill(window,[chance_index-10,chance_index-10,chance_index+y_index,chance_index+y_index],facecolor='g',alpha=0.2)
    plt.xlim([-0.5,1])
    plt.ylim([chance_index-10,chance_index+y_index])
    plt.xticks([-0.5,0,0.5,1.0])
    plt.yticks([20,40,60,80,100])

    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
 

def stat_cluster_1sample_seen(gc_mean,sub_list_a,test_win_on,test_win_off,task_index,chance_index,time_point):
    
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]

    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    

    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_test(
        [gc_mean[0,:,test_win_on:test_win_off] , gc_mean[1,:,test_win_on:test_win_off]],
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    C1_stat['cond1']=np.mean(gc_mean[0,:,:],0)
    C1_stat['cond2']=np.mean(gc_mean[1,:,:],0)
    
    
    return C1_stat

def stat_cluster_1sample_roi(gc_mean1,gc_mean2,sub_list_a,test_win_on,test_win_off,task_index,chance_index,time_point):
    
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean1.shape[1]
    ROI_name=['IIT','IITPFC']

    
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    df1 = pd.DataFrame(gc_mean1, columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='ROI',value=ROI_name[0])
    

    
    df2 = pd.DataFrame(gc_mean2, columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='ROI',value=ROI_name[1])
    
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','ROI'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)

    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_test(
        [gc_mean1[:,test_win_on:test_win_off] , gc_mean2[:,test_win_on:test_win_off]],
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(gc_mean1[:,test_win_on:test_win_off] , gc_mean2[:,test_win_on:test_win_off], paired=True, alternative='two-sided', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(1/bf)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    C1_stat['BF']=bf_ROI_value
    
       
    
    return ts_df,C1_stat

def stat_cluster_1sample(gc_mean,sub_list_a,test_win_on,test_win_off,task_index,chance_index,time_point):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    #time_point = np.array(range(-500,1001, 10))/1000
    
    df1 = pd.DataFrame(gc_mean[0,:,:], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    df2 = pd.DataFrame(gc_mean[1,:,:], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='Task',value=task_index[1])
    
    T_obs_2, clusters_2, cluster_p_values_2, H0_2 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[1,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C2_stat=dict()
    C2_stat['T_obs']=T_obs_2
    C2_stat['cluster']=clusters_2
    C2_stat['cluster_p']=cluster_p_values_2
    
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,C1_stat,C2_stat

def stat_cluster_1sample_cmb(gc_mean,sub_list_a,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    time_point = np.array(range(-500,1501, 10))/1000
    
    df1 = pd.DataFrame(gc_mean[0,:,:], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    
    
    ts_df = pd.melt(df1, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,C1_stat

def stat_cluster_1sample_GAT(gc_mean,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='mask',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
   
    T_obs_2, clusters_2, cluster_p_values_2, H0_2 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[1,:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='indices',verbose=None)
    
    C2_stat=dict()
    C2_stat['T_obs']=T_obs_2
    C2_stat['cluster']=clusters_2
    C2_stat['cluster_p']=cluster_p_values_2

    
    return C1_stat,C2_stat

def gc_JK(gc_mean):
    ##leave one out  Jackknife
    # loo = LeaveOneOut()
    a,b,c=gc_mean.shape
    gc_JK_mean=np.zeros([a,b,c])
    
    sub_index = np.arange(0,b,1)  
    
    for n in range(a):
        temp=gc_mean[n,:,:]
        #loo.get_n_splits(temp)
        
        for subnn in range(b):
            ttemp=[]
            ttemp=temp
            ttemp=np.delete(temp,subnn,axis=0)
            gc_JK_mean[n,subnn,:]=ttemp.mean(axis=0)
            
    return gc_JK_mean
            
            

def wcd_plt(group_data,roi_name='GNW', task='replay',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):

    if task=='replay':
        cond_name=['Target','Non-Target']
    
    elif task=='vg':
        cond_name=['Seen','Unseen']
       


    time_point = np.array(range(-500,1001, 10))/1000
    
    #get decoding data
    ROI_wcd_g=dat2g(group_data,roi_name,cond_name,decoding_name='wcd_acc',sub_list=sub_list)
    
   
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_wcd_g,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index)
    
    
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster=df_plot_cluster(ts_df_cluster,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=task,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    ##Jackknife
     #stat
    ROI_wcd_g_JK=gc_JK(ROI_wcd_g)
    
    df1 = pd.DataFrame(ROI_wcd_g_JK[0,:,:], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=cond_name[0])
    df2 = pd.DataFrame(ROI_wcd_g_JK[1,:,:], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='Task',value=cond_name[1])
    df=df1.append(df2)
    
    ts_df1 = pd.melt(df1, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    
    fname_JK_1= op.join(stat_figure_root,roi_name +'_'+cond_name[0]+'JK_dat.png')
    
    fig_dims=(7.5,4)
    fig,ax=plt.subplots(figsize=fig_dims)
    fig = sns.lineplot(x="time(s)", y="decoding accuracy(%)", data=ts_df1,hue='SUBID')
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='upper right', title='SUBID',fontsize=8)
    plt.xlim([-0.5,1])
    plt.ylim([15,100])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80,100])
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    g=fig.get_figure()
    #g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,h col='Task',aspect=2,palette=colors,legend=False)
    #g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    g.savefig(fname_JK_1,format="png", transparent=True, dpi=300)
    
    
    
    ts_df2 = pd.melt(df2, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    
    fname_JK_2= op.join(stat_figure_root,roi_name +'_'+cond_name[1]+'JK_dat.png')
    
    fig_dims=(7.5,4)
    fig,ax=plt.subplots(figsize=fig_dims)
    fig = sns.lineplot(x="time(s)", y="decoding accuracy(%)", data=ts_df2,hue='SUBID')
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='upper right', title='SUBID',fontsize=8)
    plt.xlim([-0.5,1])
    plt.ylim([15,100])
    plt.xticks([-0.5,0,0.5,1])
    plt.yticks([20,40,60,80,100])
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    g=fig.get_figure()
    #g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,h col='Task',aspect=2,palette=colors,legend=False)
    #g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    g.savefig(fname_JK_2,format="png", transparent=True, dpi=300)
     
    JK_ts_df_cluster,JK_C1_stat,JK_C2_stat=stat_cluster_1sample(ROI_wcd_g_JK,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index)
    
    fname_cluster_fig_JK= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster_JK" + '.svg')
    
    #plot
    JKsig1_cluster,JKsig2_cluster=df_plot_cluster(JK_ts_df_cluster,JK_C1_stat,JK_C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=task,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig_JK)
    
    
    
    # #prepare data for plt plot   
    # ROI_wcd_g_dat=g2gdat(ROI_wcd_g,time_point,sig1_cluster,sig2_cluster)

    
    # csv_fname=op.join(Stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.csv')

    # df2csv(ROI_wcd_g_dat,cond_name,csv_fname)

    #Jackknife

def wcd_plt_diff(group_data,roi_name='GNW', diff_index='seen',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):

   
    cond_name=['Seen','Unseen']
       


    time_point = np.array(range(-500,1501, 10))/1000
    time_index=201
    
    #get decoding data
    ROI_wcd_g=dat2g(group_data,roi_name,cond_name,time_index,decoding_name='wcd_acc',sub_list=sub_list)
    
   
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_wcd_g,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index,
                                                       time_point=time_point)
    
    C3_stat=stat_cluster_1sample_seen(ROI_wcd_g,
                                      sub_list,
                                      test_win_on,
                                      test_win_off,
                                      task_index=cond_name,
                                      chance_index=chance_index,
                                      time_point=time_point)
    
    
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster_diff" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster,sig3_cluster=df_plot_cluster_diff(ts_df_cluster,C1_stat,C2_stat,C3_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=task,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    
    stat=dict()
    stat['seen']=C1_stat
    stat['unseen']=C2_stat
    stat['diff']=C3_stat
    
    fname_data_stat=op.join(Stat_data_root, 'stat_'+roi_name + '.pickle')
    fw = open(fname_data_stat,'wb')
    pickle.dump(stat,fw)
    fw.close()
    
    # #prepare data for plt plot   
    # ROI_wcd_g_dat=g2gdat(ROI_wcd_g,time_point,sig1_cluster,sig2_cluster)

    
    # csv_fname=op.join(Stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.csv')

    # df2csv(ROI_wcd_g_dat,cond_name,csv_fname)

    #Jackknife

def wcd_plt_diff_ET(group_data,roi_name='GNW', diff_index='seen',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):

   
    cond_name=['Seen','Unseen']
       


    time_point = np.array(range(-500,1001, 10))/1000
    time_index=151
    
    #get decoding data
    ROI_wcd_g=dat2g_ET(group_data,roi_name,cond_name,time_index,sub_list=sub_list)
    
   
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_wcd_g,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index,
                                                       time_point=time_point)
    
    C3_stat=stat_cluster_1sample_seen(ROI_wcd_g,
                                      sub_list,
                                      test_win_on,
                                      test_win_off,
                                      task_index=cond_name,
                                      chance_index=chance_index,
                                      time_point=time_point)
    
    
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster_diff" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster,sig3_cluster=df_plot_cluster_diff(ts_df_cluster,C1_stat,C2_stat,C3_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=task,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    
    stat=dict()
    stat['seen']=C1_stat
    stat['unseen']=C2_stat
    stat['diff']=C3_stat
    
    fname_data_stat=op.join(Stat_data_root, 'stat_'+roi_name + '.pickle')
    fw = open(fname_data_stat,'wb')
    pickle.dump(stat,fw)
    fw.close()
    


def ccd_plt(group_data,roi_name='GNW',decoding_name='ccd_acc_cat',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):


    time_point = np.array(range(-500,1501, 10))/1000
    time_index=201
    #task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    cond_name=['A2B','B2A']
    ROI_ccd_g=dat2g(group_data,roi_name,cond_name=cond_name,time_index=time_index,decoding_name=decoding_name,sub_list=sub_list)
    
    
    
    # #FDR methods
    
    # #stat
    # ts_df_fdr,T1,pval1,T2,pval2=gc2df(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    # #plot
    # fname_fdr_fig= op.join(stat_figure_root, roi_name + '_'+ str(test_win_on) + '_'+ str(test_win_off) +"_acc_CCD_fdr" + '.png')
    
    # sig1_fdr,sig2_fdr=df_plot(ts_df_fdr,T1,pval1,T2,pval2,time_point,test_win_on,
    #                           roi_name,task_index=task_index,
    #                           chance_index=chance_index,y_index=y_index,fname_fig=fname_fdr_fig)
    
    
    
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_ccd_g,sub_list,test_win_on,test_win_off,task_index=cond_name,chance_index=chance_index,time_point=time_point)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + decoding_name+'_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CCD_cluster" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster=df_plot_cluster(ts_df_cluster,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=decoding_name,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
def wcd_full_plt(group_data,roi_name='GNW',decoding_name='wcd_acc_cat',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):


    time_point = np.array(range(-500,1001, 10))/1000
    time_index=151
    #task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    if decoding_name=='wcd_acc_cat':
        cond_name=['Cat']
    elif decoding_name=='wcd_acc_loc':
        cond_name=['Loc']
    ROI_ccd_g=dat2g(group_data,roi_name,cond_name=cond_name,time_index=time_index,decoding_name=decoding_name,sub_list=sub_list)
    
    
    
    # #FDR methods
    
    # #stat
    # ts_df_fdr,T1,pval1,T2,pval2=gc2df(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    # #plot
    # fname_fdr_fig= op.join(stat_figure_root, roi_name + '_'+ str(test_win_on) + '_'+ str(test_win_off) +"_acc_CCD_fdr" + '.png')
    
    # sig1_fdr,sig2_fdr=df_plot(ts_df_fdr,T1,pval1,T2,pval2,time_point,test_win_on,
    #                           roi_name,task_index=task_index,
    #                           chance_index=chance_index,y_index=y_index,fname_fig=fname_fdr_fig)
    
    
    
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat=stat_cluster_1sample_cmb(ROI_ccd_g,sub_list,test_win_on,test_win_off,task_index=cond_name,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + decoding_name+'_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster=df_plot_cluster_cmb(ts_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=cond_name,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)

def ccd_plt_diff(group_data,cond_name=['A2B'],decoding_name='ccd_acc_cat',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):


    time_point = np.array(range(-500,1001, 10))/1000
    time_index=151
    #task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    
    roi_name=['IIT','IITPFC']
    
    
    ROI_ccd_g=dat2g_roi(group_data,roi_name=roi_name,cond_name=cond_name,time_index=time_index,decoding_name=decoding_name,sub_list=sub_list_a)
    
     
    # #FDR methods
    
    # #stat
    # ts_df_fdr,T1,pval1,T2,pval2=gc2df(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    # #plot
    # fname_fdr_fig= op.join(stat_figure_root, roi_name + '_'+ str(test_win_on) + '_'+ str(test_win_off) +"_acc_CCD_fdr" + '.png')
    
    # sig1_fdr,sig2_fdr=df_plot(ts_df_fdr,T1,pval1,T2,pval2,time_point,test_win_on,
    #                           roi_name,task_index=task_index,
    #                           chance_index=chance_index,y_index=y_index,fname_fig=fname_fdr_fig)
    
    
    
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat=stat_cluster_1sample_roi(ROI_ccd_g[0,:,:],ROI_ccd_g[1,:,:],sub_list,test_win_on,test_win_off,task_index=cond_name,chance_index=chance_index,time_point=time_point)
    
    fname_data_BF=op.join(Stat_data_root, 'BF_value_'+decoding_name+'_'+cond_name + '.pickle')
    fw = open(fname_data_BF,'wb')
    pickle.dump(C1_stat,fw)
    fw.close()
    
    
    
    fname_cluster_fig= op.join(stat_figure_root, cond_name + decoding_name+'_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CCD_ROI_diff_cluster" + '.svg')
    #print(fname_cluster_fig)
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts_df_cluster,C1_stat,time_point,
                                     test_win_on,test_win_off,
                                     chance_index,y_index,
                                     fname_fig=fname_cluster_fig) 
    
    
    
    # #prepare data for plt plot       
    # fname_data_cluster=op.join(Stat_data_root, 'cluster_value_'+decoding_name+'_'+cond_name + '.pickle')
    # fw = open(fname_data_cluster,'wb')
    # pickle.dump(sig1_cluster,fw)
    # fw.close()    
    
    
    
    # fr1=open(fname_data_BF,'rb')
    # BF_data=pickle.load(fr1)
    
    # fr2=open(fname_data_cluster,'rb')
    # cluster_data=pickle.load(fr2)

def wcd_seen_plt(group_data,roi_name='GNW', test_win_on=50, test_win_off=150,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):

    #task=='replay_combine':
    cond_name=['seen']
    
       


    time_point = np.array(range(-500,1501, 10))/1000
    time_index=201
    
    #get decoding data
    ROI_wcd_g=dat2g(group_data,roi_name,cond_name,time_index,decoding_name='wcd_acc',sub_list=sub_list)
    
   
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat=stat_cluster_1sample_cmb(ROI_wcd_g,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index)
    
    fname_data_stat=op.join(Stat_data_root, 'stat_value_'+cond_name + '.pickle')
    fw = open(fname_data_stat,'wb')
    pickle.dump(C1_stat,fw)
    fw.close()
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster=df_plot_cluster_seen(ts_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=cond_name,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)

def wcd_plt_cmb(group_data,roi_name='GNW', task='replay',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):

    #task=='replay_combine':
    cond_name=['AT_combine']
    
       


    time_point = np.array(range(-500,1501, 10))/1000
    time_index=201
    
    #get decoding data
    ROI_wcd_g=dat2g(group_data,roi_name,cond_name,time_index,decoding_name='wcd_acc',sub_list=sub_list)
    
   
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat=stat_cluster_1sample_cmb(ROI_wcd_g,
                                                       sub_list,
                                                       test_win_on,
                                                       test_win_off,
                                                       task_index=cond_name,
                                                       chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster=df_plot_cluster_cmb(ts_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task=task,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    # ##Jackknife
    # #  #stat
    # ROI_wcd_g_JK=gc_JK(ROI_wcd_g)
    
    # df1 = pd.DataFrame(ROI_wcd_g_JK[0,:,:], columns=time_point)
    # df1.insert(loc=0, column='SUBID', value=sub_list)
    # df1.insert(loc=0, column='Task',value=cond_name)
  
    # ts_df1 = pd.melt(df1, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    # #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    
    # fname_JK_1= op.join(stat_figure_root,roi_name +'_'+cond_name+'JK_dat.png')
    
    # fig_dims=(7.5,4)
    # fig,ax=plt.subplots(figsize=fig_dims)
    # fig = sns.lineplot(x="time(s)", y="decoding accuracy(%)", data=ts_df1,hue='SUBID')
    # #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.legend(loc='upper right', title='SUBID',fontsize=8)
    # plt.xlim([-0.5,1.5])
    # plt.ylim([40,100])
    # plt.xticks([-0.5,0,0.5,1,1.5])
    # plt.yticks([40,60,80,100])
    # plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    # plt.axvline(0, color='k', linestyle='-', label='onset')
    # g=fig.get_figure()
    # #g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,h col='Task',aspect=2,palette=colors,legend=False)
    # #g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    # g.savefig(fname_JK_1,format="png", transparent=True, dpi=300)
    
    
    
    
    # JK_ts_df_cluster,JK_C1_stat=stat_cluster_1sample_cmb(ROI_wcd_g_JK,
    #                                                     sub_list,
    #                                                     test_win_on,
    #                                                     test_win_off,
    #                                                     task_index=cond_name,
    #                                                     chance_index=chance_index)
    
    #fname_cluster_fig_JK= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster_JK" + '.svg')
    
    #plot
    # JKsig1_cluster=df_plot_cluster_cmb(JK_ts_df_cluster,JK_C1_stat,time_point,
    #                                           test_win_on,test_win_off,
    #                                           roi_name,task=task,
    #                                           chance_index=chance_index,y_index=y_index,
    #                                           fname_fig=fname_cluster_fig_JK)
    
    
    
    # #prepare data for plt plot   
    # ROI_wcd_g_dat=g2gdat(ROI_wcd_g,time_point,sig1_cluster,sig2_cluster)

    
    # csv_fname=op.join(Stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.csv')

    # df2csv(ROI_wcd_g_dat,cond_name,csv_fname)

    #Jackknife
    
def ctwcd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):


    time_point = np.array(range(-500,1001, 10))/1000
    task_index=['Seen','Unseen']
    #get decoding data
    ROI_gat_g=dat2gat(group_data,roi_name,cond_name=['Seen','Unseen'],decoding_name='ctwcd_acc',sub_list=sub_list)
    
    
    
    #cluster based methods
    
    #stat
    C1_stat,C2_stat=stat_cluster_1sample_GAT(ROI_gat_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CTWCD_cluster")
    
    #plot
    df_plot_cluster_GAT(ROI_gat_g,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
def ctccd_plt(group_data,roi_name='GNW',decoding_name='ctccd_acc_cat',test_win_on=50, test_win_off=200,chance_index=50,y_index=15,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):


    time_point = np.array(range(-500,1001, 10))/1000
    task_index=['A2B','B2A']
    #get decoding data
    ROI_gat_g=dat2gat(group_data,roi_name,cond_name=['A2B','B2A'],decoding_name=decoding_name,sub_list=sub_list)
    
    
    
    #cluster based methods
    
    #stat
    C1_stat,C2_stat=stat_cluster_1sample_GAT(ROI_gat_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name +'_'+ decoding_name+'_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CTCCD_cluster")
    
    #plot
    df_plot_cluster_GAT(ROI_gat_g,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
#def subROI_plt()

def subROI_wcd_plt(group_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40,sub_list=sub_list_a,stat_figure_root=Stat_figure_root):
    # 2 Get Sub ROI
    if analysis_name=='dAT_subPFC_loc_full':
        ROI_name=[]
        for i in range(13):
            ROI_name.append(f'PFC13_ROI_{i+1}')
    elif analysis_name=='dAT_subP2F_loc_full':
        ROI_name=[]
        for i in range(10):
            ROI_name.append(f'P2F_ROI_{i+1}')

    num_ROI=len(ROI_name)
        
        
    task_index=['Seen','Unseen']
    time_point = np.array(range(-500,1001, 10))/1000
    cond_name=['Seen','Unseen']
    #get decoding data
    subROI_data=np.zeros([num_ROI,2,len(sub_list),151])
    for i in range(num_ROI):
        print('i=',i)
        #get decoding data
        subROI_data_temp=dat2g(group_data,ROI_name[i],cond_name=['Seen','Unseen'],time_index=151,decoding_name='wcd_acc',sub_list=sub_list)   
        #cluster based methods
        
        #stat
        ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(subROI_data_temp,
                                                           sub_list,
                                                           test_win_on,
                                                           test_win_off,
                                                           task_index=cond_name,
                                                           chance_index=chance_index,
                                                           time_point=time_point)
        
        C3_stat=stat_cluster_1sample_seen(subROI_data_temp,
                                          sub_list,
                                          test_win_on,
                                          test_win_off,
                                          task_index=cond_name,
                                          chance_index=chance_index,
                                          time_point=time_point)
    
        fname_cluster_fig= op.join(stat_figure_root, ROI_name[i] + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster_diff" + '.svg')
    
        #plot
        sig1_cluster,sig2_cluster,sig3_cluster=df_plot_cluster_diff(ts_df_cluster,C1_stat,C2_stat,C3_stat,time_point,
                                                  test_win_on,test_win_off,
                                                  ROI_name[i],task=task,
                                                  chance_index=chance_index,y_index=y_index,
                                                  fname_fig=fname_cluster_fig)
        

def run_ROI_decoding_group(make_plot = True,sub_list=sub_list_a,visit_id='V2',task='replay',analysis_name='AT'):

    # stdout_obj = sys.stdout                 # store original stdout 
    # sys.stdout = open(op.join(out_path,     # open log file
    #                            os.path.basename(__file__) + "_%s.txt" % (site_id+subject_id)),'w')
    
    # print("Processing subject: %s" % subject_id)
    

    
    
    # 1 Set path to the data root path
    # if task=='replay':
    #     analysis_name='AT'
        
    # elif task=='vg':
    #     analysis_name='dAT'
        
    decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa_e2')

    data_path=op.join(decoding_path,analysis_name)    
        
    
    
    
    
    
    ### III  Set the Output Data Path
    # Set path to decoding derivatives
    Stat_deriv_root = op.join(data_path,'group')
    if not op.exists(Stat_deriv_root):
        os.makedirs(Stat_deriv_root)
        
    
    # 1) output_data
    Stat_data_root= op.join(Stat_deriv_root,"data")
    if not op.exists(Stat_data_root):
        os.makedirs(Stat_data_root)

    # 2) output_figure
    Stat_figure_root = op.join(Stat_deriv_root,"figures")
    if not op.exists(Stat_figure_root):
        os.makedirs(Stat_figure_root)
    
    # Read data
    fname_data=op.join(Stat_deriv_root, "data_group_" + analysis_name +
                   '.pickle')
    

    fr=open(fname_data,'rb')
    group_data=pickle.load(fr)
    
    
    
    
    
    if analysis_name=='AT'or analysis_name=='dAT' or analysis_name=='AT_loc'or analysis_name=='dAT_loc' :
    
        #GNW
        wcd_plt(group_data,roi_name='GNW',task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        #IIT
        # 0ms to 1500ms
        wcd_plt(group_data,roi_name='IIT',task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        
    elif analysis_name=='dAT_full'or analysis_name=='dAT_loc_full' :  
    
        #GNW
        wcd_plt_diff(group_data,roi_name='GNW',diff_index='seen',test_win_on=50, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='ET_WCD':
        
        ch_index=['Cat_4ch','Cat_6ch','Loc_4ch','Loc_6ch']
        for i in range(4):
            wcd_plt_diff_ET(group_data,roi_name=ch_index[i],diff_index='seen',test_win_on=50,test_win_off=101,chance_index=50,y_index=40,
                         sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
    
    elif analysis_name=='seen_full':
        #GNW
        wcd_seen_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=201,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='CCD_PFC_full':  
    
        #GNW
        #cat
        ccd_plt(group_data,roi_name='GNW',decoding_name='ccd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        #GNW
        #loc
        ccd_plt(group_data,roi_name='GNW',decoding_name='ccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='GAT_PFC_dAT_full' or analysis_name=='GAT_PFC_dAT_loc_full':  
    
        #GNW
        #cat
        ctwcd_plt(group_data,roi_name='GNW',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    # elif analysis_name=='GAT_PFC_dAT_loc_full':      
    #     #GNW
    #     #loc
    #     ccd_plt(group_data,roi_name='GNW',decoding_name='ccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
    #             sub_list=sub_list,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='CTCCD_PFC_full' :  
    
        #GNW
        #cat
        ctccd_plt(group_data,roi_name='GNW',decoding_name='ctccd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        #loc
        ctccd_plt(group_data,roi_name='GNW',decoding_name='ctccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='dAT_subPFC_loc_full' or analysis_name=='dAT_subP2F_loc_full':
        subROI_wcd_plt(group_data,decoding_method ='wcd', test_win_on=25, test_win_off=151,chance_index=50,y_index=30,
                      sub_list=sub_list_a,stat_figure_root=Stat_figure_root) 
        
        
    elif analysis_name=='WCD_full':  
    
        #GNW
        #cat
        wcd_full_plt(group_data,roi_name='GNW',decoding_name='wcd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        #GNW
        #loc
        wcd_full_plt(group_data,roi_name='GNW',decoding_name='wcd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        # #IIT
        # #cat
        # wcd_full_plt(group_data,roi_name='IIT',decoding_name='wcd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
        #         sub_list=sub_list,stat_figure_root=Stat_figure_root)
        
        # #IIT
        # #loc
        # wcd_full_plt(group_data,roi_name='IIT',decoding_name='wcd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
        #         sub_list=sub_list,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='CCD_IIT_full':  
    
        #GNW
        #cat
        ccd_plt(group_data,roi_name='IIT',decoding_name='ccd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list,stat_figure_root=Stat_figure_root)
        
        #GNW
        #loc
        ccd_plt(group_data,roi_name='IIT',decoding_name='ccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='CCD_IITPFC_full':  
        ccd_plt_diff(group_data,cond_name='A2B',decoding_name='ccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        ccd_plt_diff(group_data,cond_name='B2A',decoding_name='ccd_acc_loc',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,sub_list=sub_list_a,stat_figure_root=Stat_figure_root)

       
        
        # ccd_plt_diff(group_data,cond_name='A2B',decoding_name='ccd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        # ccd_plt_diff(group_data,cond_name='B2A',decoding_name='ccd_acc_cat',test_win_on=25, test_win_off=151,chance_index=50,y_index=40,sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
    
    elif analysis_name=='AT_PFC13_control' or analysis_name=='dAT_PFC13':
        wcd_plt(group_data,roi_name='PFC13',task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='AT_subF_control':
        ROI_Name = ['F1','F2','F3','F4','F5','F6','F7',
                    'F8','F9','F10','F11','F12','F13']
        for i in range(13):
            wcd_plt(group_data,roi_name=ROI_Name[i],task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
    
    elif analysis_name=='AT_PFC6_control':
        wcd_plt(group_data,roi_name='PFC6',task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
    elif analysis_name=='AT_cmb_stf_full' or analysis_name=='AT_cmb_stf_full_loc':
        #GNW
        wcd_plt_cmb(group_data,roi_name='GNW',task=task,test_win_on=25, test_win_off=101,chance_index=50,y_index=40,
                sub_list=sub_list_a,stat_figure_root=Stat_figure_root)
        
        # #IIT
        # # 0ms to 1500ms
        # wcd_plt_cmb(group_data,roi_name='IIT',task=task,test_win_on=0, test_win_off=151,chance_index=50,y_index=40,
        #         sub_list=sub_list,stat_figure_root=Stat_figure_root)
    
    

   
    
    
   


    
  
    
    




# =============================================================================
# RUN
# =============================================================================
# #subject_list=['SA111','SA148','SB040','SB069','SB071']
# visit_id='V2'
# task='replay'
# analysis_name='AT_cmb_stf_full'
# #analysis_name='CCD_IIT_full'
# #cond_name=['Target','Non-Target']
# # task='vg'
# # cond_name=['Seen','Unseen']
# #con_C='ATTF'
# #conditions_C = ['faceT','faceNT']
# #out_path='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/Exp2/test'

# EEG = True



# if subject_id in no_eeg_sbj:
#     EEG = False
# else:
#     EEG = True

if __name__ == '__main__':
    run_ROI_decoding_group(sub_list=sub_list_a,visit_id=visit_id,task=task,analysis_name=analysis_name)
    
    
