'''
Control analysis for MEG source leakage: compare the evoked response of PFC vs
posterior (POS) ROIs. If posterior activity leaked into PFC, the seen-vs-unseen
difference should appear at the SAME time and magnitude in both ROIs; a genuine
gradient (posterior earlier / stronger) argues against leakage.

Unlike pipeline_erf_source_group.py, this loads TWO ROI group results (PFC and
POS) and runs two tests on the seen-unseen ERF difference:
  1. Amplitude: paired t-test (two-tailed, Bonferroni) on the per-subject ERF
     difference averaged over the statistic window (250-500 ms), PFC vs POS.
  2. Onset latency: onset = first crossing of the half-maximum of the averaged
     ERF difference (30 Hz low-pass + Gaussian smoothing sigma=50 ms); the
     PFC-POS onset difference is tested with a leave-one-out jackknife t-test
     (Miller et al. 1998), one-tailed (PFC later than POS).

Runs over the dAT probe configs (face and object) x {PFC, POS} group configs
under pipeline_erf_source_group_250_500 (see the loop below).

Onset detection (current settings): onset = the FIRST crossing of the
half-maximum ((max+min)/2, signed) of the averaged ERF difference, computed
after a 30 Hz low-pass + Gaussian smoothing (sigma=50 ms); a single shared
half-max threshold = mean(PFC, POS) is used for both ROIs.
'''


#%% import
import os
import sys

from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# for show time

# for t test
from scipy.stats import t
from pathlib import Path 

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy.ndimage import gaussian_filter1d # to smooth erf for onset detection

# for save data
import pickle
from _help_functions import (Project_Dir, apply_lowpass_filter, compute_mean_exp2, group_get_sublist,
                             colordict,
                             get_eps_use_info,
                                 read_configfile,
                                 )
from _help_functions import (general_param,
                             path_allana,
                                 )
from _help_module import (_slug)
import numpy as np
import pandas as pd

import mne_bids
# for statistic testing
import scipy



#%% get the erf (avg significant labels of each roi) for each group analysis
def calculate_half_abs_max(data_t, times, time_window):
    """
    Calculate half of the absolute maximum value within a specified time window.

    Parameters:
    data_t (array-like): The data values corresponding to the time series.
    times (array-like): The time points corresponding to the data values.
    time_window (tuple): A tuple specifying the start and end of the time window (start, end).

    Returns:
    float: Half of the absolute maximum value within the specified time window.
    """
    # Find indices of times within the specified time window
    indices_in_window = [
        idx for idx, t in enumerate(times)
        if time_window[0] <= t <= time_window[1]
    ]

    # Calculate half of the absolute maximum value in the window
    half_abs_max = np.max(np.abs(data_t[indices_in_window])) / 2

    return half_abs_max

#  this find the FIRST time the signal crosses the threshold from below
def find_first_crossing_onset(data_curve, times, threshold, search_window):
    """
    Find the FIRST time point where the signal crosses threshold from below.
    
    Parameters:
    -----------
    data_curve : array-like
        The signal values
    times : array-like
        Time points
    threshold : float
        The threshold value to detect crossing
    search_window : tuple
        (start_time, end_time) to search for onset
    
    Returns:
    --------
    onset_time : float
        Time of first upward threshold crossing
    onset_value : float
        Signal value at crossing
    onset_idx : int
        Index of crossing point
    """
    # Restrict to search window
    search_idx = (times >= search_window[0]) & (times <= search_window[1])
    search_signal = data_curve[search_idx]
    search_times = times[search_idx]
    
    # Find where signal crosses threshold from below
    # Look for points where: signal[i-1] < threshold AND signal[i] >= threshold
    below_threshold = search_signal < threshold
    above_threshold = search_signal >= threshold
    
    # Find transitions from below to above
    crossings = np.where(below_threshold[:-1] & above_threshold[1:])[0]
    
    if len(crossings) == 0:
        # No crossing found - signal never reaches threshold
        return np.nan, np.nan, np.nan
    
    # Take the FIRST crossing
    first_crossing_idx = crossings[0] + 1  # +1 because we compared [:-1] with [1:]
    
    # Get the actual index in original arrays
    original_idx = np.where(search_idx)[0][first_crossing_idx]
    
    onset_time = times[original_idx]
    onset_value = data_curve[original_idx]
    
    return onset_time, onset_value, original_idx

#  this find the point CLOSEST to the halfmax value
def find_timepoint_of_uppeak_for_yvalue(data_curve, times, threshold,search_window=None):
    """
    Find the time point corresponding to a specified y-value on a curve, 
    considering only the first part up to the peak.

    Parameters:
    data_curve (array-like): The y-values of the curve.
    times (array-like): The time points corresponding to the data values.
    threshold (float): The target y-value to find on the curve.

    Returns:
    float: The time point where the curve matches the target y-value.
    """
    # Find the peak index
    peak_index = np.argmax(data_curve)  
    # Consider curve up to the peak
    curve_until_peak = data_curve[:peak_index + 1]  
    # Compute differences
    differences = np.abs(curve_until_peak - threshold)  
    # Closest match index
    closest_index = np.argmin(differences)  
    
    # Find the closest y-value and corresponding time point
    y_value_of_curve = data_curve[closest_index]
    y_value_time = times[closest_index]

    # Return the corresponding time point and the actual y-value
    return   y_value_time,y_value_of_curve,closest_index
    
def calculate_half_max(data_t, times, time_window):
    """
    Calculate the half-maximum value within a specified time window.

    Parameters:
    data_t (array-like): The data values corresponding to the time series.
    times (array-like): The time points corresponding to the data values.
    time_window (tuple): A tuple specifying the start and end of the time window (start, end).

    Returns:
    float: The half-maximum value within the specified time window.
    """
    # Find indices of times within the specified time window
    indices_in_window = [
        idx for idx, t in enumerate(times)
        if time_window[0] <= t <= time_window[1]
    ]

    # Calculate the half-maximum value in the window
    data_in_window = data_t[indices_in_window]
    min_val = np.min(data_in_window)
    max_val = np.max(data_in_window)
    half_max = min_val + ((max_val - min_val) / 2)

    return half_max
def calculate_sd(subsampling_results):
    """
    Calculate the standard error (S_D) using the jackknife subsampling results.

    Parameters:
    subsampling_results (dict): Dictionary containing the latency differences for each participant iteration of subsampled.

    Returns:
    float: Standard error (S_D).
    """
    n = len(subsampling_results)
    mean_diff = np.mean(list(subsampling_results.values()))
    sd = np.sqrt(
        (n - 1) / n * \
        np.sum([(x - mean_diff) ** 2 for x in subsampling_results.values()])
    )
    return sd
def standard_t_test_table(latency_diff, standard_error, nsub, alpha=0.05):
    """
    Generate a standard t-test table for the observed latency difference.

    Parameters:
    latency_diff (float): The observed latency difference.
    standard_error (float): The standard error (S_D) calculated from subsampling.
    nsub (int): Number of participants (degrees of freedom = nsub - 1).
    alpha (float): Significance level for the test (default is 0.05).

    Returns:
    dict: A dictionary containing the t-value, degrees of freedom, critical t-value, and significance result.
    """
    # Calculate the t-value
    t_value = latency_diff / standard_error

    # Degrees of freedom for a one-sample t-test
    degrees_of_freedom = nsub - 1

    # Critical t-value for two-tailed test
    critical_t = t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Determine significance
    is_significant = abs(t_value) > critical_t

    # Construct the result table
    results = {
        "Latency Difference": latency_diff,
        "Standard Error": standard_error,
        "t-Value": t_value,
        "Degrees of Freedom": degrees_of_freedom,
        "Critical t-Value": critical_t,
        "Significant": "Yes" if is_significant else "No"
    }

    return results

def one_tailed_t_test_table(latency_diff, standard_error, nsub, alpha=0.05):
    """
    Generate a one-tailed t-test table for the observed latency difference.

    Parameters:
    latency_diff (float): The observed latency difference.
    standard_error (float): The standard error (S_D) calculated from subsampling.
    nsub (int): Number of participants (degrees of freedom = nsub - 1).
    alpha (float): Significance level for the test (default is 0.05).

    Returns:
    dict: A dictionary containing the t-value, degrees of freedom, critical t-value, and significance result.
    """
    # Calculate the t-value
    t_value = latency_diff / standard_error

    # Degrees of freedom for a one-sample t-test
    degrees_of_freedom = nsub - 1

    # Critical t-value for one-tailed test
    critical_t = t.ppf(1 - alpha, degrees_of_freedom)

    # Determine significance
    is_significant = t_value > critical_t

    # Construct the result table
    results = {
        "Latency Difference": latency_diff,
        "Standard Error": standard_error,
        "Degrees of Freedom": degrees_of_freedom,
        "t-Value": t_value,
        "Critical t-Value": critical_t,
        "Significant": "Yes" if is_significant else "No"
    }

    return results
def plot_curves(times, curve1, curve2):
    """
    Plot two curves on the same graph with time on the x-axis.

    Parameters:
    times (array-like): The time points corresponding to the data values.
    curve1 (array-like): The y-values for the first curve.
    curve2 (array-like): The y-values for the second curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(times, curve1, label="Curve 1", color="blue")
    plt.plot(times, curve2, label="Curve 2", color="red")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Comparison of Two Curves")
    plt.legend()



def plot_curves_with_lines(times, 
                            curves,
                            curves_ci_lower=None,  
                            curves_ci_upper=None,  
                            curves_color=None ,
                            curves_legend=None ,
                            vertical_lines=None, 
                            vertical_lines_color=None,
                            title=None,
                            jpg_dir= None,
                            pdf_dir= None,
                            shadow_timewindow=None,
                            x_interval = None
                        ):
    """
    Plot two curves on the same graph with time on the x-axis and optional vertical and horizontal dashed lines.

    Parameters:
    times (array-like): The time points corresponding to the data values.
    curve1 (array-like): The y-values for the first curve.
    curve2 (array-like): The y-values for the second curve.
    curves_ci : list of array-like or None
        Confidence interval half-widths for each curve. If provided, shading shows mean ± CI.

    vertical_lines (list of float): Optional list of x-values where vertical dashed lines should be added.
    vertical_lines_color (list of str): Optional list of colors for the vertical dashed lines.
    """
    
    plt.figure(figsize=(15, 6))
    default_curves_color =["blue","red"]
    default_curves_legend =["Curve 1","Curve 2"]
    

    for curve_idx,curve in enumerate(curves):
        color = curves_color[curve_idx] if curves_color else default_curves_color[curve_idx] 
        legend = curves_legend[curve_idx] if curves_legend else default_curves_legend[curve_idx] 
        # Plot the mean line
        plt.plot(times, curve, label=legend,  color=color,linewidth=1.8, )
        # Add shaded CI if provided
        if curves_ci_lower is not None and curves_ci_upper is not None:
            plt.fill_between(times, 
                           curves_ci_lower[curve_idx], 
                           curves_ci_upper[curve_idx], 
                           color=color, alpha=0.2)
    if vertical_lines:
        for idx_x, x in enumerate(vertical_lines):
            # Determine the color and legend for the line
            color = vertical_lines_color[idx_x] if vertical_lines_color else "black"
            legend = curves_legend[idx_x] if curves_legend else default_curves_legend[idx_x] 

            # Find the y-values of corresponding curves at the x-location
            ys =  curves[idx_x]
            y = ys[x]



            # Plot the vertical dashed line
            plt.axvline(x=times[x], color=color, linestyle="--", linewidth=1.8, 
                        label=f"{legend} time= {times[x]:.3f}")

        # Plot the horizontal dashed line
        # avg_halfmax,horizontal threshold line (black dashed)
        plt.axhline(y=y, color='black', linestyle="--", linewidth=1.8, alpha=1,
                        label=f"{legend} y= {y:.3f}")
            
    # Add vertical line for onset and horizontal lines for zero
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # stimulus onset
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # zero line
    
    if shadow_timewindow:
        # Add gray shaded region for analysis window
        plt.axvspan(shadow_timewindow[0], shadow_timewindow[1], alpha=0.2, color='gray')  # adjust times as needed


    plot_title = title if title else "Comparison of Two Onsettitle"
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.xlim(times[0], times[-1])
    if x_interval:
        plt.gca().xaxis.set_major_locator(MultipleLocator(x_interval))

    plt.title(plot_title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    
    plt.savefig(Path(jpg_dir) / f"{_slug(title.replace(' ', '_'))}.jpg", bbox_inches="tight", dpi=300)
    plt.savefig(Path(pdf_dir) / f"{_slug(title.replace(' ', '_'))}.pdf", bbox_inches="tight", dpi=300)
    plt.show()

def main():
    # Dictionary to store half-absolute max results
    half_max_erfdiff = {}
    plotdata_erfdiff = {}
    plotdata_erfdiff_ci_upper = {}
    plotdata_erfdiff_ci_lower = {}

    # Calculate half-absolute maximum for each region of interest (ROI)
    for roi in ['PFC', 'POS']:

        
        if flag_plot_cut:
            
            erf_diff_avgedsiglabel_subt= erfdiff_subt_dict[epoch_data_name][roi][:, plot_timewindow_idx]
            times_plot = times[plot_timewindow_idx]
        else:
            erf_diff_avgedsiglabel_subt= erfdiff_subt_dict[epoch_data_name][roi]
            times_plot = times

        
        print(erf_diff_avgedsiglabel_subt.shape)

        # Average across participants
        # Compute mean and CI from raw subject data
        erf_diff_mean = np.mean(erf_diff_avgedsiglabel_subt, axis=0)
        confidence_level = 0.95
        n = erf_diff_avgedsiglabel_subt.shape[0]
        erf_diff_ci = scipy.stats.sem(erf_diff_avgedsiglabel_subt, axis=0) * \
                    scipy.stats.t.ppf((1 + confidence_level) / 2., n - 1)
        
        # Step 2: Apply filtering to mean and CI bounds
        if flag_filter:
            erf_diff_mean = apply_lowpass_filter(erf_diff_mean,fs=sfreq)

            # Filter the bounds, not the CI width directly
            upper_bound = apply_lowpass_filter(
                np.mean(erf_diff_avgedsiglabel_subt, axis=0) + erf_diff_ci, fs=sfreq)
            lower_bound = apply_lowpass_filter(
                np.mean(erf_diff_avgedsiglabel_subt, axis=0) - erf_diff_ci, fs=sfreq)


        if flag_smooth:
            # Check: Does smoothing shift peak latency?
            erf_diff_beforesmooth = erf_diff_mean  # After 30Hz lowpass
            erf_diff_smooth = gaussian_filter1d(erf_diff_beforesmooth, sigma=50)

            upper_bound = gaussian_filter1d(upper_bound, sigma=50)
            lower_bound = gaussian_filter1d(lower_bound, sigma=50)


            # Find peak time in both
            peak_raw = times_plot[np.argmax(erf_diff_beforesmooth)]
            peak_smooth = times_plot[np.argmax(erf_diff_smooth)]
            print(f"Peak shift: {peak_smooth - peak_raw:.3f}s")

            erf_diff_mean = erf_diff_smooth

        
        plotdata_erfdiff[roi] = erf_diff_mean
        plotdata_erfdiff_ci_upper[roi] = upper_bound
        plotdata_erfdiff_ci_lower[roi] = lower_bound

        if flag_use_abs:
            # Calculate half of the absolute max
            half_max_erfdiff[roi] = calculate_half_abs_max(
                data_t=erf_diff_mean,
                times=times_plot,
                time_window=time_window_halfabsmax
            )
            
        else:
            # Calculate half of the max
            half_max_erfdiff[roi] = calculate_half_max(
                data_t=erf_diff_mean,
                times=times_plot,
                time_window=time_window_halfabsmax
            )
    avg_halfmax = np.mean([half_max_erfdiff['PFC'],
                        half_max_erfdiff['POS']])

            
    
    # setup half max
    half_max_erfdiff_PFC = avg_halfmax if flag_avg_halfmax else half_max_erfdiff['PFC']
    half_max_erfdiff_POS = avg_halfmax if flag_avg_halfmax else half_max_erfdiff['POS']
    
    # get the onset of pfc and pos
    if flag_find_firstcrossing:
        onset_function = find_first_crossing_onset
        method_name = "first_crossing"
    elif flag_find_closestpoint:
        onset_function = find_timepoint_of_uppeak_for_yvalue
        method_name = "closest_point"
    else:
        raise ValueError("Either flag_find_firstcrossing or flag_find_closestpoint must be set to True.")


    onset_PFC, y_PFC,xidx_PFC = onset_function(
                                        data_curve = plotdata_erfdiff['PFC'], 
                                        times = times_plot,
                                        threshold = half_max_erfdiff_PFC,
                                        search_window = time_window_halfabsmax)
    onset_POS,y_POS,xidx_POS  = onset_function(
                                        data_curve = plotdata_erfdiff['POS'], 
                                        times = times_plot,
                                        threshold = half_max_erfdiff_POS,
                                        search_window = time_window_halfabsmax)
    # Latency difference between POS and PFC
    latency_diff = onset_PFC - onset_POS
    
    # Jackknife subsampling
    subsampling_latency_diff = {}
    for participant_idx in range(nsub):
        subsampled_half_max = {}
        subsampled_erf_diff={}
        for roi in ['PFC', 'POS']:
            if flag_plot_cut:
                erf_diff_avgedsiglabel_subt= erfdiff_subt_dict[epoch_data_name][roi][:, plot_timewindow_idx]
            else:
                erf_diff_avgedsiglabel_subt= erfdiff_subt_dict[epoch_data_name][roi]

            # Exclude the current participant and average across the rest
            subsampled_erf_diff_mean = np.mean(
                erf_diff_avgedsiglabel_subt[
                    [idx for idx in range(nsub) if idx != participant_idx], :
                ],
                axis=0
            )
            
            # filter the erf
            if flag_filter:
                subsampled_erf_diff_mean = apply_lowpass_filter(subsampled_erf_diff_mean,fs=sfreq)
            subsampled_erf_diff[roi] = subsampled_erf_diff_mean
            
            if flag_use_abs:
                # Calculate half of the absolute max for the subsampled data
                subsampled_half_max[roi] = calculate_half_abs_max(
                    data_t=subsampled_erf_diff_mean,
                    times=times_plot,
                    time_window=time_window_halfabsmax
                )
            else:
                # Calculate half of the max for the subsampled data
                subsampled_half_max[roi] = calculate_half_max(
                    data_t=subsampled_erf_diff_mean,
                    times=times_plot,
                    time_window=time_window_halfabsmax
                )

        # setup half max
        subsampled_avg_halfmax = np.mean([subsampled_half_max['PFC'],
                                            subsampled_half_max['POS']])
        
        subsampled_halfmax_erfdiff_PFC = subsampled_avg_halfmax if flag_avg_halfmax else subsampled_half_max['PFC']
        subsampled_halfmax_erfdiff_POS = subsampled_avg_halfmax if flag_avg_halfmax else subsampled_half_max['POS']
        
        # get the onset of pfc and pos
        subsampled_onset_PFC , subsampled_y_PFC ,_= \
            onset_function(data_curve = subsampled_erf_diff['PFC'], 
                                            times = times_plot,
                                            threshold = subsampled_halfmax_erfdiff_PFC,
                                            search_window = time_window_halfabsmax)
        
        subsampled_onset_POS, subsampled_y_POS,_ = \
            onset_function(data_curve = subsampled_erf_diff['POS'], 
                                        times = times_plot,
                                        threshold = subsampled_halfmax_erfdiff_POS,
                                        search_window = time_window_halfabsmax)
        # Calculate latency difference for the subsampled data
        subsampling_latency_diff[participant_idx] = subsampled_onset_PFC - subsampled_onset_POS
    # Calculate standard error (S_D)
    standard_error = calculate_sd(subsampling_latency_diff)


    # Generate the t-test table
    t_test_results = one_tailed_t_test_table(latency_diff, standard_error, nsub, alpha)
    # Print the results in a readable format
    print("\nStandard T-Test Table:")
    for key, value in t_test_results.items():
        print(f"{key}: {value}")

    # plot ERF difference with dash line of halfmaximum
    setting_name = (
        f"{epoch_data_info} {nsub}subs {method_name} "
        f"{'avged' if flag_avg_halfmax else ''} "
        f"{'halfabsmax' if flag_use_abs else 'halfmax'} "
        f"{'smoothed ' if flag_smooth else ''} "
        f"{'Filtered curve' if flag_filter else 'Nofilter curve'} "
        f"{time_window_halfabsmax} "
        f"t{float( t_test_results['t-Value'] ):.3f} "
        f"cri_t{float(t_test_results['Critical t-Value'] ):.3f} "
        f"{t_test_results['Significant']}"
    ) 
    
    roi1 =  'POS'  
    roi2 =  'PFC'  
    title =f'compre onset {setting_name}'

    plot_curves_with_lines(times_plot, 
                            curves = [plotdata_erfdiff[roi1],
                    plotdata_erfdiff[roi2]],
                    curves_ci_lower=[plotdata_erfdiff_ci_lower[roi1],
                                    plotdata_erfdiff_ci_lower[roi2]],
                            curves_ci_upper=[plotdata_erfdiff_ci_upper[roi1],
                                    plotdata_erfdiff_ci_upper[roi2]],

                            curves_legend = [roi1,
                                            roi2],
                            curves_color=[colordict[roi1],
                                        colordict[roi2]],
                            vertical_lines=[eval(f'xidx_{roi1}'),
                                            eval(f'xidx_{roi2}')],
                            vertical_lines_color=[colordict[roi1],
                                                colordict[roi2]],
                            title =title,
                            jpg_dir = jpg_smooth_dir if  flag_smooth else jpg_dir,
                            pdf_dir = pdf_smooth_dir if  flag_smooth else pdf_dir,
                            shadow_timewindow= time_staticwindow,
                            x_interval = x_interval
                            )

    return (
                onset_PFC, y_PFC, xidx_PFC,
                onset_POS, y_POS, xidx_POS,
                avg_halfmax,
                latency_diff, standard_error,
                title,
                plotdata_erfdiff,
                plotdata_erfdiff_ci_upper,
                plotdata_erfdiff_ci_lower
                
            )

# %%
flag_test= 0
script_name = os.path.basename(__file__).replace('.py','')
group_configfolder = 'pipeline_erf_source_group_250_500'
individual_ana = 'pipeline_erf_source'
individual_configfolder = 'pipeline_erf_source'

erf_dict = {}
erfdiff_subt_dict = {}
ttest_dict = {}
onset_dict = {}
for config_individual in ['dAT_probe_face_ft_bc_deci1_nocut',
                        'dAT_probe_obje_ft_bc_deci1_nocut']: 
    
    
    for config_group in ['T_mean_V_rms_subsamp_250_500_PFC',
                        'T_mean_V_rms_subsamp_250_500_POS']:

        individual_configfile = os.path.join(Project_Dir,'config_files',
                                            individual_configfolder,
                                            f"{config_individual}.json")
        group_configfile = os.path.join(Project_Dir,'config_files',
                                        group_configfolder,
                                        f"{config_group}.json")
        # Read the configfile file:
        param_sub = read_configfile(individual_configfile)
        param_group = read_configfile(group_configfile)

        # unpack the param_sub
        epoch_setting = general_param['epoch_setting_dict'][param_sub['epoch_setting_name']]
        epoch_data_name = param_sub['epoch_data_name']
        epoch_data = general_param['epoch_data_dict'][epoch_data_name]
        roi_params = general_param['roi_params_dict']
        cond1_content_name = param_sub['cond1_content_name']
        cond2_content_name = param_sub['cond2_content_name']

        info =get_eps_use_info(**epoch_setting)
        
        # unpack the param_group
        roi = param_group['roi']
        epoch_avg_type = param_group['epoch_avg_type']
        epoch_subsampling_type = param_group['epoch_subsampling_type']
        cond1_timewindow = param_group['cond1_timewindow']
        cond2_timewindow = param_group['cond2_timewindow']

        cbpt_setting = general_param['cbpt_params_dict'][param_group['cbpt_params_name']]
        p_cluster_forming= cbpt_setting['p_cluster_forming']
        # setting the individual result folder
        deriv_root = os.path.join(path_allana, 
                                    individual_ana,
                                    info.epoch_info)
        # setting the group result folder
        path_group = os.path.join(path_allana, individual_ana,
                        "analysis_group_results",
                        group_configfolder,)
        
        path_group_plot = os.path.join(path_allana, individual_ana,
                        "analysis_group_results_plot",
                        script_name,)
        pdf_path      = os.path.join(path_group_plot, 'pdf_nosmooth')
        jpg_path      = os.path.join(path_group_plot, 'jpg_nosmooth')
        jpg_smooth_path = os.path.join(path_group_plot, 'jpg_smooth')
        pdf_smooth_path = os.path.join(path_group_plot, 'pdf_smooth')
        plotdata_path = os.path.join(path_group_plot, 'plotdata')

        os.makedirs(pdf_path, exist_ok=True)
        os.makedirs(jpg_path, exist_ok=True)  
        os.makedirs(jpg_smooth_path, exist_ok=True)
        os.makedirs(pdf_smooth_path, exist_ok=True)
        os.makedirs(plotdata_path, exist_ok=True)  


        config_setting = f"{config_individual}_{config_group}"
        cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")



        clusterfile_name  = os.path.join(path_group, f"cbpt_cluster_{config_setting}_sig.pkl")
        blankfile_name    = os.path.join(path_group, f"cbpt_cluster_{config_setting}_no_sig.pkl")
        cbpt_result_name  = os.path.join(path_group, f"cbpt_allresult_{config_setting}.pkl")

        assert os.path.exists(cbpt_result_name) & (os.path.exists(blankfile_name) or os.path.exists(clusterfile_name)), \
            'error: cbpt result file not exist'

        with open(cbpt_result_name, 'rb') as pickle_file_load:
            cluster_stats = pickle.load(pickle_file_load)
        
        sub_list_name, sub_list = group_get_sublist(param_group,
                                            param_sub,
                                            group_configfile,
                                            individual_configfile,
                                            )
        if flag_test:
            sub_list = sub_list[0:10]
        #% -------------- Get example individual time data  --------------
        df_nepoch_group = pd.DataFrame()
        sub_dict = {}
        for subject in sub_list:
            subject

            bids_path = mne_bids.BIDSPath(
                    root=deriv_root, 
                    subject=  subject,
                    session= epoch_data['exp_id'],  
                    datatype='meg',  
                    task=epoch_data['task_id'],
                    suffix=f"{param_sub['epoch_data_name']}_{param_sub['compare_trltypes']}",
                    extension='.pkl',
                    check=False)
            dir_analyse = os.path.dirname( bids_path.fpath)
            pklfile_name = bids_path.fpath

            with open(pklfile_name, 'rb') as pickle_file_load:
                loaded_data = pickle.load(pickle_file_load)
                

            if 'times' in locals():
                assert all(times == loaded_data['times']),'error: times not match'
            else:
                times = loaded_data['times']
                
            if 'sfreq' in locals():
                assert sfreq == loaded_data['sfreq'],'error: times not match'
            else:
                sfreq = loaded_data['sfreq']
                
            if 'roi_params' in locals():
                assert roi_params.keys() == loaded_data['roi_params'].keys(), "error: roi_params not match"
            else:
                roi_params = loaded_data['roi_params']


            # get channel * time data
            if len(loaded_data ['t_dict_by_roi'][roi])==0:
                continue
            roi_data = loaded_data ['t_dict_by_roi'][roi]
            label_dict= loaded_data ['label_dict']
            sub_dict[subject] = roi_data

            assert roi_data[list(roi_data.keys())[0]]\
                    [epoch_avg_type][epoch_subsampling_type]\
                    [cond1_content_name].shape[0] == len(times),\
                    'error: times not match'
            
            # get the label names
            labels = [x for x in roi_data.keys() if 'all' not in x]

            # get epoch number 
            nepoch_df =loaded_data ['nepoch_df']
            nepoch_df.insert(0,'subject',subject)
            df_nepoch_group = pd. concat([df_nepoch_group,nepoch_df])
            
        assert len(sub_dict.keys()) == len(sub_list), 'error: sub number not match'
    
        print('--------- prepare ready for data of ',config_setting)

        # ---set the time for statistic   
        cond1_tidx = [idx for idx,t in enumerate(times) if (t>=cond1_timewindow[0])&(t<=cond1_timewindow[1])]
        cond2_tidx = [idx for idx,t in enumerate(times) if (t>=cond2_timewindow[0])&(t<=cond2_timewindow[1])]
    
        #% -------------- Get Significant Clusters --------------
        # Extract cluster information
        T_obs, clusters, cluster_p_values, H0 = cluster_stats
        good_cluster_inds = np.where(cluster_p_values < p_cluster_forming)[0]
        print('------------  find good cluster ',len(good_cluster_inds))
        print(cluster_p_values)

        # Check if there are any significant clusters
        flag_sig= len(good_cluster_inds)> 0
        # ------- save cluster info
        sig_cidx = [] 
        sig_cluster_info = {}
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            
            # unpack cluster information, get unique indices
            label_inds,time_inds, = clusters[clu_idx]
            t_inds = list(np.unique(time_inds))
            l_inds = list(np.unique(label_inds))
            
            # get index for each cluster
            sig_time = [times[cond1_tidx[x]] for x in t_inds]
            sig_cidx = sig_cidx + l_inds
            
            T_sub = T_obs[np.ix_(l_inds, t_inds)]  
            # add cluster to sig_cluster_info
            sig_cluster_info[i_clu] = {
                't_inds': t_inds,
                'sig_time': sig_time,
                'l_inds': l_inds,
                'sig_cidx': sig_cidx,
                'cluster_p_values': cluster_p_values[good_cluster_inds[i_clu]],
                "cluster_T_values": T_sub,
            }
        #% -------------- Get Significant lables combined for whole time points --------------
        # compute erf of each condition and concat
        cond1_sublabelt  = \
                    np.array(
                    [np.array(
                    [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name]
                    for labi,lab in enumerate(labels) if labi in sig_cidx])
                    for sub in sub_dict.keys() ]
                    )
        cond2_sublabelt  = \
                    np.array(   
                    [np.array(
                    [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name]
                    for labi,lab in enumerate(labels) if labi in sig_cidx])
                    for sub in sub_dict.keys() ]
                    )
        condsubt = np.stack([
            cond1_sublabelt.mean(axis=1), # Average across ROI labels for each condition
            cond2_sublabelt.mean(axis=1)
        ], axis=0)  # (n_conditions, n_subjects, n_times)

        cond_t_y_erf , cond_t_ci_erf = compute_mean_exp2(
                                        data_condsub = condsubt, 
                                            axis_cond= 0,
                                            axis_sub= 1, 
                                            zscore=False, design="within")
        # compute erfdiff
        conddif_sublabelt  = \
                    np.array(
                    [np.array(
                    [ sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond1_content_name]-
                    sub_dict[sub][lab][epoch_avg_type][epoch_subsampling_type][cond2_content_name]
                    for labi,lab in enumerate(labels) if labi in sig_cidx])
                    for sub in sub_dict.keys() ]
                    )


        # Initialize nested dict if doesn't exist
        if any(epoch_data_name not in y for y in [erf_dict, erfdiff_subt_dict, ttest_dict, onset_dict]):
            erf_dict[epoch_data_name] = {}
            erfdiff_subt_dict[epoch_data_name] = {}
            ttest_dict[epoch_data_name] = {}
            onset_dict[epoch_data_name] = {}
        
        erf_dict[epoch_data_name][roi] = {
            'cond_t_y_erf': cond_t_y_erf,
            'cond_t_ci_erf': cond_t_ci_erf,
        }
        conddif_subt = conddif_sublabelt.mean(axis=1) # average across sig labels
        erfdiff_subt_dict[epoch_data_name][roi] = conddif_subt
        
        # Compute mean and CI of cond difference
        conddif_mean = np.mean(conddif_subt, axis=0)  # (n_times,)

        # For difference waves, use between-subject style CI (no Cousineau correction needed)
        confidence_level = 0.95
        n = conddif_subt.shape[0]
        conddif_ci = scipy.stats.sem(conddif_subt, axis=0) * scipy.stats.t.ppf((1 + confidence_level) / 2., n - 1)



    
    #% -------------- ERF difference onset comparation: using half-absolute-maximun 
    if __name__ == "__main__":
        # Define input variables here (e.g., nsub, times, time_window_halfabsmax, etc.)
        # input parameters
        nsub = len(sub_list)
        alpha = 0.05
        times = times
        time_window_halfabsmax =  [-0.3,1]
        time_staticwindow =  [0.25,0.5]
        x_interval = 0.5
        set_time_plotwindow = [-0.5,1.4]

        epoch_data_info = epoch_data_name
        p_cluster_forming = p_cluster_forming
        
        # --- onset-detection settings ---
        flag_find_firstcrossing = 1   # onset = first time the ERF diff crosses the threshold (the actual onset)
        flag_find_closestpoint = 0    # alternative onset = closest point to threshold (off)
        flag_use_abs = 0              # 0: half-max on signed min->max; 1: half of the absolute maximum
        flag_filter  = 1             # apply 30 Hz low-pass (remove >30 Hz artifacts) before onset detection
        flag_smooth = 1              # apply Gaussian smoothing (sigma=50 ms) on top, to avoid false threshold crossings
        flag_avg_halfmax = 1         # use one shared half-max threshold = mean(PFC, POS) instead of per-ROI
        # --- output dirs (raw vs smoothed figures) ---
        jpg_dir = jpg_path
        jpg_smooth_dir = jpg_smooth_path
        pdf_dir = pdf_path
        pdf_smooth_dir = pdf_smooth_path
        flag_plot_cut = 1             # crop the plotted curve to set_time_plotwindow
        plot_timewindow_idx = [idx for idx,t in enumerate(times) if (t>=set_time_plotwindow[0])&(t<=set_time_plotwindow[1])]
        
        # main: calulate the halfmax and conduct statistic; plot the results
        onset_PFC, y_PFC, xidx_PFC, \
        onset_POS, y_POS, xidx_POS, \
        avg_halfmax, \
        latency_diff, standard_error,\
        title,\
        plotdata_erfdiff,\
        plotdata_erfdiff_ci_upper,\
        plotdata_erfdiff_ci_lower= main()
        
        
        print(f" onset_PFC: {onset_PFC} y_PFC: {y_PFC}")
        print(f" onset_POS: {onset_POS} y_POS: {y_POS}")
        print(f"Latency Difference: {latency_diff}")
        print(f"Standard Error (S_D): {standard_error}")
    onset_dict[epoch_data_name] = {
        'nsub': nsub,
        'time_window_halfabsmax': time_window_halfabsmax,
        'time_staticwindow': time_staticwindow,
        'alpha': alpha,
        "flag_plot_cut": flag_plot_cut,
        "plot_timewindow_idx": plot_timewindow_idx,
        'flag_find_firstcrossing': flag_find_firstcrossing,
        'flag_find_closestpoint': flag_find_closestpoint,
        'flag_use_abs': flag_use_abs,
        'flag_filter': flag_filter,
        'flag_smooth': flag_smooth,
        'flag_avg_halfmax': flag_avg_halfmax,
        'onset_PFC': onset_PFC,
        'y_PFC': y_PFC,
        'xidx_PFC': xidx_PFC,
        'onset_POS': onset_POS,
        'y_POS': y_POS,
        'xidx_POS': xidx_POS,
        'avg_halfmax': avg_halfmax,
        'latency_diff': latency_diff,
        'standard_error': standard_error,
    }
    
    #% -------------- ttest on erf difference of PFC vs POS
    # each subject get one value for PFC; one value for POS
    # avg erf difference over statistic window; significant labels: each subject get one value for each roi

    subs_PFC = erfdiff_subt_dict[epoch_data_name]['PFC'][:, cond1_tidx].mean(axis=1) # n_subjects values
    subs_POS = erfdiff_subt_dict[epoch_data_name]['POS'][:, cond1_tidx].mean(axis=1) # n_subjects values

    # Perform the paired t-test 
    t_stat, p_value = stats.ttest_rel(subs_PFC, subs_POS)
    # Adjust p-value for one-tailed test
    if t_stat < 0:  # Check if the direction of the effect matches hypothesis
        one_tailed_p_value = p_value / 2
    else:
        one_tailed_p_value = 1 - (p_value / 2)

    # Calculate means and standard deviations
    pfc_mean = np.mean(subs_PFC)  # Mean of PFC
    pos_mean = np.mean(subs_POS)  # Mean of POS
    pfc_std = np.std(subs_PFC, ddof=1)  # Standard deviation of PFC (ddof=1 for sample SD)
    pos_std = np.std(subs_POS, ddof=1)  # Standard deviation of POS (ddof=1 for sample SD)

    # Output the results
    print(f"df: {len(subs_PFC)-1}")
    print(f"PFC POS t-statistic: {t_stat:.3f}")
    print(f"Two-tailed p-value: {p_value:.3f}, One-tailed p-value: {one_tailed_p_value:.3f}")
    print(f"PFC mean:  {pfc_mean:.3f} STD:{pfc_std:.3f}")
    print(f"POS mean:  {pos_mean:.3f} STD:{pos_std:.3f}")
    ttest_dict[epoch_data_name] = {
        'subs_PFC': subs_PFC,
        'subs_POS': subs_POS,
        't_stat': t_stat,
        'two_tailed_p_value': p_value,
        'one_tailed_p_value': one_tailed_p_value,
        'pfc_mean': pfc_mean,
        'pos_mean': pos_mean,
        'pfc_std': pfc_std,
        'pos_std': pos_std,
        
    }
    save_data = {
        'erf_dict': erf_dict,
        'erfdiff_subt_dict': erfdiff_subt_dict,
        'ttest_dict': ttest_dict,
        'onset_dict': onset_dict,
        'plotdata_erfdiff': plotdata_erfdiff,
        'plotdata_erfdiff_ci_upper': plotdata_erfdiff_ci_upper,
        'plotdata_erfdiff_ci_lower': plotdata_erfdiff_ci_lower,
    }
    pklfile_name = os.path.join(plotdata_path, 
                                f'{title.replace(" ", "_")}.pkl')
    with open(pklfile_name, 'wb') as pickle_file:
        pickle.dump(save_data, pickle_file)

# %%
