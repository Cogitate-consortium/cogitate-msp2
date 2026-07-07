'''
convert_none
read_configfile
read_default_epoch_setting
get_eps_use
get_eps_use_info
extract_epoch_setting
turn_sub_code_to_num



get_power_from_array_morlet
'''
# for check baseline info of pre
import numpy as np 
# for read sublist
import pandas as pd 
import json
import os
import os.path as op
import logging
# decorator from Python dataclasses module,to automatically create the “boring boilerplate” methods
from dataclasses import dataclass
# for statistic testing
import scipy
import mne

from EXP1_help_functions import (bids_root,
                                 read_epochs,
                                 read_fwd,
                                 create_inverse,
                                 crop_stcs,
                                 lowpass_filter)


#%% Define the project directory and default configuration file path
# Repository root: the folder that contains py_code/ and config_files/.
# Derived from this file's location, so it works wherever the repo is cloned
# (this file lives in <Project_Dir>/py_code/).
Project_Dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub_no_T1 = ['SA101','SA102', 'SA104', 'SA110', 'SA111', 'SA152'] # Xuan add 'SA101', has checked with Oscar 

DEFAULT_SETTING_CONFIG_FILE_DIR = os.path.join(Project_Dir,
                                               "config_files", 
                                               "default_setting_config", 
                                               "default_setting_config.json")
DEFAULT_EPOCH_SETTING_NAME = "epoch_noft_nobc_cutpre1000_1000_deci1"
GENERAL_CONFIG_FILE_DIR = os.path.join(Project_Dir,
                                       "py_code",  "general_config.json")


path_allana = os.path.join(bids_root,'derivatives','ana')
path_group_result = os.path.join(bids_root,'derivatives','ana_group')
path_oct6_fs_src = os.path.join(bids_root,'derivatives',"fs") + '/fsaverage/bem/fsaverage-oct-6-src.fif'

#%% preset_plot_timewindow
def preset_plot_timewindow(group_configfolder):
    if 'baseline' in group_configfolder:
        return [-0.7, 1.5]
    else:
        return [-0.5, 1.5]
#%%  Read the general configfile file:
with open(GENERAL_CONFIG_FILE_DIR) as f:
    general_param = json.load(f)

#%%  colordict
colordict = {
'StimUnseen'          :'#AE6BEB',  
"StimUnseen_leftVrigh":'#AE6BEB',
'StimSeen'          :'#611E9E',
"StimSeen_leftVrigh":'#611E9E',
"dATseenStim"       :'#611E9E',
"dATseenStimacti":'#611E9E',
"dATseenStimbase":'#611E9E',
'ObjeUnseen' :'#F5C349',  
'objeUnseen' :'#F5C349',  
'ObjeSeen'   :'#C29016',
'objeSeen'   :'#C29016',
'objetrl'    :'#C29016',
'object'     :'#C29016',
"dATseenObje":'#C29016',
'FaceUnseen' :'#6292BA',  
'FaceSeen'   :'#16456D',
"dATseenFace":'#16456D',
"facetrl"    :'#16456D',
"face"       :'#16456D',
"TargetSeen"          :'#AA4B00',
"TargetSeen_leftVrigh":'#AA4B00',
"Atstim":'#AA4B00',# for 3 dAT + AT 4 condition compartiaton
"Atobje":'#AA4B00',# for 3 dAT + AT 4 condition compartiaton
"Atface":'#AA4B00',# for 3 dAT + AT 4 condition compartiaton
"dATunseenBlan"  :'grey',  
"Blank"          :'grey',
"Blank_leftVrigh":'grey',
"BlankUnseen":'grey',
'StimSeen_Left'   :'#829aa3',# for all Left
'StimUnseen_Left' :'#829aa3',
'StimSeen_Right'   : '#a38b82',# for all Right
'StimUnseen_Right' :'#a38b82',
'POS':'#0173B2',# iit'blue''purple'
'PFC':'#029E73',#
'V1V2':'#04a5fe',##65c8fe',#'blue',#blackpurple',#
'FF': '#cd7f32',#'pink',#
}
#%% 
roi_longname_dict = {
'G_and_S_frontomargin'          :'Fronto-marginal gyrus and sulcus',  
"G_and_S_occipital_inf"   :'Inferior occipital gyrus and sulcus',
'G_and_S_transv_frontopol':'Transverse frontopolar gyri and sulci',
"G_and_S_cingul-Ant":'Anterior part of the cingulate gyrus and sulcus',
"G_and_S_cingul-Mid-Ant"       :'Middle-anterior part of the cingulate gyrus and sulcus',
"G_and_S_cingul-Mid-Post":'Middle-posterior part of the cingulate gyrus and sulcus',
"G_cuneus":'Cuneus',
'G_front_inf-Opercular' :'Opercular part of the inferior frontal gyrus',  
'G_front_inf-Orbital' :'Orbital part of the inferior frontal gyrus',  
'G_front_inf-Triangul'   :'Triangular part of the inferior frontal gyrus',
'G_front_middle'   :'Middle frontal gyrus',
"G_front_sup":'Superior frontal gyrus',
'G_occipital_middle' :'Middle occipital gyrus',  
'G_occipital_sup'   :'Superior occipital gyrus',
"G_oc-temp_lat-fusifor"  :'Lateral occipito-temporal gyrus',  
"G_oc-temp_med-Lingual" :'Lingual gyrus, ligual part of the medial occipito-temporal gyrus',
"G_oc-temp_med-Parahip":'Parahippocampal gyrus, parahippocampal part of the medial occipito-temporal gyrus',
"G_orbital"  :'Orbital gyri',  
"G_pariet_inf-Angular"          :'Angular gyrus',
"G_pariet_inf-Supramar":'Supramarginal gyrus',
"G_precentral"  :'Precentral gyrus',  
"G_rectus"          :'Straight gyrus Gyrus rectus',
"G_subcallosal":'Subcallosal area subcallosal gyrus',
"G_temp_sup-Lateral"  :'Lateral aspect of the superior temporal gyrus',  
"G_temp_sup-Plan_tempo"          :'Planum temporale or temporal plane of the superior temporal gyrus',
"G_temporal_inf":'Inferior temporal gyrus ',
"G_temporal_middle"  :'Middle temporal gyrus',  
"Lat_Fis-ant-Horizont":'Horizontal ramus of the anterior segment of the lateral sulcus',
"Lat_Fis-ant-Vertical":'Vertical ramus of the anterior segment of the lateral sulcus',
"Pole_occipital"  :'Occipital pole',  
"Pole_temporal":'Temporal pole',
"S_calcarine":'Calcarine sulcus',
"S_front_inf"  :'Inferior frontal sulcus',  
"S_front_middle":'Middle frontal sulcus',
"S_front_sup":'Superior frontal sulcus',
"S_interm_prim-Jensen"  :'Sulcus intermedius primus',  
"S_intrapariet_and_P_trans":'Intraparietal sulcus and transverse parietal sulci',
"S_oc_middle_and_Lunatus":'Middle occipital sulcus and lunatus sulcus',
"S_oc_sup_and_transversal"  :'Superior occipital sulcus and transverse occipital sulcus',  
"S_occipital_ant":'Anterior occipital sulcus and preoccipital notch',
"S_oc-temp_lat":'Lateral occipito-temporal sulcus',
"S_orbital_lateral"  :'Lateral orbital sulcus',  
"S_orbital_med-olfact":'Medial orbital sulcus',
"S_orbital-H_Shaped":'Orbital sulci',
"S_precentral-inf-part"  :'Inferior part of the precentral sulcus',  
"S_suborbital":'Suborbital sulcus',
"S_temporal_inf":'S_temporal_inf',
"S_temporal_sup"  :'Superior temporal sulcus',}
#%% convert_none_recursive
def convert_none(value):
    """
    Recursively replace string 'None' with Python None in config dicts/lists.
    """
    if isinstance(value, dict):
        return {k: convert_none(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_none(v) for v in value]
    elif value == "None":
        return None
    else:
        return value
# %%  convert_none_to_string
def convert_none_to_string(value):
    """Convert None type to string 'None'."""
    if value is None:
        return "None"
    else:
        return value
    
#%% read_configfile Define the default configuration file location
def read_configfile(FILE_DIR):
    """
    Loads configuration settings from the default settings JSON file.

    Parameters
    ----------
    config_path : str | None
        Path to the configuration JSON file. If None, uses DEFAULT_SETTING_CONFIG_FILE_DIR.

    Returns
    -------
    dict | None
        Parsed configuration dictionary if successful, otherwise None.
    """
    with open(FILE_DIR, 'r') as f:
        config = json.load(f)
    clean_default_config = convert_none(config)
    return clean_default_config

#%% read_default_epoch_setting
def read_default_epoch_setting( 
                          DEFAULT_EPOCH_SETTING_NAME = DEFAULT_EPOCH_SETTING_NAME):
    print('default epoch setting:\n--', DEFAULT_EPOCH_SETTING_NAME)

    # Load the entire configuration file.
    default_config = read_configfile(FILE_DIR = DEFAULT_SETTING_CONFIG_FILE_DIR)

    # Use the function's argument to look up the specific setting.
    epoch_parameters = default_config[DEFAULT_EPOCH_SETTING_NAME]

    return epoch_parameters
#%% apply_filter_to_epochs
def apply_filter_to_epochs(epochs, l_freq, h_freq):
    """
    Applies optional frequency filtering to a single MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The Epochs object to be filtered.
    l_freq : float | int | str | None
        The lower cutoff frequency in Hz. Accepts None, "None", or numeric strings.
    h_freq : float | int | str | None
        The upper cutoff frequency in Hz. Accepts None, "None", or numeric strings.

    Returns
    -------
    mne.Epochs
        A new, filtered Epochs object.
    """
    epochs_flt = epochs.copy()

    # --- Normalize inputs ---
    l_freq = convert_none(l_freq)
    h_freq = convert_none(h_freq)

    # --- Apply filter or skip ---
    if l_freq is None and h_freq is None:
        print("------------ No filter applied.")
        return epochs_flt
    
    epochs_flt.filter(l_freq=l_freq, h_freq=h_freq)
    
    # Construct message for clarity
    if l_freq is not None and h_freq is not None:
        message = f"Band-pass filter applied: {l_freq}-{h_freq} Hz"
    elif l_freq is not None:
        message = f"High-pass filter applied: {l_freq} Hz"
    else:
        message = f"Low-pass filter applied: {h_freq} Hz"    
    logging.info(message)
    print(f"------------ {message}")
    
    
    return epochs_flt
#%% get_ftinfo
def get_ftinfo(l_freq, h_freq):
    """
    Generates a filename string for filter information.

    Args:
        l_freq (float or None): Lower cutoff frequency.
        h_freq (float or None): Upper cutoff frequency.

    Returns:
        str: Filter info, e.g. 'ft1_40' or 'noft'.
    """
    if l_freq in [None, "None"] and h_freq in [None, "None"]:
        return "noft"

    if not isinstance(l_freq, (int, float)) and l_freq is not None:
        raise TypeError("l_freq must be a number if not None")
    if not isinstance(h_freq, (int, float)) and h_freq is not None:
        raise TypeError("h_freq must be a number if not None")

    ft_info = f"ft{convert_none_to_string(l_freq)}_{convert_none_to_string(h_freq)}".replace('None', 'N')
    print(f"ft {ft_info}")
    return ft_info
#%% get_bcinfo
# def _get_baseline_name(value,  unit='s',n_digit):
#     """
#     Format a single baseline value into a string-safe name.
#     Negative values are prefixed with 'pre'.
#     """
#     val_rounded = np.round(value, n_digit)
#     if value < 0:
#         name = f'pre{np.abs(val_rounded)}'
#     else:
#         name = str(val_rounded)
#     return name.replace('.', '')
def _get_baseline_name(value):
    """
    Format a single baseline value into a string-safe name.
    Negative values are prefixed with 'pre'.
    """
    val_rounded = int(value)
    if value < 0:
        name = f'pre{np.abs(val_rounded)}'
    else:
        name = str(val_rounded)
    return name.replace('.', '')
def get_bcinfo(baseline_on, baseline_off, unit='s'):
    """
    Generates a filename string for baseline correction information.

    Args:
        baseline_on (float or None): Start time of the baseline.
        baseline_off (float or None): End time of the baseline.
        unit (str, optional): Time unit ('s' or 'ms'). Defaults to 's'.
        debug (bool, optional): If True, print the bc_info string.

    Returns:
        str: Baseline correction info, e.g. 'bcpre200_0' or 'nobc'.
    """
    print("input bc", baseline_on, baseline_off)
 
    if baseline_on in [None, "None"] and baseline_off in [None, "None"]:
        return "nobc"
    
    if unit == 's':
        if abs(baseline_on) + abs(baseline_off) < 10:
            baseline_on = baseline_on *1000
            baseline_off = baseline_off *1000
            print("convert bc to ms", baseline_on, baseline_off)
    elif unit == 'ms':
        n_digit = 0
    else:
        raise ValueError("time unit can only be 's' or 'ms'")
    if not isinstance(baseline_on, (int, float)) or not isinstance(baseline_off, (int, float)):
        raise TypeError("baseline_on and baseline_off must be numbers if not None")

    on_name = _get_baseline_name(baseline_on)
    off_name = _get_baseline_name(baseline_off)

    bc_info = f"bc{on_name}_{off_name}".replace('None', 'N')
    print(f"bc {bc_info}")

    return bc_info

#%% get_cutinfo
def get_cutinfo(cut_on, cut_off, unit='s'):
    """
    Generates a filename string for time cropping information.

    Args:
        cut_on (float or None): Start time of the crop window.
        cut_off (float or None): End time of the crop window.
        unit (str, optional): Time unit ('s' or 'ms'). Defaults to 's'.

    Returns:
        str: Crop info, e.g. 'cutpre200_0' or 'nocut'.
    """
    print("input cut", cut_on, cut_off)
 
    if cut_on in [None, "None"] and cut_off in [None, "None"]:
        return "nocut"
    
    if unit == 's':
        if abs(cut_on) + abs(cut_off) < 10:
            cut_on = cut_on *1000
            cut_off = cut_off *1000
            print("convert cut to ms", cut_on, cut_off)
    elif unit == 'ms':
        n_digit = 0
    else:
        raise ValueError("time unit can only be 's' or 'ms'")
    if not isinstance(cut_on, (int, float)) or not isinstance(cut_off, (int, float)):
        raise TypeError("cut_on and cut_off must be numbers if not None")

    on_name = _get_baseline_name(cut_on)
    off_name = _get_baseline_name(cut_off)

    cut_info = f"cut{on_name}_{off_name}".replace('None', 'N')
    print(f"cut {cut_info}")

    return cut_info

#%% apply_baseline_to_epochs
def apply_baseline_to_epochs(epochs, baseline_on, baseline_off):
    """
    Applies optional baseline correction to a single MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The Epochs object to be baseline corrected.
    baseline_on : float | int | str | None
        Start of the baseline period in seconds. Accepts None, "None", or numeric strings.
    baseline_off : float | int | str | None
        End of the baseline period in seconds. Accepts None, "None", or numeric strings.

    Returns
    -------
    mne.Epochs
        A new, baseline-corrected Epochs object.
    """
    epochs_bl = epochs.copy()

    # --- Normalize inputs ---
    baseline_on = convert_none(baseline_on)
    baseline_off = convert_none(baseline_off)
    
    
    # --- Apply baseline or skip ---
    if baseline_on is None and baseline_off is None:
        print("------------ No baseline correction applied.")
        return epochs_bl
    epochs_bl.apply_baseline((baseline_on, baseline_off))
    
    message = f"Baseline correction applied: {baseline_on}s to {baseline_off}s"
    logging.info(message)
    print(f"------------ {message}")
    
    
    return epochs_bl


#%% apply_crop_to_epochs
def apply_crop_to_epochs(epochs, cut_on, cut_off):
    """
    Applies optional time cropping to a single MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The Epochs object to be cropped.
    cut_on : float | int | str | None
        Start time of the crop window in seconds. Accepts None, "None", or numeric strings.
    cut_off : float | int | str | None
        End time of the crop window in seconds. Accepts None, "None", or numeric strings.

    Returns
    -------
    mne.Epochs
        A new, cropped Epochs object.
    """
    epochs_cut = epochs.copy()

    # --- Normalize inputs ---
    cut_on = convert_none(cut_on)
    cut_off = convert_none(cut_off)

    # --- Apply crop or skip ---
    if cut_on is None and cut_off is None:
        print("------------ No crop applied.")
        return epochs_cut
        
    epochs_cut.crop(cut_on, cut_off)
    
    message = f"Crop applied: {cut_on}s to {cut_off}s"
    logging.info(message)
    print(f"------------ {message}")
    

    return epochs_cut
#%% apply_decimation_to_epochs
def apply_decimation_to_epochs(epochs, epoch_decim):
    """
    Applies optional decimation to a single MNE Epochs object."""
    epochs_deci = epochs.copy()
    # --- Normalize inputs ---
    epoch_decim = convert_none(epoch_decim)
    
    if epoch_decim is None or epoch_decim == 1:
        print("------------ No decimation applied.")
        return epochs_deci
    epochs_deci.decimate(epoch_decim, verbose=False)
    message = f"Decimation applied: factor {epoch_decim}"
    logging.info(message)
    print(f"------------ {message}")
    return epochs_deci

#%% get_eps_use
def get_eps_use(subject,
                exp_id,
                task_id,
                trl_condition,
                l_freq,
                h_freq,
                baseline_on,
                baseline_off,
                epoch_decim ,
                cut_on ,
                cut_off,
                debug=False,
                **kwargs
    ):
    """
    Loads and processes MNE Epochs data using a modular pipeline.
    
    This function reads epochs and resting-state data, then applies a
    sequence of filtering, baseline correction, cropping, and decimation
    steps by calling dedicated helper functions.
    """
    logging.info(f"------ Starting pipeline for subject: {subject} ------")

    # 1. Read MEG epoched data
    epochs = read_epochs(
                subject_id   = subject, 
                visit_id     = exp_id, 
                bids_task_v2 = task_id, 
                preload      = True, # Preloading is better for sequential processing
                pick_meg_only= True, 
                is_rest      = False, 
                debug        = debug)

    # 2. Read MEG resting-state epoched data (if applicable)
    if exp_id == "V2":
        epochs_rs = read_epochs(
            subject_id   = subject, 
            visit_id     = exp_id, 
            bids_task_v2 = task_id, 
            preload      = True, 
            pick_meg_only= True, 
            is_rest      = True, 
            debug        = debug)
        use_rs_noise = True
        print('--------- Experiment V2 use resting-state epoched data as noise')
    else:
        epochs_rs = None
        use_rs_noise = False

    # --- 3. Modular Processing Pipeline ---
    # Select condition before processing
    trl_condition = convert_none(trl_condition)
    if trl_condition is None:
        epochs_selected = epochs
    else:
        epochs_selected = epochs[trl_condition]
    

    # Step A: Apply filtering to both task and resting-state data
    eps_flt = apply_filter_to_epochs(epochs_selected, l_freq, h_freq)
    eprs_flt = apply_filter_to_epochs(epochs_rs, l_freq, h_freq) if epochs_rs is not None else epochs_rs

    # Step B: Apply baseline correction (only to task data)
    eps_flt_bc= apply_baseline_to_epochs(eps_flt, baseline_on, baseline_off)

    # Step C: Apply crop 
    eps_flt_bc_cut = apply_crop_to_epochs(eps_flt_bc, cut_on, cut_off)
    # This is only used in presaved power
    eps_flt_cut = apply_crop_to_epochs(eprs_flt, cut_on, cut_off) if eprs_flt is not None else eprs_flt

    # Step D: Apply decimation to both
    eps_use  = apply_decimation_to_epochs(eps_flt_bc_cut, epoch_decim)
    eprs_use = apply_decimation_to_epochs(eps_flt_cut, epoch_decim) if eps_flt_cut is not None else eps_flt_cut

    # --- 4. Package and Return Results ---
    eps_use_result = {
        'eps_use': eps_use,
        'eprs_use': eprs_use,
        'epochs_metadata': eps_use.metadata,
        'times': eps_use.times,
        'ch_names': eps_use.ch_names,
        'sfreq': eps_use.info['sfreq'],
        'use_rs_noise': use_rs_noise
    }
    logging.info(f"------ Finished pipeline for subject: {subject} ------")
    
    return eps_use_result

#%% get_eps_use_info
@dataclass
class EpochSettingInfo:
    l_freq: float
    h_freq: float
    baseline_on: float
    baseline_off: float
    cut_on: float
    cut_off: float
    epoch_decim: int
    bc_info: str
    cut_info: str
    ft_info: str
    epoch_info:str
    
def get_eps_use_info (
                l_freq,
                h_freq,
                baseline_on,
                baseline_off,
                cut_on ,
                cut_off,
                epoch_decim ,
                showinfo: bool = True,
     **kwargs
) -> EpochSettingInfo:
    # Normalize "None" strings to real None
    l_freq       = convert_none(l_freq)
    h_freq       = convert_none(h_freq)
    baseline_on  = convert_none(baseline_on)
    baseline_off = convert_none(baseline_off)
    cut_on       = convert_none(cut_on)
    cut_off      = convert_none(cut_off)
    epoch_decim  = convert_none(epoch_decim)
    
    ft_info = get_ftinfo(l_freq, h_freq)
    bc_info = get_bcinfo(baseline_on, baseline_off)
    deci_info = f'deci{epoch_decim}' if epoch_decim is not None else 'deci1'
    cut_info = get_cutinfo(cut_on, cut_off)
    print(f"cut {cut_info}")
    
    epoch_info = f"{ft_info}_{bc_info}_{deci_info}_{cut_info}"


    
    # Light validation
    if epoch_decim <= 0:
        raise ValueError(f"epoch_decim must be positive, got {epoch_decim}.")
    if (baseline_on is not None and baseline_off is not None) and (baseline_on >= baseline_off):
        raise ValueError(f"baseline_on ({baseline_on}) must be < baseline_off ({baseline_off}).")
    if (cut_on is not None and cut_off is not None) and (cut_on >= cut_off):
        raise ValueError(f"cut_on ({cut_on}) must be < cut_off ({cut_off}).")
    if showinfo:
        print(f"[Filter]   {ft_info}")
        print(f"[Baseline] {bc_info}")
        print(f"[Cut]      {cut_info}")
        print(f"[Decim]    {epoch_decim}")
        print(f"[Epoch]    {epoch_info}")

    return EpochSettingInfo(
        l_freq=l_freq,
        h_freq=h_freq,
        baseline_on=baseline_on,
        baseline_off=baseline_off,
        cut_on=cut_on,
        cut_off=cut_off,
        epoch_decim=epoch_decim,
        bc_info=bc_info,
        cut_info=cut_info,
        ft_info=ft_info,
        epoch_info = epoch_info
    )
    

# %% turn_sub_code_to_num
def turn_sub_code_to_num(s):
    # Extract letters and digits
    letters = s[:2]
    digits = s[2:]
    
    # Convert letters to numbers and sum them
    letter_sum = sum(ord(char) - ord('A') + 1 for char in letters)
    
    # Take the last three digits as a number
    last_three_digits = int(digits)
    
    # Calculate the result
    result = letter_sum * last_three_digits
    
    return result

#%% get_fw_inv
def get_fw_inv(subject,
                exp_id,
                task_id,
                inv_method,
                active_win,
                eps_use,
                eprs_use,
                use_rs_noise,
                **kwargs
    ):
    if kwargs:
        print('------start get fw inv ',subject)
        print(f"Subject: {subject}")
        print(f"Experiment ID: {exp_id}")
        print(f"Task ID: {task_id}")
        print(f"Inversion Method: {inv_method}")
        print(f"Active Window: {active_win}")
        print(f"Eps Use: {len(eps_use)}")
        print(f"Use Resting-State Noise: {use_rs_noise}")
    
    # Read forward model
    fwd = read_fwd(
            subject_id   = subject, 
            visit_id     = exp_id, 
            bids_task_v2 = task_id, 
            inv_method   = inv_method)

    # Create inverse solution
    inverse, rank, covs = create_inverse(
            subject_id  = subject, 
            visit_id    = exp_id, 
            epochs      = eps_use, 
            fwd         = fwd, 
            use_rs_noise= use_rs_noise, 
            epochs_rs   = eprs_use,
            fr_band     = None, 
            active_win  = active_win, 
            baseline_win= None, 
            plot_cov    = False)


    fw_inv_result = {
        'src':inverse['src'],
        'fwd':fwd,
        'inverse':inverse,
        'covs':covs,
        'rank':rank,
    }
    print('------finish get fw inv ',subject)
    return  fw_inv_result
#%% get_stcs_allsrc
def get_stcs_allsrc(
          subject,
        eps_use,
        snr,
        stc_method,
        pick_ori,
        inverse,
        **kwargs
        ):
    # inverse   = kwargs.get('inverse', None)
    # eps_use   = kwargs.get('eps_use', None)
    if kwargs:
        print('------ Start get_stcs_allsrc')
        print(f"Subject: {subject}")
        print(f"SNR: {snr}")
        print(f"STC Method: {stc_method}")
        print(f"Pick Orientation: {pick_ori}")
        print(f"inverse: {inverse}")
        print(f"eps_use: {len(eps_use)}")
    
    stcs_metadata = eps_use.metadata.fillna('no').reset_index(drop = True)
    
    # get stcs_epoch
    lambda2 = 1.0 / snr**2
    stcs_epoch = mne.minimum_norm.apply_inverse_epochs(
        eps_use,
        inverse_operator = inverse,
        lambda2 =lambda2,
        method = stc_method ,
        label= None,
        pick_ori=pick_ori,
        )
    times =  stcs_epoch[0].times
    sfreq =  stcs_epoch[0].sfreq
    stcs_use_result = {
        'stcs_epoch':stcs_epoch,
        'times': times,
        'sfreq' :sfreq,
        'stcs_metadata':stcs_metadata,
    }
    print('------ finish get_stcs_allsrc subject:', subject)
    return stcs_use_result
#%% get_stcs_of_label
def get_stcs_of_label(
        subject,
        label,
        eps_use,
        snr,
        stc_method,
        pick_ori,
        inverse,
        **kwargs
        ):
    if kwargs:
        print('------ Start get_stcs_of_label subject:', subject)
        print(f"Subject: {subject}")
        print(f"Label: {label}")
        print(f"SNR: {snr}")
        print(f"STC Method: {stc_method}")
        print(f"Pick Orientation: {pick_ori}")

    # snr       = kwargs.get('snr', None)
    # stc_method= kwargs.get('stc_method', None)
    # pick_ori   = kwargs.get('pick_ori', None)
    # eps_use   = kwargs.get('eps_use', None)
    # inverse   = kwargs.get('inverse', None)
    # eps_use   = kwargs.get('eps_use', None)
    
    
    stcs_metadata = eps_use.metadata.fillna('no').reset_index(drop = True)


    # get stcs_epoch
    lambda2 = 1.0 / snr**2
    stcs_epoch = mne.minimum_norm.apply_inverse_epochs(
        eps_use,
        label   = label,
        inverse_operator = inverse,
        lambda2 =lambda2,
        method  = stc_method ,
        pick_ori=pick_ori,
        nave=1,)  
    
    # stcs_epoch[0].data shape (n_dipoles, n_times)
    times =  stcs_epoch[0].times
    sfreq =  stcs_epoch[0].sfreq
    stcs_use_result = {
        'stcs_epoch':stcs_epoch,
        'times': times,
        'sfreq' :sfreq,
        'stcs_metadata':stcs_metadata,
    }
    print('------ finish get_stcs_of_label subject:', subject)
    return stcs_use_result
# %% read_labels_exp2
def read_labels_exp2(bids_paths, 
                     parc     ="aparc.a2009s", 
                     labels_list       = [], # this is the raw roi name to load from parc
                     labels_list_names = [],
                     rois_list         = [], # "iit_1","gnw" if no label_list then read roi from iit_gnw_rois.json 
                     combine_labels_list=True, 
                     merge_hemi=True,
                     **kwargs):
    '''
    lb_list = read_labels_exp2(
                bids_paths = BidsPath(subject_id = subject, 
                                    visit_id   = exp_id), 
                parc       ="wang2015_mplbl", 
                labels_list= [],
                rois_list  = ["V1V2_responsive_activewin_vs_baseline_vertices"], 
                combine_labels_list=True, 
                merge_hemi=True
                )  

    '''
    if kwargs:
        print('------ Start reading labels for subject:', bids_paths.subject_id)
        print(f"Subject ID: {bids_paths.subject_id}")
        print(f"Visit ID:   {bids_paths.visit_id}")
        print(f"Parcellation: {parc}")
        print(f"Labels List: {labels_list}")
        print(f"ROIs List: {rois_list}")
        print(f"Combine Labels List: {combine_labels_list}")
        print(f"Merge Hemispheres: {merge_hemi}")
        
    #  load vertex label 
    if any(['vertice' in x for x in rois_list]):
        assert len(rois_list) == 1, "When using vertex-based ROIs, provide exactly one entry in rois_list."
        verttype = rois_list[0]
        vert_dir= os.path.join(path_allana,'pipeline_IIT_vertices_select')
        # if   any(['activewin_vs_baseline' in x for x in rois_list]):
        #     paths_of_label = [os.path.join(vert_dir,f"{verttype}_loweralpha",'label'),
        #                      os.path.join(vert_dir,f"{verttype}_highergamma",'label')]
        # else:
        #     paths_of_label = [os.path.join(vert_dir,f"{verttype}_loweralpha", 'logpow','label'),#'logbcratiopow'
        #                       os.path.join(vert_dir,f"{verttype}_highergamma",'logpow','label')]#'logbcratiopow'
        paths_of_label = [os.path.join(vert_dir,'_label'),]
        label_files_r = [   os.path.join(path, f) 
                            for path in paths_of_label
                            for f in os.listdir(path) if verttype in f
                            if bids_paths.subject_id in f and f.endswith('-rh.label')
                            ]
        label_files_l = [   os.path.join(path, f) 
                            for path in paths_of_label
                            for f in os.listdir(path) if verttype in f
                            if bids_paths.subject_id in f and f.endswith('-lh.label')
                            ]
        label_file_lh = label_files_l + label_files_r

        flag_r = len(label_files_r)> 0
        flag_l = len(label_files_l)> 0
        
        labels ={}
        if flag_r | flag_l:
            if merge_hemi:
                print('------- load ',verttype,' rl sig vertices')
                labels [verttype]= np.sum([mne.read_label(filename = label_file , 
                                                            subject  = bids_paths.subject_id, 
                                                            color=None,  
                                                            verbose=None) 
                                                    for label_file in label_file_lh])
            else:
                if flag_r:
                    print('------- load ',verttype,' right hemi sig vertices')
                    labels[f'{verttype}_rh'] = np.sum([mne.read_label(filename = label_file , 
                                                                        subject  = bids_paths.subject_id, 
                                                                        color=None,  
                                                                        verbose=None) 
                                                            for label_file in label_files_r])
                else:
                    print('------- no   ',verttype,' right hemi sig vertices')
                if flag_l:
                    print('------- load ',verttype,' left hemi sig vertices')
                    labels[f'{verttype}_lh'] = np.sum([mne.read_label(filename = label_file , 
                                                                        subject  = bids_paths.subject_id, 
                                                                        color=None,  
                                                                        verbose=None) 
                                                            for label_file in label_files_l])
                else:
                    print('------- no   ',verttype,' left hemi sig vertices')
        else:
            print('------- ',verttype,' rl no sig vertices')
            print('label is empty ',labels)
            return labels
                    
    else:
       
        # Add bids path
        bids_paths.add_deriv("roilabel", "rois")
        
        # Read labels from FS parc
        if bids_paths.subject_id in sub_no_T1: 
            labels_atlas = mne.read_labels_from_annot(
                "fsaverage", 
                parc=parc,
                subjects_dir=bids_paths.fs_deriv_root)
        else:
            labels_atlas = mne.read_labels_from_annot(
                "sub-"+bids_paths.subject_id, 
                parc=parc,
                subjects_dir=bids_paths.fs_deriv_root)
        # labels_atlas_names = [l.name for l in labels_atlas]
        
        # Create labels for selected ROIs
        labels = {}
        if labels_list:
            print('input labels_list ',labels_list)
            # Loop over labels
            for labi in range(len(labels_list)):
                lab = labels_list[labi]
                
                # print(lab)
                if isinstance(lab,list):
                    lab_name = labels_list_names[labi]
                    # print(lab)ƒ
                    if merge_hemi:
                        print('merge ',lab)
                        labels[lab_name] = np.sum(
                                [l for l in labels_atlas for lab_one in lab if lab_one in l.name ])
                        
                    else:
                        raise NotImplementedError(
                            "sep-hemi (merge_hemi=False) not supported for list-type labels")
                else:
                    if merge_hemi:
                        print('merge ',lab)
                        labels[lab] = np.sum(
                            [l for l in labels_atlas if lab in l.name])
                    else:
                        print('sep-hemi ',lab)
                        labels[lab+ "_lh"] = [l for l in labels_atlas if  lab in l.name and "-lh" in l.name]
                        labels[lab+ "_rh"] = [l for l in labels_atlas if  lab in l.name and "-rh" in l.name]
                # Combine labels
                if combine_labels_list:
                    labels = np.sum(labels)
        else:
            print('no labels_list')
            # Read GNW and IIT ROI list
            f = open(op.join(bids_paths.rois_deriv_root,
                            'iit_gnw_rois.json'))
            gnw_iit_rois = json.load(f)
            
            # Loop over labels
            for roi in rois_list:
                for lab in gnw_iit_rois['surf_labels'][roi]:
                    print(lab)
                    # Fix the label name to match the template one
                    if bids_paths.subject_id in sub_no_T1:
                        lab = lab.replace('&','_and_')
                    
                    if merge_hemi:
                        labels[f"{roi[:3]}_{lab.replace('_and_','&')}"] = np.sum(
                            [l for l in labels_atlas if lab in l.name])
                    else:
                        labels[f"{roi[:3]}_{lab.replace('_and_','&')}_lh"] = [l for l in labels_atlas if l.name == lab+"-lh"]
                        labels[f"{roi[:3]}_{lab.replace('_and_','&')}_rh"] = [l for l in labels_atlas if l.name == lab+"-rh"]
                
                # Combine labels in single rois
                labels[roi[:3]+"_all"] = np.sum([l for l_name, l in labels.items() if roi[:3] in l_name])
                
                # Xuan add: Combine labels in single hemisphere
                if not merge_hemi:
                    labels[roi[:3]+"_all_lh"] = np.sum([l for l_name, l in labels.items() if (roi[:3] in l_name) & ('_lh' in l_name)])
                    labels[roi[:3]+"_all_rh"] = np.sum([l for l_name, l in labels.items() if (roi[:3] in l_name) & ('_rh' in l_name)])

    return labels
#%% get_label_vertidx
def get_label_vertidx(lb_use, 
                src,
                flag_return_dict=0
                ):
    '''
            # get vertno index
            vidx_l_dic,vidx_r_dic,vidx_dic,vidx_dic_allowoverlap,\
                    flip_dict ,flip_dict_allowoverlap,flip_l_dic,flip_r_dic,\
                    v_l_dic,v_r_dic,v_dic,v_dic_allowoverlap ,\
                        nvidx_l_dic,nvidx_r_dic,nvidx_dic,nvidx_dic_allowoverlap=\
                            get_label_vertidx(lb_use =labels,src = src_load)
    '''
    src_vertidx_list = [x['vertno'] for x in src ]
    src_vertidx = list(src[0]['vertno']) + list(src[1]['vertno'])
    
    v_l_dic={}
    v_r_dic={}
    vidx_l_dic={}
    vidx_r_dic={}

    flip_l_dic={}
    flip_r_dic={}
    
    # label of two hemisphere label.lh.vertices; label of one hemisphere label.vertices
    for label_name_withhemi in list(lb_use.keys()):
        print(label_name_withhemi)

        label_name = label_name_withhemi.replace('_lh','').replace('_rh','')
        lb_1label = lb_use[label_name_withhemi]
        
        # when vertex roi only have results of one hemisphere
        if 'vertice' in label_name_withhemi :
            if '+' not in lb_1label.name :
                label_name = label_name_withhemi
                label_name_withhemi = lb_use[label_name_withhemi].name.replace('-lh','_lh').replace('-rh','_rh')
                if  'rh' not in label_name_withhemi:
                    v_r_dic[label_name]    = []
                    vidx_r_dic[label_name] = []
                    flip_r_dic[label_name] = []
                elif 'lh' not in label_name_withhemi:
                    v_l_dic[label_name]    = []
                    vidx_l_dic[label_name] = [] 
                    flip_l_dic[label_name] = []
                else:
                    ValueError
        
        if ('_lh' in label_name_withhemi) or ('-lh' in label_name_withhemi):
            if ('all' in label_name_withhemi ) | ('vertice' in  label_name_withhemi):
                lb_1label_use = lb_1label
            else:
                lb_1label_use = lb_1label[0]
                
            v_l_dic[label_name]    = [v for v in lb_1label_use.vertices if v  in src_vertidx_list[0]]
            vidx_l_dic[label_name] = [list(src_vertidx_list[0]).index(v) for v in v_l_dic[label_name]] 
            flip_l_dic[label_name] = list( mne.label_sign_flip(lb_1label_use, src))
        
        elif ('_rh' in label_name_withhemi) or ('-rh' in label_name_withhemi):
            if ('all' in label_name_withhemi) | ('vertice' in label_name_withhemi) :
                lb_1label_use = lb_1label
            else:
                lb_1label_use = lb_1label[0]
                
            v_r_dic[label_name] = [v for v in lb_1label_use.vertices if v  in src_vertidx_list[1]]
            vidx_r_dic[label_name] = [list(src_vertidx_list[1]).index(v) for v in v_r_dic[label_name]]
            flip_r_dic[label_name] = list( mne.label_sign_flip(lb_1label_use, src))
            
        else:
            v_l_dic[label_name] = [v for v in lb_1label.lh.vertices if v  in src_vertidx_list[0]]
            v_r_dic[label_name] = [v for v in lb_1label.rh.vertices if v  in src_vertidx_list[1]]

            vidx_l_dic[label_name] = [list(src_vertidx_list[0]).index(v) for v in v_l_dic[label_name]] 
            vidx_r_dic[label_name] = [list(src_vertidx_list[1]).index(v) for v in v_r_dic[label_name]]
            
            flip_l_dic[label_name] = list( mne.label_sign_flip(lb_1label.lh, src))
            flip_r_dic[label_name] = list( mne.label_sign_flip(lb_1label.rh, src))
            
    v_dic = {}
    v_dic_allowoverlap = {}
    vidx_dic = {}
    vidx_dic_allowoverlap = {}
    flip_dict ={}
    flip_dict_allowoverlap = {}
    
    for label_name_withhemi in list(lb_use.keys()):
        label_name = label_name_withhemi.replace('_lh','').replace('_rh','')
        
        v_dic[label_name] = v_l_dic[label_name] + v_r_dic[label_name]
        v_dic_allowoverlap[label_name] = list(set( v_dic[label_name]))
        
        v_inuse_lr_common = np.intersect1d(v_l_dic[label_name],v_r_dic[label_name])
        vidx_dic[label_name]              = vidx_l_dic[label_name] +  \
                                [list(src_vertidx_list[1]).index(v) + len(src_vertidx_list[0])\
                                                                    for v in v_r_dic[label_name] 
                                                                          if v not in v_inuse_lr_common] 
        vidx_dic_allowoverlap[label_name] = vidx_l_dic[label_name]+  \
                                [list(src_vertidx_list[1]).index(v) + len(src_vertidx_list[0])\
                                                                    for v in v_r_dic[label_name] ] 
        flip_dict [label_name] = flip_l_dic[label_name] + [flip_r_dic[label_name][ii] for ii,v in enumerate(v_r_dic[label_name] ) if v not in v_inuse_lr_common]
        flip_dict_allowoverlap [label_name] = flip_l_dic[label_name]+ flip_r_dic[label_name]
        assert len(flip_dict [label_name]) == len(vidx_dic[label_name]),'flip dict not right'
 
    roi_types = [x.replace('_all','') for x in list(lb_use.keys()) if 'all'  in x]  

    nvidx_l_dic={key:len(value) for key,value in vidx_l_dic.items()}
    nvidx_r_dic={key:len(value) for key,value in vidx_r_dic.items()}
    nvidx_dic = {key:len(value) for key,value in vidx_dic.items()}
    nvidx_dic_allowoverlap = {key:len(value) for key,value in vidx_dic_allowoverlap.items()}
        
    label_vertidx_dict = {
                        'vidx_l_dic': vidx_l_dic,
                        'vidx_r_dic': vidx_r_dic,
                        'vidx_dic': vidx_dic,
                        'vidx_dic_allowoverlap': vidx_dic_allowoverlap,
                        'flip_dict': flip_dict,
                        'flip_dict_allowoverlap': flip_dict_allowoverlap,
                        'flip_l_dic': flip_l_dic,
                        'flip_r_dic': flip_r_dic,
                        'v_l_dic': v_l_dic,
                        'v_r_dic': v_r_dic,
                        'v_dic': v_dic,
                        'v_dic_allowoverlap': v_dic_allowoverlap,
                        'nvidx_l_dic': nvidx_l_dic,
                        'nvidx_r_dic': nvidx_r_dic,
                        'nvidx_dic': nvidx_dic,
                        'nvidx_dic_allowoverlap': nvidx_dic_allowoverlap }
        
    if flag_return_dict:
        return label_vertidx_dict
    else:
        return (vidx_l_dic,vidx_r_dic,vidx_dic,vidx_dic_allowoverlap,
                flip_dict ,flip_dict_allowoverlap,
                flip_l_dic,flip_r_dic,v_l_dic,v_r_dic,v_dic,
                v_dic_allowoverlap,nvidx_l_dic,nvidx_r_dic,nvidx_dic,nvidx_dic_allowoverlap)




# %% get_sublist_fromparam
def get_sublist_fromparam(param):
    '''
    get sub list based on behavior and meg check
    sub_list_all = get_sublist_fromparam(param)
    '''
    for key,val in param.items():
        # print(key,val)
        
        globals()[key] = val
        if key == "sub_list_name":
            sub_list_name = val  # Assign val to sub_list_name
        if key == "sub_list_name":
            print(sub_list_name)
        exec(f"{key} = val") 
        assert key in locals(),f'error: {key} not exec'
    

    checksub_behavior = os.path.join(bids_root, 'derivatives', 'qcs', 'ses-v2-analysis-subs.csv')
    df_checksub_bh = pd.read_csv(checksub_behavior)

    df_checksub_bh ['allcondi_false'] = df_checksub_bh.apply(lambda x: 
        all(x[col]==False for col in df_checksub_bh.columns if '_min_' in col),axis = 1)
    max_sub = list(np.sort(df_checksub_bh[df_checksub_bh['allcondi_false']==False]['sub_code'].unique()))
    print('behavior check max sub number',len(max_sub))
    
    
    checksub_meg = os.path.join(bids_root, 'derivatives', 'qcs', 'MEG_docs', 'MEG_n_valid_trials.csv')
    df_checksub_meg = pd.read_csv(checksub_meg)
    # below is not right as the V1V2 of some subject has been in dirrent grid and so the results of V1V2 part is not right
    # checksub_vertROI = os.path.join(path_group_result, 'ses-v2-vertROI-sub.csv')
    checksub_vertROI = os.path.join(path_allana, 'pipeline_IIT_vertices_select', '_label_info', 'ses-v2-vertROI-sub.csv')

    df_checksub_vertROI = pd.read_csv(checksub_vertROI)
    
    if 'sub_list_name' in list(param.keys()):
        print('--- one sub_list')
        # get subject based on behavior
        df_groupcheck = df_checksub_bh.copy()
        if 'corresponding_sub_check_list_logic' in param.keys():
            print('----- 1 behavior condition',corresponding_sub_check_list)
            if corresponding_sub_check_list_logic == "or":
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                not all(x[col]==False for col in corresponding_sub_check_list ),axis = 1)
                
            elif corresponding_sub_check_list_logic == "and":  
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    all(x[col]==True for col in corresponding_sub_check_list ),axis = 1)
            else:
                ValueError
            df_sub_bh = df_groupcheck[df_groupcheck['condi_true']==True]
            
        elif 'corresponding_sub_check_list_logics' in param.keys():   
            print('-----multiple behavior condition',corresponding_sub_check_lists)

            for idx,logic_cond in enumerate(corresponding_sub_check_list_logics):
                if logic_cond == "or":
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_lists[idx] ),axis = 1)
                if logic_cond == "and":  
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_lists[idx]  ),axis = 1)
            df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
            df_sub_bh = df_groupcheck[df_groupcheck['condi_true']==True]
            
        else :
            print('-----no behavior condition max',len(max_sub))
            df_sub_bh = df_checksub_bh[df_checksub_bh['allcondi_false']!=True]
        print('-----after behavior sub number',len(df_sub_bh))
        removed_sub_behavior = [x for x in max_sub if x not in df_sub_bh['sub_code'].unique()]
        print(f'-----  sub in max_sub are removed in behavior {removed_sub_behavior}')
        
        
        df_groupcheck = df_checksub_meg.copy()
        if 'corresponding_sub_check_meg_list_logic' in param.keys():
            print('----- 1 meg condition',corresponding_sub_check_meg_list)

            if corresponding_sub_check_meg_list_logic == "or":
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                not all(x[col]==False for col in corresponding_sub_check_meg_list ),axis = 1)
                
            elif corresponding_sub_check_meg_list_logic == "and":  
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    all(x[col]==True for col in corresponding_sub_check_meg_list ),axis = 1)
            else:
                ValueError

            df_sub_meg = df_groupcheck[df_groupcheck['condi_true']==True]
        elif 'corresponding_sub_check_meg_list_logics' in param.keys():    
            print('-----multiple meg condition',corresponding_sub_check_meg_lists)

            for idx,logic_cond in enumerate(corresponding_sub_check_meg_list_logics):
                if logic_cond == "or":
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_meg_lists[idx] ),axis = 1)
                if logic_cond == "and":  
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_meg_lists[idx]  ),axis = 1)
            if 'corresponding_sub_check_meg_lists_logic' in param.keys():
                if corresponding_sub_check_meg_lists_logic =='and':
                    df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
            else:
                df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                    any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
            df_sub_meg = df_groupcheck[df_groupcheck['condi_true']==True]
            
        else :
            print('-----no meg condition max',df_checksub_meg['subject'].nunique())
            df_sub_meg = df_checksub_meg
        print('-----in MEG check how many left',len(df_sub_meg))   
        removed_sub_meg = [x for x in max_sub if x not in df_sub_meg['subject'].unique()]
        print(f'----- {len(removed_sub_meg)} sub in max_sub are removed in MEG \n{removed_sub_meg}')
        lefted_sub_meg = [x for x in df_sub_bh['sub_code'].unique() if x in df_sub_meg['subject'].unique()]
        print(f'----- {len(lefted_sub_meg)} sub in behavior are lefted in MEG')
        removed_sub_meg_bh = [x for x in df_sub_bh['sub_code'].unique() if x not in df_sub_meg['subject'].unique()]
        print(f'----- {len(removed_sub_meg_bh)} sub is behavior are removed in MEG \n{removed_sub_meg_bh}')   
        
        df_groupcheck = df_checksub_vertROI.copy()
        if 'corresponding_sub_check_vertROI_list_logic' in param.keys():
            print('----- 1 vertROI condition',corresponding_sub_check_vertROI_list)

            if corresponding_sub_check_vertROI_list_logic == "or":
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                not all(x[col]==False for col in corresponding_sub_check_vertROI_list ),axis = 1)
                
            elif corresponding_sub_check_vertROI_list_logic == "and":  
                df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    all(x[col]==True for col in corresponding_sub_check_vertROI_list ),axis = 1)
            else:
                ValueError

            df_sub_vertROI = df_groupcheck[df_groupcheck['condi_true']==True]
        elif 'corresponding_sub_check_vertROI_list_logics' in param.keys():    
            print('-----multiple vertROI condition',corresponding_sub_check_vertROI_lists)

            for idx,logic_cond in enumerate(corresponding_sub_check_vertROI_list_logics):
                if logic_cond == "or":
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_vertROI_lists[idx] ),axis = 1)
                if logic_cond == "and":  
                    df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_vertROI_lists[idx]  ),axis = 1)
            df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
            df_sub_vertROI = df_groupcheck[df_groupcheck['condi_true']==True]
            
        else :
            print('-----no vertROI condition max',df_checksub_vertROI['subject'].nunique())
            df_sub_vertROI = df_checksub_vertROI
        print('-----in vertROI check how many left',len(df_sub_vertROI)) 

        removed_sub_vertROI = [x for x in max_sub if x not in df_sub_vertROI['subject'].unique()]
        print(f'----- {len(removed_sub_vertROI)} sub in max_sub are removed in vertROI {removed_sub_vertROI}')

        removed_sub_vertROI_bhmeg = [x for x in lefted_sub_meg if x not in df_sub_vertROI['subject'].unique()]
        print(f'----- {len(removed_sub_vertROI_bhmeg)} sub in behavior & meg are removed in vertROI /n{removed_sub_vertROI_bhmeg}')
        
        sub_list_all = list(np.sort(
        [x for x in df_sub_bh['sub_code'].unique() if (x in df_sub_meg['subject'].unique()) 
         and (x in df_sub_vertROI['subject'].unique())]   
        ))
        beh_remove = [x for x in max_sub if x not in df_sub_bh['sub_code'].unique()]
        meg_remove = [x for x in max_sub if x not in df_sub_meg['subject'].unique()]
        vert_remove = [x for x in max_sub if x not in df_sub_vertROI['subject'].unique()]
        bh_not_meg = [x for x in df_sub_bh['sub_code'].unique() if x not in df_sub_meg['subject'].unique()]
        meg_not_bh = [x for x in df_sub_meg['subject'].unique() if x not in df_sub_bh['sub_code'].unique()]
        bh_not_vert = [x for x in df_sub_bh['sub_code'].unique() if x not in df_sub_vertROI['subject'].unique()]
        vert_not_bh = [x for x in df_sub_vertROI['subject'].unique() if x not in df_sub_bh['sub_code'].unique()]
        meg_not_vert = [x for x in df_sub_meg['subject'].unique() if x not in df_sub_vertROI['subject'].unique()]
        vert_not_meg = [x for x in df_sub_vertROI['subject'].unique() if x not in df_sub_meg['subject'].unique()]
        
        bh_not_list = [x for x in df_sub_bh['sub_code'].unique() if x not in sub_list_all]
        meg_not_list = [x for x in df_sub_meg['subject'].unique() if x not in sub_list_all]
        vert_not_list = [x for x in df_sub_vertROI['subject'].unique() if x not in sub_list_all]

        print('final sub number',len(sub_list_all))
        print('behavior remove', len(beh_remove),beh_remove)
        print('meg      remove', len(meg_remove),meg_remove)
        print('vertROI  remove', len(vert_remove),vert_remove)

        print('participant in bh but not in final list', len(bh_not_list),bh_not_list)
        print('participant in meg but not in final list', len(meg_not_list),meg_not_list)
        print('participant in vertROI but not in final list', len(vert_not_list),vert_not_list)
        
        print('participant in bh but not in meg', len(bh_not_meg),bh_not_meg)
        print('participant in meg but not in bh', len(meg_not_bh),meg_not_bh)
        print('participant in bh but not in vertROI', len(bh_not_vert),bh_not_vert)
        print('participant in vertROI but not in bh', len(vert_not_bh),vert_not_bh)
        print('participant in meg but not in vertROI', len(meg_not_vert),meg_not_vert)
        print('participant in vertROI but not in meg', len(vert_not_meg),vert_not_meg)
        

        
        print('behavior', df_sub_bh['sub_code'].nunique())
        print('meg     ', df_sub_meg['subject'].nunique())
        print('vertROI ', df_sub_vertROI['subject'].nunique())

        print('finial sub number',len(sub_list_all))

    elif 'sub_list_names' in param.keys():
        print('--- more than one sub_list')
        sub_list_all = []
        for anai, sub_list_name in enumerate(sub_list_names):

            # get subject based on behavior
            df_groupcheck = df_checksub_bh.copy()
            
            if 'corresponding_sub_check_list_logic' in param.keys():
                print(anai +1 ,'----- 1 behavior condition',corresponding_sub_check_list)
                if corresponding_sub_check_list_logic[anai] == "or":
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_list[anai] ),axis = 1)
                    
                elif corresponding_sub_check_list_logic[anai] == "and":  
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_list[anai] ),axis = 1)
                else:
                    ValueError
                df_sub_bh = df_groupcheck[df_groupcheck['condi_true']==True]
                
            elif 'corresponding_sub_check_list_logics' in param.keys():   
                print(anai +1 ,'-----multiple behavior condition',corresponding_sub_check_lists)

                for idx,logic_cond in enumerate(corresponding_sub_check_list_logics[anai]):
                    
                    if logic_cond == "or":
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        not all(x[col]==False for col in corresponding_sub_check_lists[anai][idx] ),axis = 1)
                    if logic_cond == "and":  
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                            all(x[col]==True for col in corresponding_sub_check_lists[anai][idx]  ),axis = 1)
                df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                    any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
                df_sub_bh = df_groupcheck[df_groupcheck['condi_true']==True]
            else :
                print(anai +1 ,'-----no behavior condition max',len(max_sub))
                df_sub_bh = df_checksub_bh[df_checksub_bh['allcondi_false']!=True]
            print(anai +1 ,'-----after behavior sub number',len(df_sub_bh)) 

            df_groupcheck = df_checksub_meg.copy()
            if 'corresponding_sub_check_meg_list_logic' in param.keys():
                print(anai +1 ,'----- 1 meg condition',corresponding_sub_check_meg_list)

                if corresponding_sub_check_meg_list_logic[anai] == "or":
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_meg_list[anai] ),axis = 1)
                    
                elif corresponding_sub_check_meg_list_logic[anai] == "and":  
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_meg_list[anai] ),axis = 1)
                else:
                    ValueError
                df_sub_meg = df_groupcheck[df_groupcheck['condi_true']==True]
                
            elif 'corresponding_sub_check_meg_list_logics' in param.keys():    
                print(anai +1 ,'-----multiple meg condition',corresponding_sub_check_meg_lists)

                for idx,logic_cond in enumerate(corresponding_sub_check_meg_list_logics[anai]):
                    if logic_cond == "or":
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        not all(x[col]==False for col in corresponding_sub_check_meg_lists[anai][idx] ),axis = 1)
                    if logic_cond == "and":  
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                            all(x[col]==True for col in corresponding_sub_check_meg_lists[anai][idx]  ),axis = 1)
                df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                    any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
                df_sub_meg = df_groupcheck[df_groupcheck['condi_true']==True]
                
            else :
                print(anai +1 ,'-----no meg condition max',df_checksub_meg['subject'].nunique())
                df_sub_meg = df_checksub_meg
            print(anai +1 ,'-----after MEG sub number',len(df_sub_meg)) 

            df_groupcheck = df_checksub_vertROI.copy()
            if 'corresponding_sub_check_vertROI_list_logic' in param.keys():
                print(anai +1 ,'----- 1 vertROI condition',corresponding_sub_check_vertROI_list)

                if corresponding_sub_check_vertROI_list_logic == "or":
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                    not all(x[col]==False for col in corresponding_sub_check_vertROI_list ),axis = 1)
                    
                elif corresponding_sub_check_vertROI_list_logic == "and":  
                    df_groupcheck ['condi_true'] = df_groupcheck.apply(lambda x: 
                        all(x[col]==True for col in corresponding_sub_check_vertROI_list ),axis = 1)
                else:
                    ValueError

                df_sub_vertROI = df_groupcheck[df_groupcheck['condi_true']==True]
            elif 'corresponding_sub_check_vertROI_list_logics' in param.keys():    
                print(anai +1 ,'-----multiple vertROI condition',corresponding_sub_check_vertROI_lists)

                for idx,logic_cond in enumerate(corresponding_sub_check_vertROI_list_logics):
                    if logic_cond == "or":
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                        not all(x[col]==False for col in corresponding_sub_check_vertROI_lists[idx] ),axis = 1)
                    if logic_cond == "and":  
                        df_groupcheck [f'condi_true_{idx}'] = df_groupcheck.apply(lambda x: 
                            all(x[col]==True for col in corresponding_sub_check_vertROI_lists[idx]  ),axis = 1)
                df_groupcheck['condi_true'] =   df_groupcheck.apply(lambda x: 
                    any(x[col]==True for col in df_groupcheck.columns if 'condi_true_' in col),axis = 1)  
                df_sub_vertROI = df_groupcheck[df_groupcheck['condi_true']==True]
                
            else :
                print(anai +1 ,'-----no vertROI condition max',df_checksub_vertROI['subject'].nunique())
                df_sub_vertROI = df_checksub_vertROI
            print(anai +1 ,'-----after vertROI sub number',len(df_sub_vertROI)) 


            sub_list_all_one = list(np.sort(
            [x for x in df_sub_bh['sub_code'].unique() if (x in df_sub_meg['subject'].unique()) and (x in df_sub_vertROI['subject'].unique())]   
            ))
            
      
            print(anai +1 ,'behavior', df_sub_bh['sub_code'].nunique())
            print(anai +1 ,'meg     ', df_sub_meg['subject'].nunique())
            print(anai +1 ,'vertROI ', df_sub_vertROI['subject'].nunique())

            print(anai +1 ,'finial sub number',len(sub_list_all_one),'\n')
            sub_list_all.append(sub_list_all_one)
    else:
        sub_list_all = max_sub
        print('finial sub number',len(sub_list_all),'\n')

    return sub_list_all 
# %% get_sub_list
def get_sublist(
    config
):
    '''
    sub_list_all = get_sublist(param)
    '''
    try:
        del param
    except Exception:
        Exception
    configfilename_with_ext = os.path.basename(config)
    config_filename, _ = os.path.splitext(configfilename_with_ext)
    
    # Read the config file:
    with open(config) as f:
        param = json.load(f)
    param = {k: convert_none_to_string(v) for k, v in param.items()}
    print(f'---------- Load sublist from config file: {config_filename} ----------')
    sub_list_all = get_sublist_fromparam(param)
    return sub_list_all
# %% group_get_sublist
def group_get_sublist(param_group,
                      param_sub,
                      group_configfile,
                      individual_configfile,
                      ):
    if "sub_list_name" in param_group.keys():
        sub_list_name = param_group['sub_list_name']
        sub_list = get_sublist(group_configfile)
        print(f'get sublist from group configfile: {sub_list_name}, n={len(sub_list)}')
    elif "sub_list_name" in param_sub.keys():
        sub_list_name = param_sub['sub_list_name']
        sub_list = get_sublist(individual_configfile)
        print(f'get sublist from individual configfile: {sub_list_name}, n={len(sub_list)}')
    else:
        raise ValueError('no sub_list_name in either group or individual configfile')
    return sub_list_name, sub_list
#%% old get_trl_indices
# # select trial for each condition
# def get_trl_indices(df, columns, conditions):
#     """Get indices where all conditions are met in the specified columns."""
#     mask = pd.Series([True] * len(df))
#     for col, cond in zip(columns, conditions):
#         mask &= df[col].isin(cond)
#     idx  = df.index[mask].tolist()
#     # print('extract ',len(idx),' index')
#     return idx
                
#%% get_trl_indices
# select trial for each condition
def get_trl_indices(df, column, condition):
    """
    Get indices where all conditions are met in the specified columns.
    idx = get_trl_indices(
                        df,
                        column, 
                        condition)
    Parameters
    ----------
    df : pd.DataFrame
    column : str | list[str]
        Column name(s) to filter on.
    condition : list | list[list]
        Values or list of values to match in each column.
        If `columns` is a string, `conditions` can be a list of values.

    Returns
    -------
    idx : list[int]
        List of row indices satisfying all conditions.
    """
    # As reulsts use index, make sure index is reset
    df = df.reset_index(drop=True)  # Ensure index is 0, 1, 2, ...
    # normalize input to lists
    if isinstance(column, str):
        column = [column]
        # if user passed a flat list of values, wrap it
        if not isinstance(condition[0], (list, tuple, set)):
            condition = [condition]
    elif not isinstance(column, (list, tuple)):
        raise TypeError("column must be str or list of str")

    mask = pd.Series(True, index=df.index)
    for col, cond in zip(column, condition):
        mask &= df[col].isin(cond)
                
    return df.index[mask].tolist()

#%% funrms funrms(data,d_mean)
funrms = lambda data,d_mean: np.sqrt((data**2).mean(axis=d_mean))



#%% compute_mean_exp2
def compute_mean_exp2(data_condsub,axis_cond, axis_sub, zscore=False, design="within"):
    print("\nAveraging the data and computing CIs...")
    # Average across participants
    data_condmean = np.mean(data_condsub, axis=axis_sub)
    
    # Z-score data by group
    if zscore:
        data_condmean = scipy.stats.zscore(data_condmean, axis=-1)
        data_ci = np.zeros(data_condmean.shape) #fake values 
    else:
        # Compute condidence intervals
        if design == "within": # within-subject design CIs from Cousineau (2005)
            sbj_m = np.mean(data_condsub, axis= axis_cond, keepdims=True)
            grp_m = np.mean(data_condsub, axis=axis_sub, keepdims=True)
            new_sbj = data_condsub - sbj_m + grp_m
            

            # Calculate margin of error
            confidence_level = 0.95
            z_score = np.abs(scipy.stats.norm.ppf((1 - confidence_level) / 2))
            
            data_ci = scipy.stats.sem(new_sbj, axis=axis_sub) * z_score
        elif design == "between":
            # the choice of the distribution used for calculating the critical value is not nomal distribution but t distribution
            # calculate the critical value (t-score) for constructing a confidence interval for the mean 
            # when the sample size is small and the population standard deviation is unknown.
            n = data_condsub.shape[axis_sub]
            data_ci = scipy.stats.sem(data_condsub, axis=axis_sub) * scipy.stats.t.ppf((1 + .95) / 2., n - 1)
    
    return data_condmean , data_ci

#%% convert_to_ci_1D - Convert data and error bars to confidence interval format
def convert_to_ci_1D(data_cound_time, err_cound_time):
    """
    Convert mean data and error bars into confidence interval bounds format.
    
    INPUT:
    ------
    data_cound_time : ndarray
        Mean values for each condition and time point
        Shape: (n_condition, n_times) or (n_times,)
        
    err_cound_time : ndarray
        Error magnitude (e.g., SEM, STD) for each condition and time point
        Shape: (n_condition, n_times) or (n_times,)
        Must match the shape of data_cound_time
    
    OUTPUT:
    -------
    ci_1D_ctb : ndarray
        Confidence interval bounds in format: [lower_bound, upper_bound]
        Shape: (n_condition, n_times, 2)
        - Axis 0: condition index (1 if input was 1D)
        - Axis 1: time point index
        - Axis 2: [0] = lower bound, [1] = upper bound
        
    CALCULATION:
    ------------
    lower_bound = data - err  (mean minus error)
    upper_bound = data + err  (mean plus error)
    
    EXAMPLE:
    --------
    # Single condition case
    data = np.array([1.0, 2.0, 3.0])  # shape: (3,)
    err = np.array([0.1, 0.2, 0.1])   # shape: (3,)
    result = convert_to_ci_1D(data, err)
    # result.shape = (1, 3, 2)
    # result[0, 0, :] = [0.9, 1.1]  # first time point: [lower, upper]
    
    # Multiple conditions case
    data = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)
    err = np.array([[0.1, 0.2], [0.3, 0.1]])   # shape: (2, 2)
    result = convert_to_ci_1D(data, err)
    # result.shape = (2, 2, 2)
    # result[0, 0, :] = [0.9, 1.1]  # condition 0, time 0: [lower, upper]
    # result[1, 1, :] = [3.9, 4.1]  # condition 1, time 1: [lower, upper]
    """
    
    print('-------- Converting error bars to confidence interval format --------')
    print(f'Input data shape: {data_cound_time.shape}')
    print(f'Input error shape: {err_cound_time.shape}')
    
    # ========================================================================
    # STEP 1: Handle 1D input - convert to 2D with single condition
    # ========================================================================
    # WHY: We need consistent 2D format (n_condition, n_times) for processing
    # If input is (n_times,), reshape to (1, n_times) meaning 1 condition
    if data_cound_time.ndim == 1:
        print("→ Detected 1D input, reshaping to 2D with single condition")
        print(f"  Before: {data_cound_time.shape} → After: ", end="")
        
        # Add new axis at the beginning: (n_times,) → (1, n_times)
        data_cound_time = data_cound_time[np.newaxis, :]  
        err_cound_time = err_cound_time[np.newaxis, :]
        
        print(f"{data_cound_time.shape}")
    
    # ========================================================================
    # STEP 2: Validate input shapes match
    # ========================================================================
    if data_cound_time.shape != err_cound_time.shape:
        raise ValueError(
            f"Shape mismatch! data shape {data_cound_time.shape} "
            f"!= error shape {err_cound_time.shape}"
        )
    
    n_conditions, n_times = data_cound_time.shape
    print(f"Processing: {n_conditions} condition(s), {n_times} time point(s)")
    
    # ========================================================================
    # STEP 3: Calculate confidence interval bounds
    # ========================================================================
    # Lower bound = mean - error
    # Upper bound = mean + error
    lower_bounds = data_cound_time - err_cound_time  # Shape: (n_condition, n_times)
    upper_bounds = data_cound_time + err_cound_time  # Shape: (n_condition, n_times)
    
    # ========================================================================
    # STEP 4: Stack lower and upper bounds into 3D array
    # ========================================================================
    # Stack along last axis to create [..., [lower, upper]] structure
    # Result shape: (n_condition, n_times, 2)
    # where [:, :, 0] = lower bounds, [:, :, 1] = upper bounds
    ci_1D_ctb = np.stack((lower_bounds, upper_bounds), axis=-1)
    
    print(f"Output CI shape: {ci_1D_ctb.shape}")
    print("  → ci_1D_ctb[condition_idx, time_idx, 0] = lower bound")
    print("  → ci_1D_ctb[condition_idx, time_idx, 1] = upper bound")
    print('-------- Conversion complete --------\n')
    
    return ci_1D_ctb
# #%% convert_to_ci_1D  modify the variable name
# def convert_to_ci_1D(data_cound_time, err_cound_time):
#     """
#     Convert data_cound_time ( n_condition,n_times,) or (n_times,)
#     and err_cound_time  ( n_condition,n_times,) or (n_times,)
#     into ci_1D_ctb format (n_condition, n_times, 2), 
#     representing lower and upper bounds for each condition at each time point.

#     Parameters:
#     - data_cound_time: ndarray of shape ( n_condition,n_times,) or (n_times,)
#     - err_cound_time: ndarray of shape ( n_condition,n_times,) or (n_times,)

#     Returns:
#     - ci_1D_ctb: ndarray of shape (n_condition, n_times, 2), representing lower and upper bounds
#                 for each condition at each time point.
#     """
#     print('-------- Converting error into to (n_condition, n_times, 2): lower and upper bounds for each condition')
    
#     # If input is one-dimensional, add an extra axis to make it 2D with a single condition
#     # if data_cound_time.ndim == 1:
#     #     print("Converting 1D data and error to ci_1D_ctb format condtion")
#     #     data_cound_time = data_cound_time[:, np.newaxis]  # Shape: (n_times, 1)
#     #     err_cound_time = err_cound_time[:, np.newaxis]    # Shape: (n_times, 1)
#     # If input is one-dimensional, add an extra axis to make it 2D with a single condition
#     if data_cound_time.ndim == 1:
#         print("Converting 1D data and error to ci_1D_ctb format with single condition")
#         data_cound_time = data_cound_time[np.newaxis, :]  # Shape: (1, n_times)
#         err_cound_time = err_cound_time[np.newaxis, :]    # Shape: (1, n_times)
#     # Check that data_cound_time and err_cound_time have the same shape
#     if data_cound_time.shape != err_cound_time.shape:
#         raise ValueError("data_cound_time and err_cound_time must have the same shape.")

#     # Calculate lower and upper bounds
#     lower_bounds = data_cound_time - err_cound_time  # Shape: ( n_condition,n_times,)
#     upper_bounds = data_cound_time + err_cound_time  # Shape: ( n_condition,n_times,)

#     # Generate ci_1D_ctb in 3D format: (n_condition, n_times, 2)
#     ci_1D_ctb = np.stack((lower_bounds, upper_bounds), axis=-1)  # Shape: (n_condition, n_times, 2)

#     return ci_1D_ctb
#%% apply_lowpass_filter - Apply low-pass filter to multi-dimensional time series data
def apply_lowpass_filter(cond_t, fs, filt_type="butter", order=6, cutoff=30, flag_verbose=False):
    """
    Apply a low-pass Butterworth filter to 1D, 2D, or 3D time series data.
    
    INPUT:
    ------
    cond_t : np.ndarray
        Time series data with TIME always along the LAST axis (1D) or axis=1 (2D/3D)
        
        Supported shapes:
        - 1D: (n_times,)
          Single time series
          
        - 2D: (n_conditions, n_times) or (n_channels, n_times)
          Multiple time series, each row is filtered independently
          Example: ERP data with different conditions
          
        - 3D: (n_conditions, n_times, n_channels) 
          Multiple time series across two other dimensions
          Example: Source-space data with conditions x time x sources
          TIME must be on axis=1 (middle axis)
    
    fs : float
        Sampling frequency in Hz (e.g., 1000 for 1000 Hz = 1ms resolution)
        
    filt_type : str, optional (default="butter")
        Filter type passed to lowpass_filter function
        Options: "butter" (Butterworth), "fir", etc.
        
    order : int, optional (default=6)
        Filter order - higher order = sharper cutoff but more ringing
        
    cutoff : float, optional (default=30.0)
        Low-pass cutoff frequency in Hz
        Must be < Nyquist frequency (fs/2)
        Example: cutoff=30 removes frequencies > 30 Hz
    
    OUTPUT:
    -------
    cond_t_ft : np.ndarray
        Filtered time series data with SAME shape as input
        All non-time dimensions preserved
    
    RAISES:
    -------
    ValueError
        - If input is not 1D, 2D, or 3D
        - If cutoff frequency is invalid (≤0 or ≥ Nyquist)
    RuntimeError
        - If output shape doesn't match input shape (internal error)
    
    EXAMPLES:
    ---------
    # 1D: Single time series
    data_1d = np.random.randn(1000)  # 1000 time points
    filtered = apply_lowpass_filter(data_1d, fs=1000, cutoff=30)
    # Output shape: (1000,)
    
    # 2D: Multiple conditions
    data_2d = np.random.randn(5, 1000)  # 5 conditions x 1000 time points
    filtered = apply_lowpass_filter(data_2d, fs=1000, cutoff=30)
    # Output shape: (5, 1000) - each condition filtered independently
    
    # 3D: Conditions x Time x Channels
    data_3d = np.random.randn(5, 1000, 64)  # 5 cond x 1000 times x 64 channels
    filtered = apply_lowpass_filter(data_3d, fs=1000, cutoff=30)
    # Output shape: (5, 1000, 64) - each cond x channel combo filtered
    
    NOTES:
    ------
    - The function assumes TIME is on axis=1 for 2D/3D inputs
    - Filtering is applied independently to each "time series slice"
    - Edge artifacts may occur at boundaries (inherent to filtering)
    """
    
    # ========================================================================
    # STEP 1: Input validation
    # ========================================================================
    cond_t = np.asarray(cond_t)
    
    if cond_t.ndim not in (1, 2, 3):
        raise ValueError(
            f"Input must be 1D, 2D, or 3D. Got {cond_t.ndim}D with shape {cond_t.shape}"
        )

    # Check cutoff frequency is valid (must be between 0 and Nyquist)
    nyq = fs / 2.0  # Nyquist frequency = half of sampling rate
    if not (0 < cutoff < nyq):
        raise ValueError(
            f"Cutoff frequency must be between 0 and Nyquist frequency.\n"
            f"  Nyquist (fs/2) = {nyq:.3f} Hz\n"
            f"  Requested cutoff = {cutoff} Hz\n"
            f"  Valid range: (0, {nyq:.3f})"
        )
    print(f"Applying {cutoff} Hz low-pass filter to {cond_t.ndim}D data (shape: {cond_t.shape})")
    
    # ========================================================================
    # STEP 2: Apply filtering based on dimensionality
    # ========================================================================
    
    # --- 1D CASE: Single time series ---
    # Input: (n_times,)
    # Action: Filter the entire array directly
    if cond_t.ndim == 1:
        if flag_verbose:
            print(f"  → Filtering single time series ({cond_t.shape[0]} samples)")
        cond_t_ft = lowpass_filter(
                cond_t, fs=fs, filt_type=filt_type, order=order, cutoff=cutoff
            )

    # --- 2D CASE: Multiple time series (rows) ---
    # Input: (n_rows, n_times)
    # Action: Filter each row independently
    # Example: 5 conditions x 1000 time points → filter each of 5 time series
    elif cond_t.ndim == 2:
        n_rows, n_times = cond_t.shape
        if flag_verbose:
            print(f"  → Filtering {n_rows} time series (each with {n_times} samples)")
        
        cond_t_ft = np.stack(
            [lowpass_filter(row, fs=fs, filt_type=filt_type, order=order, cutoff=cutoff)
             for row in cond_t],
            axis=0
        )

    # --- 3D CASE: Multiple time series across two dimensions ---
    # Input: (n_dim0, n_times, n_dim2)
    # Action: Filter along axis=1 (time) for each combination of [dim0, dim2]
    # Example: 5 conditions x 1000 times x 64 channels
    #          → filter 5×64=320 separate time series
    else:  # cond_t.ndim == 3
        n0, n_times, n2 = cond_t.shape
        n_total_series = n0 * n2
        if flag_verbose:
            print(f"  → Filtering {n_total_series} time series "
              f"({n0}x{n2} combinations, each with {n_times} samples)")
        
        # Pre-allocate output array for efficiency
        out = np.empty_like(cond_t, dtype=float)
        
        # Loop over all combinations of first and third dimensions
        # Filter each (n_times,) slice independently
        for i in range(n0):
            for k in range(n2):
                out[i, :, k] = lowpass_filter(
                    cond_t[i, :, k],  # Extract 1D time series
                    fs=fs, filt_type=filt_type, order=order, cutoff=cutoff
                )
        cond_t_ft = out

    # ========================================================================
    # STEP 3: Validate output shape matches input
    # ========================================================================
    if cond_t_ft.shape != cond_t.shape:
        raise RuntimeError(
            f"Shape mismatch after filtering!\n"
            f"  Input shape:  {cond_t.shape}\n"
            f"  Output shape: {cond_t_ft.shape}\n"
            f"  This indicates an internal error in the filtering logic."
        )
    if flag_verbose:
        print(f"  ✓ Filtering complete. Output shape: {cond_t_ft.shape}")
    return cond_t_ft
#%% old apply_lowpass_filter
# def apply_lowpass_filter(cond_t, fs, filt_type="butter", order=6, cutoff=30.):
#     """
#     Apply a low-pass filter to 1D, 2D, or 3D data.

#     Assumptions (kept same as your current behavior):
#     - 1D: cond_t.shape == (n_times,)  -> filter along the only axis
#     - 2D: cond_t.shape == (n_rows, n_times)  -> filter each row (time is axis=1)
#     - 3D: cond_t.shape == (n0, n_times, n2) -> filter along axis=1 for each [i, :, k]

#     Parameters
#     ----------
#     cond_t : np.ndarray
#         1D, 2D, or 3D array.
#     fs : float
#         Sampling rate in Hz.
#     filt_type : str
#         Filter type for `lowpass_filter`.
#     order : int
#         Filter order.
#     cutoff : float
#         Cutoff frequency (Hz).

#     Returns
#     -------
#     np.ndarray
#         Filtered array with the same shape as `cond_t`.
#     """
#     cond_t = np.asarray(cond_t)
#     if cond_t.ndim not in (1, 2, 3):
#         raise ValueError("Input array cond_t must be 1D, 2D, or 3D.")

#     # Basic sanity check on cutoff
#     nyq = fs / 2.0
#     if not (0 < cutoff < nyq):
#         raise ValueError(f"cutoff must be between 0 and Nyquist ({nyq:.3g} Hz). Got {cutoff} Hz.")

#     # 1D: direct call
#     if cond_t.ndim == 1:
#         cond_t_ft = lowpass_filter(cond_t, fs=fs, filt_type=filt_type, order=order, cutoff=cutoff)

#     # 2D: filter each row (time axis = 1)
#     elif cond_t.ndim == 2:
#         cond_t_ft = np.stack(
#             [lowpass_filter(row, fs=fs, filt_type=filt_type, order=order, cutoff=cutoff)
#              for row in cond_t],
#             axis=0
#         )

#     # 3D: filter along axis=1 for each (i, :, k), preserve original shape
#     else:  # cond_t.ndim == 3
#         n0, nT, n2 = cond_t.shape
#         out = np.empty_like(cond_t, dtype=float)
#         for i in range(n0):
#             for k in range(n2):
#                 out[i, :, k] = lowpass_filter(
#                     cond_t[i, :, k], fs=fs, filt_type=filt_type, order=order, cutoff=cutoff
#                 )
#         cond_t_ft = out
        

#     # Final shape check
#     if cond_t_ft.shape != cond_t.shape:
#         raise RuntimeError(f"Filtered output shape {cond_t_ft.shape} does not match input {cond_t.shape}.")

#     return cond_t_ft
# %% bootstrap_ci_1D
def bootstrap_ci_1D(X_1D,
    n_loop =10000,
    alpha = 0.05, #
    flag_method_1= 0,
    flag_method_2= 1,
):
    mean_estimates = []
    for _ in range(n_loop):
        
        if flag_method_1:
            re_sample_idx = np.random.randint(0, 
                                            len(X_1D), 
                                            X_1D.shape)
            mean_estimates.append(np.mean(X_1D[re_sample_idx]))
            
        if flag_method_2:
        # Generate a bootstrap sample by sampling with replacement from the data
            bootstrap_sample = np.random.choice(X_1D, 
                                                size=len(X_1D), 
                                                replace=True)
            mean_estimates.append(np.mean(bootstrap_sample))
            
    sorted_estimates = np.sort(np.array(mean_estimates))
    # print(sorted_estimates)
    conf_interval = [sorted_estimates[int(alpha/2 * n_loop)], 
                     sorted_estimates[int((1 - alpha/2)* n_loop)]]
    return conf_interval
# %% compute_mean_bootstrap_1d
def compute_mean_bootstrap_1d(
                    data_sub_t, 
                    alpha = 0.05, 
                    n_repeat = 10000,
                    flag_test = 0):
    '''
    use bootstrap to get confidence interval
    '''
    if flag_test:
        n_repeat = 100
        


    n_subjects, n_time_points = data_sub_t.shape
    data_t_mean = np.mean        (data_sub_t,axis=0)
    data_t_sem  = scipy.stats.sem(data_sub_t,axis=0)
    data_t_ci = np.zeros((n_time_points, 2))  # Initialize array for confidence intervals
    
    for t in range(n_time_points):
        t_allsub = np.array(data_sub_t[:, t])
        # print(t_allsub)
        conf_interval   = bootstrap_ci_1D(t_allsub,
                                            n_loop =n_repeat,
                                            alpha = alpha )
        # print(conf_interval)
        assert conf_interval[0] != conf_interval[1] 
        data_t_ci[t, 0] = conf_interval[0] # Lower bound
        data_t_ci[t, 1] = conf_interval[1]  # Upper bound
    return data_t_mean, data_t_ci  
#%% fun: get_file_name
def get_file_name(configfile: str) -> str:
    """
    Extract the analysis name (filename without extension) from a config file path.

    Parameters
    ----------
    configfile : str
        Path to the configuration file, e.g. "/path/to/my_config.json".

    Returns
    -------
    analysis_name : str
        The filename without its extension, e.g. "my_config".
    """
    
    config_filename_with_ext = os.path.basename(configfile)
    analysis_name, _ = os.path.splitext(config_filename_with_ext)
    return analysis_name
#%% fun: turn_string_to_num
def turn_string_to_num(s):
    total_sum = 0  # Initialize the sum to 0
    
    for char in s:
        if char.isalpha():  # Check if the character is a letter
            # Convert letter to its position in the alphabet (A=1, B=2, ..., Z=26)
            num = ord(char.lower()) - ord('a') + 1
        elif char.isdigit():  # Check if the character is a digit
            num = int(char)
        else:
            continue  # Ignore non-alphabetic, non-numeric characters
        
        # Add the current number to the total sum
        total_sum += num
    print('input string:',s,' converted to number:',total_sum)
    
    return total_sum
#%% fun: create_save_label_from_vert
def create_save_label_from_vert(
    stc,
    src,
    sig_vidx_lr,
    labelaveDir,
    groupdata_name,
    subdata_name,
    subject,
    flag_get_label = 1,    
    
    ):

    sig_vidx_l = [x             for x in sig_vidx_lr if x  < len(stc.vertices[0])]
    sig_vidx_r = [x-len(stc.vertices[0]) for x in sig_vidx_lr if x  >=len(stc.vertices[0])]
    assert set(stc.vertices[0]) == set(src[0]['vertno'])
    
    sig_virt_l = [x for idx,x in enumerate(stc.vertices[0]) if idx in sig_vidx_l]
    sig_virt_r = [x for idx,x in enumerate(stc.vertices[1]) if idx in sig_vidx_r]
    
    flag_sigl = len(sig_virt_l) > 0
    flag_sigr = len(sig_virt_r) > 0

    # Create labels for both hemispheres
    if flag_get_label:
        label_lr = []
        if flag_sigl:
            label_l = mne.Label(
                vertices= sig_virt_l, 
                hemi    = 'lh', 
                name    = f"{groupdata_name.replace(' ','_')}-lh", 
                subject = subject)
            
            l_label_filename = os.path.join(labelaveDir,
                                            f"{subdata_name}-lh.label").replace(' ','_')
            
            label_lr.append(label_l)
            label_l.save(l_label_filename)
            print('save left label\n', l_label_filename)
            
        if flag_sigr:
            label_r = mne.Label(
                vertices= sig_virt_r, 
                hemi    = 'rh', 
                name    = f"{groupdata_name.replace(' ','_')}-rh", 
                subject = subject)
            r_label_filename = os.path.join(labelaveDir, 
                                            f"{subdata_name}-rh.label").replace(' ','_')
            
            label_lr.append(label_r)
            label_r.save(r_label_filename)
            print('save right label\n', r_label_filename)
            
        # if label_lr:
        #     label_filename = os.path.join(labelaveDir,
        #                                 f"{subdata_name}.label").replace(' ','_')
        #     label = np.sum([x for x in label_lr])
        #     label.save(label_filename)
        #     print('save label\n', label_filename)
            
    return flag_sigl,flag_sigr,sig_vidx_l,sig_vidx_r
        
#%% fun: welch_df
def welch_df(tltv1, tltv2):
    """
    Calculate the degrees of freedom for Welch's t-test using data matrices.

    Parameters:
    tltv1 : numpy.ndarray
        Data for the first group (2D matrix where each row is a trial, and each column is a measurement).
    tltv2 : numpy.ndarray
        Data for the second group (2D matrix where each row is a trial, and each column is a measurement).

    Returns:
    float
        Degrees of freedom for Welch's t-test.
    """
    # Calculate sample sizes
    n1 = tltv1.shape[0]
    n2 = tltv2.shape[0]
                
    # Calculate variance across trials for each time point and vertex
    # "Delta Degrees of Freedom." 1 calculates the sample variance
    # By default, np.var uses ddof=0, which calculates the population variance.
    var1_tv = np.var(tltv1, axis=0, ddof=1)  # Variance for tltv1 (time x vertices)
    var2_tv = np.var(tltv2, axis=0, ddof=1)  # Variance for tltv2 (time x vertices)

    # Average the variances across time points and vertices
    # mean of all elements in the array
    var1_mean = var1_tv.mean()  # Mean variance across time and vertices for tltv1
    var2_mean = var2_tv.mean()  # Mean variance across time and vertices for tltv2

    # Calculate Welch-Satterthwaite degrees of freedom
    numerator = (var1_mean/n1 + var2_mean/n2) ** 2
    denominator = ((var1_mean/n1) ** 2) / (n1 - 1) + ((var2_mean/n2) ** 2) / (n2 - 1)
    df = numerator / denominator
    return df
#%% ttest_ind_nop
def ttest_ind_nop(data1, data2):
    #tvals, _ = ttest_ind(*args)
    tvals, _ = scipy.stats.ttest_ind(data1, data2, equal_var=False)
    return tvals
#%%  ged_stcs_for_GEDpref
from EXP1_help_functions import crop_stcs
def seperate_stcs_for_GED(
        subject,
        stcs_epoch,
        stcs_metadata ,
        cond_GEDpref_select_columns,
        cond_GEDpref_content_list,
        cond_GEDpref_tw,
        cond_noGEDpref_tw,
        cond_GEDirre_select_columns = None, # for visual responseive GED, there is no irrelevant stimulus
        cond_GEDirre_content_list = None, # for visual responseive GED, there is no irrelevant stimulus
        **kwargs):
    
        
        # cond_GEDirre_select_columns=None,
        # cond_GEDirre_content_list=None,
    if kwargs:
        print('------ Start seperate_stcs_for_GED for subject:', subject)
        print(f"Subject: {subject}")
        print(f"STCs Epoch: {len(stcs_epoch)}")
        print(f"STCs Metadata: {len(stcs_metadata)}")
        print(f"Condition GED Preferred Select Columns: {cond_GEDpref_select_columns}")
        print(f"Condition GED Preferred Content List: {cond_GEDpref_content_list}")
        print(f"Condition GED Preferred Time Window: {cond_GEDpref_tw}")
        print(f"Condition No GED Preferred Time Window: {cond_noGEDpref_tw}")
        
    # cond_GEDirre_select_columns   = kwargs.get('cond_GEDirre_select_columns', None)
    # cond_GEDirre_content_list   = kwargs.get('cond_GEDirre_content_list', None)
    flag_stcs_irrstim = cond_GEDirre_select_columns is not None
    if flag_stcs_irrstim:
        print('--- load stcs_irrstim')
        print(f"Condition GED Irrelevant Select Column: {cond_GEDirre_select_columns}")
        print(f"Condition GED Irrelevant Content List: {cond_GEDirre_content_list}")
    else:
        print('--- No stcs_irrstim')
        
        
    # stcs_epoch   = kwargs.get('stcs_epoch', None)
    # stcs_metadata= kwargs.get('stcs_metadata', None)
    
    # cond_GEDpref_select_columns  = kwargs.get('cond_GEDpref_select_columns', None)
    # cond_GEDpref_content_list   = kwargs.get('cond_GEDpref_content_list', None)
    # cond_GEDpref_tw   = kwargs.get('cond_GEDpref_tw', None)
    # cond_noGEDpref_tw   = kwargs.get('cond_noGEDpref_tw', None)
    
    # cond_GEDirre_select_columns  = kwargs.get('cond_GEDirre_select_columns', None)
    # cond_GEDirre_content_list   = kwargs.get('cond_GEDirre_content_list', None)
    
    times =  stcs_epoch[0].times
    sfreq =  stcs_epoch[0].sfreq
    
    
    trlidx_GEDpref = get_trl_indices(stcs_metadata, 
                                 cond_GEDpref_select_columns, 
                                 cond_GEDpref_content_list)
    print(trlidx_GEDpref)
    trlidx_noGEDpref = [x for x in range(len(stcs_epoch)) if x not in trlidx_GEDpref]
    
    if flag_stcs_irrstim:
        trlidx_GEDirre = get_trl_indices(stcs_metadata, 
                                    cond_GEDirre_select_columns, 
                                    cond_GEDirre_content_list)
        # trlidx_noGEDirre = [x for x in range(len(stcs_epoch)) if x not in trlidx_GEDirre]
        stcs_irrstim   = [x for idx,x in enumerate(stcs_epoch) if idx in trlidx_GEDirre]
        # stcs_noirrstim = [x for x in enumerate(stcs_epoch) if idx in trlidx_noGEDirre]

    # get stcs of category
    stcs_stim      = [x for idx,x in enumerate(stcs_epoch) if idx in trlidx_GEDpref]
    stcs_nostim    = [x for idx,x in enumerate(stcs_epoch) if idx in trlidx_noGEDpref]


    # Select activation (i.e., stimulus presentation) window
    stcs_stim_active = crop_stcs(
                    stcs = stcs_stim, 
                    tmin = cond_GEDpref_tw[0], 
                    tmax = cond_GEDpref_tw[-1])
    stcs_nostim_active = crop_stcs(
                    stcs = stcs_nostim, 
                    tmin = cond_noGEDpref_tw[0], 
                    tmax = cond_noGEDpref_tw[-1])
    if flag_stcs_irrstim:
        stcs_prepare_for_GED = {
            'stcs_stim'   :stcs_stim,   #
            'stcs_nostim' :stcs_nostim,
            'stcs_irrstim':stcs_irrstim,# 'stcs_noirrstim':stcs_noirrstim,
            'stcs_stim_active'   :stcs_stim_active,
            'stcs_nostim_active' :stcs_nostim_active,
            'times': times,        # 'label_vertidx_dict':label_vertidx_dict
            'sfreq' :sfreq,
        }
    else:
        stcs_prepare_for_GED = {
            'stcs_stim'   :stcs_stim,   #
            'stcs_nostim' :stcs_nostim,
            'stcs_stim_active'   :stcs_stim_active,
            'stcs_nostim_active' :stcs_nostim_active,
            'times': times,        # 'label_vertidx_dict':label_vertidx_dict
            'sfreq' :sfreq,
        }
    print('------ finish seperate_stcs_for_GED for subject:', subject)
    return stcs_prepare_for_GED



#%% get_power_from_array_morlet
def get_power_from_array_morlet(
    tlvt,
    sfreq,
    decim_of_array=1,
    **kwargs,  #'power',output,
        ):
    '''
    tlvtpft =get_power_from_array_morlet(
        
    )
    '''
    for key, value in kwargs.items():
        globals()[key] = value
        exec(f"{key} = value") 
    

    if kwargs:
        print('------ Start get_power_from_array ')
        print(f"freqs: {freqs}")
        print(f"n_cycles: {n_cycles}")
        print(f"output: {output}")
    # freqs            = kwargs.get('freqs', None)
    # n_cycles         = kwargs.get('n_cycles', None)
    # time_bandwidth   = kwargs.get('time_bandwidth', None)
    
    # get power: the mean of all epochs
    # if output in ('complex',' 'phase'), 
    # array of shape (n_epochs, n_chans, n_freqs, n_times)
    tlvft =  mne.time_frequency.tfr_array_morlet(
        tlvt, 
        sfreq         = sfreq,
        freqs         = freqs, 
        n_cycles      = n_cycles, 
        use_fft     = True, 
        output      =  output,
        decim  =  decim_of_array,
        n_jobs = -1,
        verbose=True)

    return tlvft
#%% get_power_from_array_multitaper
def get_power_from_array_multitaper(
    tlvt,
    sfreq,
    output,
    freqs,
    n_cycles,
    time_bandwidth,
    decim_of_array=1,
    **kwargs,
        ):
    '''
    tlvtpft =get_power_from_array_multitaper(
        
    )
    '''
    if kwargs:
        print('------ Start get_power_from_array ')
        print(f"freqs: {freqs}")
        print(f"n_cycles: {n_cycles}")
        print(f"time_bandwidth: {time_bandwidth}")
    # freqs            = kwargs.get('freqs', None)
    # n_cycles         = kwargs.get('n_cycles', None)
    # time_bandwidth   = kwargs.get('time_bandwidth', None)
    
    # get power: the mean of all epochs
    # if output in ('complex',' 'phase'), 
    # array of shape (n_epochs, n_chans, n_tapers, n_freqs, n_times)
    tlvtpft =  mne.time_frequency.tfr_array_multitaper(
        tlvt, # n_epochs, n_channels, n_times
        sfreq         = sfreq,
        freqs         = freqs, 
        n_cycles      = n_cycles, 
        time_bandwidth=time_bandwidth,
        use_fft = True, 
        output =  output,
        decim  =  decim_of_array,
        n_jobs = -1,
        verbose=True)

    return tlvtpft
#%% fun: nested_dicts_equal
def nested_dicts_equal(dict1, dict2):
    """Recursively compare nested dictionaries"""
    if dict1.keys() != dict2.keys():
        return False
    
    for key in dict1.keys():
        v1, v2 = dict1[key], dict2[key]
        
        # If both are dictionaries, recurse
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not nested_dicts_equal(v1, v2):
                return False
        # If both are arrays
        elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        # Regular comparison
        else:
            if v1 != v2:
                return False
    
    return True


# %%
