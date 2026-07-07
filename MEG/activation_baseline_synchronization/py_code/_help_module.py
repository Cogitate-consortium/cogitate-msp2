#%% import
# Python feature flag. instead of as real Python objects at runtime, it changes how type hints are stored --- as strings.
from __future__ import annotations

import mne  #  for pick channels by type

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# for show time
from datetime import datetime

# for save plot data
import json 

# for combing all avg methods in different dimensions
from itertools import product

# for function parameter use path
from pathlib import Path 



from _help_functions import (get_trl_indices,
                                 compute_mean_exp2,
                                 convert_to_ci_1D,
                                 apply_lowpass_filter,
                                compute_mean_bootstrap_1d ,
                                 )

# # %% compute_mean_bootstrap_1d
# def compute_mean_bootstrap_1d(
#                     data_sub_t, 
#                     alpha = 0.05, 
#                     n_repeat = 100):
#     '''
#     use bootstrap to get confidence interval
#     '''

#     n_subjects, n_time_points = data_sub_t.shape
#     data_t_mean = np.mean        (data_sub_t,axis=0)
#     data_t_sem  = scipy.stats.sem(data_sub_t,axis=0)
#     data_t_ci = np.zeros((n_time_points, 2))  # Initialize array for confidence intervals
    
#     for t in range(n_time_points):
#         t_allsub = np.array(data_sub_t[:, t])
#         # print(t_allsub)
#         conf_interval   = bootstrap_ci_1D(t_allsub,
#                                             n_loop =n_repeat,
#                                             alpha = alpha )
#         # print(conf_interval)
#         assert conf_interval[0] != conf_interval[1] 
#         data_t_ci[t, 0] = conf_interval[0] # Lower bound
#         data_t_ci[t, 1] = conf_interval[1]  # Upper bound
#     return data_t_mean, data_t_ci  
# for plot
import seaborn as sns
import re # regular expressions module
import matplotlib.pyplot as plt
from cog_plot import (
                      plot_time_series_sigline,
                      )
# for save data
import pickle

# decorator from Python dataclasses module,to automatically create the “boring boilerplate” methods
from dataclasses import dataclass
# type-hinting helpers from Python’s typing module, makes it easy to know what types are expected/returned by functions in this
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd




# for flat the list 
# evaluates strings that represent Python literals (lists, dicts, tuples, numbers, booleans, etc.) 
# turns them into real Python objects
# literal_eval("[1, 2, 3]")     # -> [1, 2, 3]   (list)
from ast import literal_eval


#%% fun: _slug
# deal with string to be used in file name
def _slug(s: str) -> str:
    # Strips whitespace
    s = s.strip().replace(" ", "_")
    # Removes any characters except letters, digits, ., _, and -.
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)


#%% fun: merge_sig_windows
def merge_sig_windows(sig_timepoints, times_use):
    """
    Merges adjacent significant timepoints into continuous windows based on
    contiguity in the original time vector.

    Two timepoints are considered contiguous if they are consecutive
    in the master time vector `times_use`.

    Parameters
    ----------
    sig_timepoints : list[float] or np.ndarray
        List/array of significant timepoints (seconds).
        Unsorted and duplicate values are allowed.
    times_use : np.ndarray
        Full 1D time vector (sorted, seconds).

    Returns
    -------
    list[list[float]]
        List of merged windows [t_start, t_end].
    """
    if not sig_timepoints:
        return []

    # Ensure timepoints are unique and sorted for efficient processing
    sig_timepoints = sorted(list(set(sig_timepoints)))

    windows = []
    start_time = sig_timepoints[0]

    # Iterate through timepoints to find discontinuities in the master time vector
    #  except the last one, except the first one
    for prev_t, curr_t in zip(sig_timepoints[:-1], sig_timepoints[1:]):
        # Use binary search to find the indices of the time points
        prev_idx = np.searchsorted(times_use, prev_t, side='left')
        curr_idx = np.searchsorted(times_use, curr_t, side='left')

        # Check for a gap > 1 index in the original time vector
        if curr_idx - prev_idx > 1:
            windows.append([start_time, prev_t])
            start_time = curr_t

    # Append the last window
    windows.append([start_time, sig_timepoints[-1]])
    
    return windows
#%% fun plot: plot_and_save_trial_distribution
def plot_and_save_trial_distribution(
    df,
    cond1: str,
    cond2: str,
    title: str,
    jpg_dir: str | Path, 
    pdf_dir: str | Path, 
    dpi: int = 300,
) -> None:
    """
    Plot the distribution of trial numbers for two conditions and save as PDF & JPG.
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'subject', cond1, and cond2 columns.
    cond1 : str
        Column name for the first condition (default: 'StimSeen').
    cond2 : str
        Column name for the second condition (default: 'StimUnseen').
    title : str
        Title for the plot (used also in filenames).
    jpg_dir : str | Path
        Directory to save JPG.
    pdf_dir : str | Path
        Directory to save PDF.
    dpi : int
        Resolution of saved figures.
    """
    # Melt to long format
    df_long = df.melt(
        id_vars="subject",
        value_vars=[cond1, cond2],
        var_name="Condition",
        value_name="TrialCount",
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        data=df_long,
        x="TrialCount",
        hue="Condition",
        kde=True,
        element="step",
        bins=20,
        palette="Set2",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Trials", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # ax.legend(title="Condition")
    plt.tight_layout()

    # Prepare save paths
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)

    stem = _slug(title)
    print(f"Saving trial distribution plot as {stem}.jpg",
          "in ", jpg_dir,)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)

#%% fun plot: plot_and_save_trial_distribution
def plot_and_save_trial_distribution_multiple(
    df,
    conditions: list,
    title: str,
    jpg_dir: str | Path, 
    pdf_dir: str | Path, 
    dpi: int = 300,
) -> None:
    """
    Plot the distribution of trial numbers for two conditions and save as PDF & JPG.
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'subject', cond1, and cond2 columns.
    cond1 : str
        Column name for the first condition (default: 'StimSeen').
    cond2 : str
        Column name for the second condition (default: 'StimUnseen').
    title : str
        Title for the plot (used also in filenames).
    jpg_dir : str | Path
        Directory to save JPG.
    pdf_dir : str | Path
        Directory to save PDF.
    dpi : int
        Resolution of saved figures.
    """
    assert all([x in df.columns for x in conditions +["subject"]  ]), "Some conditions not in DataFrame."
    
    # Melt to long format
    df_long = df.melt(
        id_vars="subject",
        value_vars=conditions,
        var_name="Condition",
        value_name="TrialCount",
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        data=df_long,
        x="TrialCount",
        hue="Condition",
        kde=True,
        element="step",
        bins=20,
        palette="Set2",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Trials", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # ax.legend(title="Condition")
    plt.tight_layout()

    # Prepare save paths
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)

    stem = _slug(title)
    print(f"Saving trial distribution plot as {stem}.jpg",
          "in ", jpg_dir,)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)


#%% fun: _ensure_list
def _ensure_list(v):
    # turn string like "[1, 2, 3]" into a real list; leave lists as-is
    if isinstance(v, str):
        try:
            return literal_eval(v)
        except Exception:
            return [v]
    return v
#%% fun: _flatten_list
def _flatten_list(obj):
    '''
    goes through any nested structure (lists of lists, arrays, etc.),
    extracts all the individual scalar elements, 
    skips NaNs, 
    and yields them one by one.

    data = [[1, [2, 3]], [4]]
    list(chain.from_iterable(data))
    --- only work for one nesting level deep
    '''
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        for x in obj:
            # to go down any depth (e.g. lists inside lists)
            yield from _flatten_list(x)
    elif pd.isna(obj):
        return
    else:
        #if obj is a scalar value (like an int, str, or float) direct return
        yield obj
#%% fun: unique_in_order
def unique_in_order(it):
    '''
    goes through an iterable (like a list) and returns a new list containing
    each unique element once, in the order it first appeared.
    '''
    seen, out = set(), []
    for x in it:
        if x not in seen:
            seen.add(x)
            out.append(int(x))  # drop int() if your indices aren’t ints
    return out
#%% fun: _scale_by_chan_type
def _scale_by_chan_type(data, chan_type):
    """
    Scale MEG data from SI units to plotting units using per-channel types.
    Magnetometers: internally stored in Tesla (T) → displayed in fT (x 1e15).
    Planar gradiometers: internally in T/m → displayed in fT/cm (x 1e13).
    Parameters
    ----------
    data : np.ndarray
        Data in SI units (any shape, all channels same type).
    chan_type : str
        Channel type keyword: 'mag', 'grad', 'eeg', ...

    Returns
    -------
    scaled : np.ndarray
        Data scaled to display units (mag → fT, grad → fT/cm).
    units : list[str]
        Unit label per channel.
    factors : np.ndarray
        Applied scaling factor per channel (shape: [n_channels]).
    """
    # SI → display scaling
    _DISPLAY_SCALE = {
        'mag':  dict(factor=1e15, unit='fT'),     # Tesla → fT
        'grad': dict(factor=1e13, unit='fT/cm'),  # T/m → fT/cm
    }
    scale = _DISPLAY_SCALE.get(chan_type, dict(factor=1.0, unit='AU'))
    factor, unit = scale['factor'], scale['unit']
    scaled = data * factor
    print(f"Scaling {chan_type} data by {factor:.1e} to {unit}")
    return scaled, unit, factor

#%% fun plot: _plot_subject_subplots
def _plot_subject_subplots(condsubt_use, 
                           times_use, 
                           title,
                           cond1_content_name,
                            cond2_content_name,
                          n_cols=6, 
                          palette="dark", 
                          linewidth=1,
                          y_suptitle=1.01,
                              vlines=None,
                            vlines_colors=None,
                            vlines_linestyle=None,
                            vlines_linewidth=None,):
    """
    Plot each participant's time series for two conditions in separate subplots.

    Parameters
    ----------
    condsubt : np.ndarray
        Array of shape (num_conditions, num_subjects, num_time_points).
    times_use : np.ndarray
        1D array of length num_time_points, the x-axis values.
    cond1_content_name : str, optional
        Label for the first condition (default: "Cond1").
    cond2_content_name : str, optional
        Label for the second condition (default: "Cond2").
    n_cols : int, optional
        Number of subplot columns (default: 5).
    palette : str or list, optional
        Seaborn palette for conditions (default: "dark").
    linewidth : float, optional
        Width of plotted lines (default: 2).
    """
    num_conditions, num_subjects, num_time_points = condsubt_use.shape
    print(f"Plotting {num_subjects} subjects with {num_conditions} conditions {num_time_points} each.")

    # Map condition index → name
    cond_names = {0: cond1_content_name, 1: cond2_content_name}

    # Build df_plot
    records = []
    for cond in range(num_conditions):
        for subj in range(num_subjects):
            records.append(pd.DataFrame({
                "time": times_use,
                "value": condsubt_use[cond, subj],
                "condition": cond_names[cond],
                "subject": f"Subj{subj+1}"
            }))
    df_plot = pd.concat(records, ignore_index=True)

    # Subplot grid
    n_rows = int(np.ceil(num_subjects / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2 * n_cols, 1.5 * n_rows),
        sharex=True, sharey=True
    )

    def _add_line(lines, colors, linestyles, linewidths, line_type, 
                  default_color='k', default_linestyle='-', default_linewidth=1):
        """Helper function to add lines of a given type (vertical or horizontal)."""
        if lines is not None:
            lines = [lines] if not isinstance(lines, (list, tuple)) else lines
            for idx, line in enumerate(lines):
                color = colors[idx] if colors and idx < len(colors) else default_color
                # linestyle = linestyles[idx] if linestyles and line !=0 else default_linestyle
                # linewidth = linewidths[idx] if linewidths and line !=0 else default_linewidth
                # for line = 0 it could also be not default but be what is set
                linestyle = linestyles[idx] if linestyles  else default_linestyle
                linewidth = linewidths[idx] if linewidths  else default_linewidth
                line_func = ax.axvline if line_type == 'v' else ax.axhline
                line_func(line, color=color, linestyle=linestyle, linewidth=linewidth)


    for subj_idx in range(num_subjects):
        ax = axes.flat[subj_idx]
        sns.lineplot(
            data=df_plot[df_plot["subject"] == f"Subj{subj_idx+1}"],
            x="time", y="value",
            style="condition", hue="condition",
            palette=palette, linewidth=linewidth,
            ax=ax, legend=False
        )
        ax.set_title(f"Subj{subj_idx+1}")
        # ax.axvline(0, color="k", linestyle="--", lw=1)

        # Add vertical and horizontal lines
        _add_line(vlines, 
                  vlines_colors, 
                  vlines_linestyle, 
                  vlines_linewidth, 'v')

    
    
    # Remove empty panels
    for j in range(num_subjects, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(title, y=y_suptitle)
    plt.tight_layout()
    plt.show()
    return fig, axes
#%% fun: _peak_per_window
def _peak_per_window(y, times, windows, mode="abs"):
    """
    y: (n_times,) curve to search
    times: (n_times,) time axis (s)
    windows: list[(t0, t1)] in seconds
    mode: 'pos' | 'neg' | 'abs'  (max, min, or max |y|)
    returns: list of dict rows
    """
    rows = []
    for wi, (t0, t1) in enumerate(windows):
        i0 = int(np.argmin(np.abs(times - t0)))
        i1 = int(np.argmin(np.abs(times - t1)))
        if i1 <= i0:  # skip bad window
            continue
        seg = y[i0:i1+1]
        if mode == "pos":
            print("get the positive peak")
            ip = int(np.argmax(seg))
        elif mode == "neg":
            print("get the negative peak")
            ip = int(np.argmin(seg))
        elif mode =='abs':
            print("get the absolute peak")
            ip = int(np.argmax(np.abs(seg)))
        else:
            raise ValueError("mode must be 'pos', 'neg', or 'abs'")

        idx = i0 + ip
        rows.append(dict(window_idx=wi, 
                         t0=float(times[i0]), 
                         t1=float(times[i1]),
                         t_peak=float(times[idx]), 
                         amp=float(y[idx]), 
                         i_peak=int(idx)))
    return rows


#%% fun: _validate_and_select
def _validate_and_select(
    sub_dict: dict,
    cond1: str,
    cond2: str,
    ch_names_all: list[str],
    sig_chans_use: Iterable[str],
    times_use: np.ndarray,
    times_use_idx: np.ndarray,
):
    """
    Validate input and select channels of interest.
    Returns
    -------
    participants : list[str]
    ch_sel_idx : list[int]
    ch_sel_names : list[str]
    """
    assert len(times_use) == len(times_use_idx), "times_use and times_use_idx mismatch."
    participants = list(sub_dict.keys())
    if not participants:
        raise ValueError("sub_dict is empty.")

    first = participants[0]
    if cond1 not in sub_dict[first] or cond2 not in sub_dict[first]:
        raise KeyError(f"Missing '{cond1}' or '{cond2}' in participant '{first}'.")

    a0, b0 = sub_dict[first][cond1], sub_dict[first][cond2]
    if a0.shape != b0.shape or a0.ndim != 2:
        raise ValueError(f"Condition arrays must be (n_channels, n_times), got {a0.shape} vs {b0.shape}.")

    if len(ch_names_all) != a0.shape[0]:
        raise AssertionError(
            f"len(ch_names_all)={len(ch_names_all)} does not match data channels={a0.shape[0]}"
        )

    wanted = set(sig_chans_use)
    ch_sel_idx = [i for i, ch in enumerate(ch_names_all) if ch in wanted]
    ch_sel_names = [ch_names_all[i] for i in ch_sel_idx]
    if not ch_sel_idx:
        raise ValueError("No channels left after intersecting with sig_chans_use.")

    return participants, ch_sel_idx, ch_sel_names



#%% fun: _stack_chsubcondt
# ---------- Stack helpers (names == dimension order) ----------
def _stack_chsubcondt(
    sub_dict: dict,
    participants: list[str],
    ch_sel_idx: list[int],
    cond1: str,
    cond2: str,
) -> np.ndarray:
    """
    Build a 4D array with explicit order:
        chsubcondt -> (Ch, Sub, Cond, T)
    """
    per_ch = []
    for ch in ch_sel_idx:
        per_sub = []
        for p in participants:
            x1 = sub_dict[p][cond1][ch, :]
            x2 = sub_dict[p][cond2][ch, :]
            if x1.shape != x2.shape:
                raise ValueError(f"Shape mismatch for participant {p} at channel {ch}.")
            per_sub.append(np.stack([x1, x2], axis=0))  # (Cond=2, T)
        per_ch.append(np.stack(per_sub, axis=0))        # (Sub, Cond, T)
    return np.stack(per_ch, axis=0)                     # (Ch, Sub, Cond, T)

#%% fun: _stack_chsubt_diff
def _stack_chsubt_diff(
    sub_dict: dict,
    participants: list[str],
    ch_sel_idx: list[int],
    cond1: str,
    cond2: str,
) -> np.ndarray:
    """
    Condition difference stack (cond1 - cond2) with order:
        chsubt -> (Ch, Sub, T)
    """
    per_ch = []
    for ch in ch_sel_idx:
        per_sub = []
        for p in participants:
            x1 = sub_dict[p][cond1][ch, :]
            x2 = sub_dict[p][cond2][ch, :]
            per_sub.append(x1 - x2)                     # (T,)
        per_ch.append(np.stack(per_sub, axis=0))        # (Sub, T)
    return np.stack(per_ch, axis=0)                     # (Ch, Sub, T)
#%% fun: _compute_erf_core
def _compute_erf_core(
    sub_dict: dict,
    cond1_name: str,
    cond2_name: str,
    ch_names_all: list[str],
    sig_chans_use: Iterable[str],
    chan_type: str,
    times_use: np.ndarray,
    times_use_idx: np.ndarray,
    reduce: str,  # {"rms", "mean"}
    flag_scale_channel_value: bool,
    want_diff: bool = False,
    erfdiffdata_dir: str | Path | None = None,
    erfdiffdata_stem: str | None = None,
    flag_test: bool = False
) -> dict:
    """
    Compute ERFs for two conditions (and optionally their difference).

    Naming convention == dimension order:
      chsubcondt : (Ch, Sub, Cond, T)
      subcondt   : (Sub, Cond, T)
      condsubt   : (Cond, Sub, T)
      condt      : (Cond, T)
      condt_ci   : (Cond, T, 2)
      chsubt     : (Ch, Sub, T)
      diff_subt  : (Sub, T)
      dift       : (T,)
      dift_ci    : (T, 2)
    """
    # --- validate & select
    participants, ch_sel_idx, ch_sel_names = _validate_and_select(
        sub_dict, 
        cond1_name, 
        cond2_name, 
        ch_names_all, 
        sig_chans_use, 
        times_use, 
        times_use_idx
    )
    print("------------ compute_erf_core -----------")
    print(f"compare conditions '{cond1_name}' vs '{cond2_name}'")
    print(f"Selected {chan_type} {len(ch_sel_idx)} channels for analysis.")
    print(f"use time points from {times_use[0]:.3f}s to {times_use[-1]:.3f}s, total {len(times_use)} points.")
    print(f"reduce: {reduce}")
    print(f"scale channel value: {flag_scale_channel_value}")
    print(f"want difference: {want_diff}")


    # --- stack both conditions -> chsubcondt
    chsubcondt_raw = _stack_chsubcondt(sub_dict, 
                                   participants, 
                                   ch_sel_idx, 
                                   cond1_name, 
                                   cond2_name)
    # --- scale
    if flag_scale_channel_value:
        chsubcondt, unit, factor = _scale_by_chan_type(chsubcondt_raw, chan_type)
    else:
        chsubcondt, unit, factor = chsubcondt_raw, "AU", 1.0

    # --- reduce channels -> subcondt
    if reduce.lower() == "rms":
        subcondt = np.sqrt(np.mean(chsubcondt ** 2, axis=0))  # (Sub, Cond, T)
        ylabel_reduce = "RMS"
    elif reduce.lower() == "mean":
        subcondt = chsubcondt.mean(axis=0)                    # (Sub, Cond, T)
        ylabel_reduce = "Mean"
    else:
        raise ValueError("reduce must be 'rms' or 'mean'.")

    # --- reorder -> condsubt (Cond, Sub, T)
    condsubt= np.transpose(subcondt, (1, 0, 2))

    # # --- scale
    # if flag_scale_channel_value:
    #     condsubt, unit, factor = _scale_by_chan_type(condsubt_raw, chan_type)
    # else:
    #     condsubt, unit, factor = condsubt_raw, "AU", 1.0

    # --- group mean & CI
    condt, condt_ci_raw = compute_mean_exp2(
        condsubt, axis_cond=0, axis_sub=1, zscore=False, design="within"
    )
    condt_ci = convert_to_ci_1D(condt, condt_ci_raw)

    # --- optional difference
    diff_subt, dift, dift_ci, erfdiffdata_path_str = None, None, None, None
    if want_diff:
        #
        # chsubt_diff = _stack_chsubt_diff(sub_dict, 
        #                                  participants, 
        #                                  ch_sel_idx, 
        #                                  cond1_name, 
        #                                  cond2_name)
        # if reduce.lower() == "rms":
        #     diff_subt_raw = np.sqrt(np.mean(chsubt_diff ** 2, axis=0))  # (Sub, T)
        # else:
        #     diff_subt_raw = chsubt_diff.mean(axis=0)                    # (Sub, T)

        # if flag_scale_channel_value:
        #     diff_subt, unit_d, _ = _scale_by_chan_type(diff_subt_raw, chan_type)
        #     assert unit_d == unit, "Unit mismatch between conditions and difference."
        # else:
        #     diff_subt, unit_d = diff_subt_raw, "AU"

        # changed 2511 to get condition avg first and then get cond diff
        diff_subt = condsubt[0] - condsubt[1] # (Sub, T)



        if erfdiffdata_dir and erfdiffdata_stem:
            print(f"use erfdiffdata_dir {erfdiffdata_dir}")
            erfdiffdata_dir = Path(erfdiffdata_dir); 
            erfdiffdata_dir.mkdir(parents=True, exist_ok=True)
            erfdiffdata_path = erfdiffdata_dir / f"{_slug(erfdiffdata_stem)}.pkl"
            if erfdiffdata_path.exists():
                with open(erfdiffdata_path, "rb") as f:
                    dd = pickle.load(f)
                dift, dift_ci = dd["erfdif_y_t"], dd["erfdif_c12bound_t"]
            else:
                dift, dift_ci = compute_mean_bootstrap_1d(diff_subt, flag_test=flag_test)
                with open(erfdiffdata_path, "wb") as f:
                    pickle.dump({"erfdif_y_t": dift, 
                                 "erfdif_c12bound_t": dift_ci}, f)
            erfdiffdata_path_str = str(erfdiffdata_path)
        else:
            dift, dift_ci = compute_mean_bootstrap_1d(diff_subt,flag_test=flag_test)
        print(f"dift: {dift}")

    return dict(
        participants=participants,
        selected_channels=ch_sel_names,
        ylabel_reduce=ylabel_reduce,
        unit=unit,
        factor=factor,
        condsubt=condsubt,   # (Cond, Sub, T)
        condt=condt,         # (Cond, T)
        condt_ci=condt_ci,   # (Cond, T, 2)
        diff_subt=diff_subt, # (Sub, T) or None
        dift=dift,           # (T,)     or None
        dift_ci=dift_ci,     # (T, 2)   or None
        erfdiffdata_path=erfdiffdata_path_str,
    )



#%% fun plot: plot_erf_reduce_and_save 
# ---------------- plot: two conditions (no diff) ---------------- #
def plot_erf_reduce_and_save(
    sub_dict: dict,
    cond1_name: str,
    cond2_name: str,
    ch_names_all: list[str],
    sig_chans_use: set[str] | list[str],
    chan_type: str,
    fs: float,
    times_use: np.ndarray,
    times_use_idx: np.ndarray,
    flag_preset_ylim: bool,
    window_list: list | None,
    colordict: dict,
    flag_plot_each_participant: bool,
    reduce: str,                         # {"rms", "mean"}
    title: str,
    jpg_dir: str | Path,
    pdf_dir: str | Path,
    data_dir: str | Path,
    flag_scale_channel_value: bool,
    # cosmetics
    ax_preset=None,
    linewidth_list=None,
    linestyle_list=None,
    xtick_interval: float = 0.1,
    vlines=None,
    vlines_colors=None,
    vlines_linestyle=None,
    vlines_linewidth=None,
    ylim_erf=None,
    dpi: int = 300,
    # peaks
    compute_peaks: bool = False,
    peak_windows: list[tuple[float, float]] | None = None,
    peak_mode: str = "abs",              # 'pos'|'neg'|'abs'
    peak_plot_alpha: float = 0.0,
    peak_marker_size: float = 50.0,
    flag_test: bool = False,
    **kwargs
):
    """
    Plot ERFs for two conditions reduced over channels by mean or RMS.
    Saves JPG/PDF and returns useful arrays.
    """
    # --- compute core (no diff) ---
    core = _compute_erf_core(
        sub_dict=sub_dict,
        cond1_name=cond1_name,
        cond2_name=cond2_name,
        ch_names_all=ch_names_all,
        sig_chans_use=sig_chans_use,
        chan_type=chan_type,
        times_use=times_use,
        times_use_idx=times_use_idx,
        reduce=reduce,
        flag_scale_channel_value=flag_scale_channel_value,
        want_diff=False,
        flag_test = flag_test,
    )
    # participants=participants,
    # selected_channels=ch_sel_names,
    # ylabel_reduce=ylabel_reduce,
    # unit=unit,
    # factor=factor,
    # condsubt=condsubt,   # (Cond, Sub, T)
    # condt=condt,         # (Cond, T)
    # condt_ci=condt_ci,   # (Cond, T, 2)
    # diff_subt=diff_subt, # (Sub, T) or None
    # dift=dift,           # (T,)     or None
    # dift_ci=dift_ci,     # (T, 2)   or None
    # erfdiffdata_path=erfdiffdata_path_str,
    
    # --- low-pass + slice to requested window ---
    condt_filt = apply_lowpass_filter(core["condt"], fs=fs)                 # (Cond, T)
    condt_ci_filt = apply_lowpass_filter(core["condt_ci"], fs=fs)           # (Cond, T, 2)
    plat_condt = condt_filt[:, times_use_idx]                               
    plat_condt_ci = condt_ci_filt[:, times_use_idx, :]

    # --- plot ---
    conds_name = [cond1_name, cond2_name]
    conds_colors = [colordict[cond1_name], colordict[cond2_name]]
    ylabel = f"Activity ({core['ylabel_reduce']} {core['unit']})"
    ylim = ylim_erf if flag_preset_ylim else None

    ax = plot_time_series_sigline(
        plotdata_save_folder_path=os.path.join(data_dir, _slug(title)),
        data=plat_condt,
        err=None,
        ci_1D=plat_condt_ci,
        t0=times_use[0],
        tend=times_use[-1],
        ax=ax_preset,
        linewidth_list=linewidth_list,
        linestyle_list=linestyle_list,
        xtick_interval=xtick_interval,
        colors=conds_colors,
        vlines=vlines,
        vlines_colors=vlines_colors,
        vlines_linestyle=vlines_linestyle,
        vlines_linewidth=vlines_linewidth,
        xlim=None,
        ylim=ylim,
        xlabel="Time (s)",
        ylabel=ylabel,
        err_transparency=0.2,
        title=title,
        square_fig=False,
        conditions=conds_name,
        do_legend=False,
        sig_hatchedpatterns=window_list,
        sig_hatchedpattern_dataidx=[0, 1],
        dpi=300,
    )

    # --- optional peaks on the group curves ---
    peaks_table = None
    if compute_peaks:
        wins = peak_windows or [(times_use[0], times_use[-1])]
        rows = []
        for ci in range(plat_condt.shape[0]):  # each condition
            rows_ci = _peak_per_window(plat_condt[ci], times_use, wins, mode=peak_mode)
            for r in rows_ci:
                r["condition"] = conds_name[ci]
                rows.append(r)
                if peak_plot_alpha and peak_plot_alpha > 0:
                    print("plot the peaks x ", r["t_peak"],"y ",r["amp"])
                          
                    ax.scatter(r["t_peak"], r["amp"],
                               s=peak_marker_size,
                               edgecolor=conds_colors[ci],
                               facecolor=conds_colors[ci],
                               alpha=peak_plot_alpha,
                               zorder=5)
        peaks_table = pd.DataFrame(rows)

    # legend & save
    h, l = ax.get_legend_handles_labels()
    fig = ax.figure
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)

    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(jpg_dir / f"{_slug(title)}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{_slug(title)}.pdf", bbox_inches="tight", dpi=300)

    # --- optional: plot each participant (Cond x Sub x T) ---
    if flag_plot_each_participant:
        title_each = f"Each Participant {title}"
        try:
            import seaborn as sns
            palette = sns.color_palette("husl", 2)
        except Exception:
            palette = None
        fig_each, ax_each = _plot_subject_subplots(
            condsubt_use=core["condsubt"][:, :, times_use_idx],
            times_use=times_use,
            title=title_each,
            cond1_content_name=cond1_name,
            cond2_content_name=cond2_name,
            palette=palette,
        )
        fig_each.savefig(jpg_dir / f"{_slug(title_each)}.jpg", bbox_inches="tight", dpi=300)
        fig_each.savefig(pdf_dir / f"{_slug(title_each)}.pdf", bbox_inches="tight", dpi=300)

    return {
        "condt_raw": core["condt"],            # (Cond, T) before filtering
        "condt": condt_filt,                   # (Cond, T) after filtering
        "plat_condt": plat_condt,              # (Cond, len(times_use))
        "condt_ci": condt_ci_filt,             # (Cond, T, 2) after filtering
        "times_use_idx": times_use_idx,
        "times_use": times_use,
        "condsubt": core["condsubt"],          # (Cond, Sub, T) scaled
        "participants": core["participants"],
        "ch_names_all" : ch_names_all,
        "selected_channels": core["selected_channels"],
        "window_list": window_list,
        "unit": core["unit"],
        "factor": core["factor"],
        "peaks_table": peaks_table,
    }
#%% fun plot: plot_erf_and_diff_reduce_and_save
# --------------- plot: two conditions + their difference --------------- #
def plot_erf_and_diff_reduce_and_save(
    sub_dict: dict,
    cond1_name: str,
    cond2_name: str,
    ch_names_all: list[str],
    sig_chans_use: set[str] | list[str],
    chan_type: str,
    fs: float,
    times_use: np.ndarray,
    times_use_idx: np.ndarray,
    flag_preset_ylim: bool,
    window_list: list | None,
    colordict: dict,
    reduce: str,                       # {"rms", "mean"}
    erfdiffdata_dir: str | Path,
    erfdiffdata_stem: str,                   # e.g., "ERFdiff 1ofN"
    title: str,
    jpg_dir: str | Path,
    pdf_dir: str | Path,
    data_dir: str | Path,
    flag_scale_channel_value: bool = True,
    # cosmetics
    ax_preset=None,
    ax_diff_preset=None,
    linewidth_list=None,
    linestyle_list=None,
    xtick_interval: float = 0.1,
    vlines=None,
    vlines_colors=None,
    vlines_linestyle=None,
    vlines_linewidth=None,
    ylim_erf=None,
    diff_color: str = "grey",
    dpi: int = 300,
    flag_plot_diff_separate: bool = False,
    # peaks
    compute_peaks: bool = False,
    peak_windows: list[tuple[float, float]] | None = None,
    peak_mode: str = "pos",
    peak_marker_size: float = 40.0,
    peak_plot_alpha: float = 0.0,
    flag_test: bool = False,
    **kwargs
):
    """
    Plot ERFs for two conditions AND their difference
    (cond1 - cond2 with bootstrap CI, cached). Saves JPG/PDF.
    """
    # --- compute core (with diff) ---
    core = _compute_erf_core(
        sub_dict=sub_dict,
        cond1_name=cond1_name,
        cond2_name=cond2_name,
        ch_names_all=ch_names_all,
        sig_chans_use=sig_chans_use,
        chan_type=chan_type,
        times_use=times_use,
        times_use_idx=times_use_idx,
        reduce=reduce,
        flag_scale_channel_value=flag_scale_channel_value,
        want_diff=True,
        erfdiffdata_dir=erfdiffdata_dir,
        erfdiffdata_stem=erfdiffdata_stem,
        flag_test=flag_test,
    )
    # participants=participants,
    # selected_channels=ch_sel_names,
    # ylabel_reduce=ylabel_reduce,
    # unit=unit,
    # factor=factor,
    # condsubt=condsubt,   # (Cond, Sub, T)
    # condt=condt,         # (Cond, T)
    # condt_ci=condt_ci,   # (Cond, T, 2)
    # diff_subt=diff_subt, # (Sub, T) or None
    # dift=dift,           # (T,)     or None
    # dift_ci=dift_ci,     # (T, 2)   or None
    # erfdiffdata_path=erfdiffdata_path_str,

    # --- low-pass + slice: stack 3 curves (cond1, cond2, diff) ---
    condt_filt = apply_lowpass_filter(core["condt"], fs=fs)                    # (2, T)
    dift_filt = apply_lowpass_filter(core["dift"][np.newaxis, :], fs=fs)        # (1, T)
    condt_ci_filt = apply_lowpass_filter(core["condt_ci"], fs=fs)               # (2, T, 2)
    dift_ci_filt = apply_lowpass_filter(core["dift_ci"][np.newaxis, :, :], fs=fs)  # (1, T, 2)

    three_cond_t = np.concatenate((condt_filt, dift_filt), axis=0)              # (3, T)
    three_ci = np.concatenate((condt_ci_filt, dift_ci_filt), axis=0)            # (3, T, 2)

    # slice to window
    plat_three_t = three_cond_t[:, times_use_idx]
    plat_three_ci = three_ci[:, times_use_idx, :]

    # --- plot combined ---
    conds_name = [cond1_name, cond2_name, "ERF Difference"]
    conds_colors = [colordict[cond1_name], colordict[cond2_name], diff_color]
    ylabel = f"Activity ({core['ylabel_reduce']} {core['unit']})"
    ylim = ylim_erf if flag_preset_ylim else None

    ax = plot_time_series_sigline(
        plotdata_save_folder_path=os.path.join(data_dir, _slug(title)),
        data=plat_three_t,
        err=None,
        ci_1D=plat_three_ci,
        t0=times_use[0],
        tend=times_use[-1],
        ax=ax_preset,
        linewidth_list=linewidth_list,
        linestyle_list=linestyle_list,
        xtick_interval=xtick_interval,
        colors=conds_colors,
        vlines=vlines,
        vlines_colors=vlines_colors,
        vlines_linestyle=vlines_linestyle,
        vlines_linewidth=vlines_linewidth,
        xlim=None,
        ylim=ylim,
        xlabel="Time (s)",
        ylabel=ylabel,
        err_transparency=0.2,
        title=title,
        square_fig=False,
        conditions=conds_name,
        do_legend=False,
        sig_hatchedpatterns=window_list,
        sig_hatchedpattern_dataidx=[0, 1],
        dpi=300,
    )

    # --- optional peaks (on all three curves) ---
    peaks_table = None
    if compute_peaks:
        wins = peak_windows or [(times_use[0], times_use[-1])]
        rows = []
        for ci in range(plat_three_t.shape[0]):
            rows_ci = _peak_per_window(plat_three_t[ci], times_use, wins, mode=peak_mode)
            for r in rows_ci:
                r["condition"] = conds_name[ci]
                rows.append(r)
                if peak_plot_alpha and peak_plot_alpha > 0:
                    print("plot the peaks x ", r["t_peak"],"y ",r["amp"])

                    ax.scatter(r["t_peak"], r["amp"],
                               s=peak_marker_size,
                               edgecolor=conds_colors[ci],
                               facecolor=conds_colors[ci],
                               alpha=peak_plot_alpha,
                               zorder=5)
        peaks_table = pd.DataFrame(rows)

    # legend & save
    h, l = ax.get_legend_handles_labels()
    fig = ax.figure
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)

    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(title)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)

    # --- optional separate diff-only plot ---
    if flag_plot_diff_separate:
        title_diff = f"Onlydiff {title}"
        ax_diff = plot_time_series_sigline(
            plotdata_save_folder_path=os.path.join(data_dir, _slug(title_diff)),
            data=dift_filt[:, times_use_idx],                # (1, len(times_use))
            ci_1D=dift_ci_filt[:, times_use_idx, :],        # (1, len(times_use), 2)
            err=None,
            t0=times_use[0],
            tend=times_use[-1],
            ax=ax_diff_preset,
            linewidth_list=linewidth_list,
            linestyle_list=linestyle_list,
            xtick_interval=xtick_interval,
            colors=["black"],
            vlines=vlines,
            vlines_colors=vlines_colors,
            vlines_linestyle=vlines_linestyle,
            vlines_linewidth=vlines_linewidth,
            xlim=None,
            ylim=None,
            xlabel="Time (s)",
            ylabel=ylabel,
            err_transparency=0.2,
            title=title_diff,
            square_fig=False,
            conditions=["ERF Difference"],
            do_legend=False,
            dpi=300,
        )
        fig_d = ax_diff.figure
        fig_d.savefig(jpg_dir / f"{_slug(title_diff)}.jpg", bbox_inches="tight", dpi=300)
        fig_d.savefig(pdf_dir / f"{_slug(title_diff)}.pdf", bbox_inches="tight", dpi=300)

    return {
        "condt_raw": core["condt"],         # (2, T) before filtering
        "dift_raw": core["dift"],           # (T,) before filtering
        "three_cond_t": three_cond_t,       # (3, T) filtered (stacked)
        "three_ci": three_ci,               # (3, T, 2) filtered
        "plat_three_t": plat_three_t,       # (3, len(times_use))
        "times_use_idx": times_use_idx,
        "times_use": times_use,
        "condsubt": core["condsubt"],       # (Cond, Sub, T) scaled
        "diff_subt": core["diff_subt"],     # (Sub, T)
        "participants": core["participants"],
        "ch_names_all" : ch_names_all,
        "selected_channels": core["selected_channels"],
        "unit": core["unit"],
        "window_list": window_list,
        "factor": core["factor"],
        "erfdiffdata_path": core["erfdiffdata_path"],
        "peaks_table": peaks_table,
    }

#%% fun plot: plot_erf_erfdiff
def plot_erf_erfdiff(
    # ---- names for the two figures ----
    pic_name_erf: str,
    pic_name_erfdiff: str,

    # ---- plot averaging setting ----
    reduce: str,                                  # {"rms","mean"}

    # ---- data ----
    sub_dict: Dict[str, Any],

    # ---- channel config ----
    chan_type: str,
    ch_names_all: Sequence[str],
    sig_chans_use: Sequence[str],
    flag_scale_channel_value: bool ,

    # ---- time config ----
    times_use_idx: Sequence[int],
    times_use: Sequence[float],              # passed as 'times_use' to wrappers
    # epoch_setting: Dict[str, Any],                # expects {'epoch_decim': ...}
    fs: float,

    # ---- plotting choices / labels ----
    colordict: Dict[str, Any],
    cond1_content_name: str,
    cond2_content_name: str,

    # ---- save paths & diff cache ----
    erfdiffdata_dir: str ,
    erfdiffdata_stem: str,
    jpg_dir: str,
    pdf_dir: str,
    data_dir: str ,

    # ---- windows & peaks ----
    window_list: Optional[List[List[float]]] ,        # e.g., [[t0,t1], [t2,t3]]
    compute_peaks: bool ,
    peak_mode: str,
    peak_windows_erf: Optional[List[List[float]]] = None,   # default -> window_list
    peak_windows_erfdiff: Optional[List[List[float]]] = None,  # default -> window_list
    peak_plot_alpha_erf: float = 0.0,                       # 0 = compute only
    peak_plot_alpha_diff: float = 1.0,                      # 1 = draw


    # ---- define when plot difference bootstrap number----
    flag_test: bool = False,

    # ---- toggles ----
    flag_plot_diff_separate: bool = True,   
    flag_plot_each_participant: int = 0,
    do_erf: bool = True,
    do_erfdiff: bool = True,

    # ---- style (shared) ----
    ax_preset = None,
    ax_diff_preset = None,
    flag_preset_ylim: bool = False,
    ylim_erf: Optional[Tuple[float, float]] = None,
    linewidth_list: Optional[Sequence[float]] = None,
    linestyle_list: Optional[Sequence[str]] = None,
    xtick_interval: float = 0.1,
    vlines: Optional[Sequence[float]] = None,
    vlines_colors: Optional[Sequence[str]] = None,
    vlines_linestyle: Optional[Sequence[str]] = None,
    vlines_linewidth: Optional[Sequence[float]] = None,
    diff_color: str = "grey",
    **kwargs    
):
    """
    Thin adapter that calls:
      - plot_erf_reduce_and_save (ERF only)
      - plot_erf_and_diff_reduce_and_save (ERF + Difference)

    It accepts your current common_kwargs layout and maps names to the new wrappers.
    Returns (erf_out, erfdiff_out).
    """

    # --- defaults ---
    if peak_windows_erf is None:
        peak_windows_erf = window_list
    if peak_windows_erfdiff is None:
        peak_windows_erfdiff = window_list

    # --- shared kwargs for both wrappers (map names internally) ---
    shared = dict(
        sub_dict=sub_dict,
        cond1_name=cond1_content_name,         # wrapper expects *_name
        cond2_name=cond2_content_name,
        ch_names_all=ch_names_all,          # wrapper expects all names here
        sig_chans_use=sig_chans_use,
        chan_type=chan_type,
        fs=fs,
        times_use=times_use,              # wrapper expects times_use
        times_use_idx=times_use_idx,
        flag_preset_ylim=flag_preset_ylim,
        window_list=window_list,
        colordict=colordict,
        reduce=reduce,
        jpg_dir=jpg_dir,
        pdf_dir=pdf_dir,
        data_dir=data_dir,
        flag_scale_channel_value=flag_scale_channel_value,
        # style
        ax_preset=ax_preset,
        linewidth_list=linewidth_list,
        linestyle_list=linestyle_list,
        xtick_interval=xtick_interval,
        vlines=vlines,
        vlines_colors=vlines_colors,
        vlines_linestyle=vlines_linestyle,
        vlines_linewidth=vlines_linewidth,
        ylim_erf=ylim_erf,
        dpi=300,
    )

    erf_out = None
    erfdiff_out = None

    # --- ERF only ---
    if do_erf:
        erf_out = plot_erf_reduce_and_save(
            **shared,
            title=pic_name_erf,
            flag_plot_each_participant=flag_plot_each_participant,
            compute_peaks=compute_peaks,
            peak_windows=peak_windows_erf,
            peak_mode=peak_mode,
            peak_plot_alpha=peak_plot_alpha_erf,
            flag_test=flag_test,
        )

    # --- ERF + DIFF ---
    if do_erfdiff:
        erfdiff_out = plot_erf_and_diff_reduce_and_save(
            **shared,
            ax_diff_preset=ax_diff_preset,
            title=pic_name_erfdiff,
            flag_plot_diff_separate=flag_plot_diff_separate,  # spelling mapped
            erfdiffdata_dir=erfdiffdata_dir,
            erfdiffdata_stem=erfdiffdata_stem,
            diff_color=diff_color,
            compute_peaks=compute_peaks,
            peak_windows=peak_windows_erfdiff,
            peak_mode=peak_mode,
            peak_plot_alpha=peak_plot_alpha_diff,
            flag_test=flag_test,
        )

    return erf_out, erfdiff_out
#%% fun plot: subplot_erf_and_diff_reduce_and_save
def subplot_erf_and_diff_reduce_and_save(
    # layout
    ncols: int,

    
    # ---- toggles ----
    flag_scale_channel_value: bool,
    plot_what: str ,  # {"conds+diff","conds","diff"}
    fs: float,
   # ---- data ----
    sub_dict: dict,
    # ---- channel config ----
    ch_names_all: List[str],
    chan_type: str,
    # ---- time config ----
    times_use: np.ndarray,
    times_use_idx: np.ndarray,
    
    # ---- pannel config ----
    channels_per_panel: List[List[str]],
    windows_per_panel: List[List[Tuple[float, float]]],

    

    # ---- plot averaging setting ----
    reduce: str,
    # ---- plotting choices / labels ----
    colordict: dict,
    cond1_content_name: str,
    cond2_content_name: str,
    
    # ---- save paths & diff cache ----
    erfdiffdata_dir: str | Path,
    erfdiffdata_stem: str,
    jpg_dir: str | Path,
    pdf_dir: str | Path,
    data_dir: str | Path,
    
    # ---- style (shared) ----
    flag_preset_ylim: bool ,
    ylim_erf: Optional[Tuple[float, float]] = None,
    linewidth_list: Optional[Sequence[float]] = None,
    linestyle_list: Optional[Sequence[str]] = None,
    xtick_interval: float = 0.1,
    vlines: Optional[Sequence[float]] = None,
    vlines_colors: Optional[Sequence[str]] = None,
    vlines_linestyle: Optional[Sequence[str]] = None,
    vlines_linewidth: Optional[Sequence[float]] = None,
    diff_color: str = "grey",
    # ---- windows & peaks ----
    compute_peaks: bool = False,
    peak_windows: Optional[List[Tuple[float, float]]] = None,
    peak_mode: str = "pos",
    peak_marker_size: int = 40,
    peak_plot_alpha: float = 0.0,
    # layout
    suptitle: str = "",
    tile_size_in: Tuple[float, float] = (7.2, 4.25),
    hspace: float = 0.35,
    wspace: float = 0.25,
    # ---- pannel config ----
    titles_per_panel: Optional[List[str]] = None,
   **kwargs

):
    """
    Grid plot: each panel = its own channel set + windows.
    Reuses _compute_erf_core and plot_time_series_sigline.
    Saves combined JPG/PDF 
    """

    assert plot_what in {"conds+diff", "conds", "diff"}
    n_panels = len(channels_per_panel)
    assert n_panels > 0
    assert len(windows_per_panel) == n_panels
    if titles_per_panel is not None:
        assert len(titles_per_panel) == n_panels

    # ---- layout ----
    ncols = int(ncols)
    nrows = int (np.ceil(n_panels / ncols))
    print(nrows, ncols)
    fig_w = tile_size_in[0] * ncols
    fig_h = tile_size_in[1] * nrows
    fig, axes_grid = plt.subplots(nrows, 
                                  ncols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    axes = axes_grid.flatten()  # concise & robust (works for 1+ panels)

    # ---- shared ----
    ylim = ylim_erf if flag_preset_ylim else None
    conds_name = [cond1_content_name, cond2_content_name]
    conds_colors = [colordict[cond1_content_name], colordict[cond2_content_name]]

    per_panel_outputs = []
    for i in range(n_panels):
        ax = axes[i]
        panel_chans = channels_per_panel[i]
        panel_windows = windows_per_panel[i] or []
        panel_title = titles_per_panel[i] if titles_per_panel else f"Panel {i+1}"

        want_diff = (plot_what != "conds")
        core = _compute_erf_core(
            sub_dict=sub_dict,
            cond1_name=cond1_content_name,
            cond2_name=cond2_content_name,
            ch_names_all=ch_names_all,
            sig_chans_use=panel_chans,
            chan_type=chan_type,
            times_use=times_use,
            times_use_idx=times_use_idx,
            reduce=reduce,
            flag_scale_channel_value=flag_scale_channel_value,
            want_diff=want_diff,
            erfdiffdata_dir=erfdiffdata_dir,
            erfdiffdata_stem=f"{erfdiffdata_stem}_p{i+1}of{n_panels}",
        )

        # filter & slice
        condt_filt = apply_lowpass_filter(core["condt"], fs=fs)                  # (2, T)
        condt_ci_filt = apply_lowpass_filter(core["condt_ci"], fs=fs)            # (2, T, 2)
        plat_condt = condt_filt[:, times_use_idx]
        plat_condt_ci = condt_ci_filt[:, times_use_idx, :]

        if plot_what in {"conds+diff", "diff"}:
            dift_filt = apply_lowpass_filter(core["dift"][np.newaxis, :], fs=fs)      # (1, T)
            dift_ci_filt = apply_lowpass_filter(core["dift_ci"][np.newaxis, :, :], fs=fs)  # (1, T, 2)
            plat_dift = dift_filt[:, times_use_idx]
            plat_dift_ci = dift_ci_filt[:, times_use_idx, :]

        # stack to plot
        if plot_what == "conds":
            data_stack = plat_condt
            ci_stack = plat_condt_ci
            colors_stack = conds_colors
            names_stack = conds_name
            hatched_idx = [0, 1]
        elif plot_what == "diff":
            data_stack = plat_dift
            ci_stack = plat_dift_ci
            colors_stack = [diff_color]
            names_stack = ["ERF Difference"]
            hatched_idx = [0]
        else:  # "conds+diff"
            data_stack = np.concatenate((plat_condt, plat_dift), axis=0)
            ci_stack = np.concatenate((plat_condt_ci, plat_dift_ci), axis=0)
            colors_stack = conds_colors + [diff_color]
            names_stack = conds_name + ["ERF Difference"]
            hatched_idx = [0, 1]  # hatch only conditions

        # peaks windows per panel
        if compute_peaks:
            if panel_windows:
                wins_for_peaks = panel_windows
            elif peak_windows:
                wins_for_peaks = peak_windows
            else:
                wins_for_peaks = [(times_use[0], times_use[-1])]
        else:
            wins_for_peaks = []

        # draw
        ylabel = f"Activity ({core['ylabel_reduce']} {core['unit']})"
        ax_function = plot_time_series_sigline(
            data=data_stack,
            err=None,
            ci_1D=ci_stack,
            t0=times_use[0],
            tend=times_use[-1],
            ax=ax,
            colors=colors_stack,
            linewidth_list=None,
            linestyle_list=None,
            xtick_interval=xtick_interval,
            vlines=vlines,
            vlines_colors=vlines_colors,
            vlines_linestyle=vlines_linestyle,
            vlines_linewidth=vlines_linewidth,
            xlim=None,
            ylim=ylim,
            xlabel="Time (s)",
            ylabel=ylabel,
            err_transparency=0.2,
            title=panel_title,
            square_fig=False,
            conditions=names_stack,
            do_legend=False,
            sig_hatchedpatterns=panel_windows,
            sig_hatchedpattern_dataidx=hatched_idx,
            dpi=300,
        )

        # optional peaks
        peaks_table = None
        if compute_peaks and wins_for_peaks:
            rows = []
            for ti in range(data_stack.shape[0]):
                rows_ti = _peak_per_window(data_stack[ti], times_use, wins_for_peaks, mode=peak_mode)
                for r in rows_ti:
                    r["panel"] = i + 1
                    r["trace"] = names_stack[ti]
                    rows.append(r)
                    if peak_plot_alpha and peak_plot_alpha > 0:
                        print("plot the peaks x ", r["t_peak"],"y ",r["amp"])

                        ax_function.scatter(r["t_peak"], r["amp"],
                                   s=peak_marker_size,
                                   edgecolor=colors_stack[ti],
                                   facecolor=colors_stack[ti],
                                   alpha=peak_plot_alpha,
                                   zorder=5)
            if rows:
                peaks_table = pd.DataFrame(rows)

        per_panel_outputs.append({
            "panel_index": i,
            "panel_title": panel_title,
            "channels_used": panel_chans,
            "windows_used": panel_windows,
            "condt_raw": core["condt"],
            "condt_ci_raw": core["condt_ci"],
            "dift_raw": core.get("dift", None),
            "dift_ci_raw": core.get("dift_ci", None),
            "peaks_table": peaks_table,
        })

    # hide unused tiles
    total_tiles = nrows * ncols
    for j in range(n_panels, total_tiles):
        axes[j].axis("off")

    # save
    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=12)
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(suptitle)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)

    # sidecar
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "suptitle": suptitle, 
        "plot_what": plot_what,
        "ncols": ncols, "nrows": nrows,
        "channels_per_panel": channels_per_panel,
        "windows_per_panel": windows_per_panel,
        "reduce": reduce, "chan_type": chan_type,
        "flag_scale_channel_value": flag_scale_channel_value,
        "flag_preset_ylim": flag_preset_ylim,
        "ylim_erf": list(ylim_erf) if ylim_erf is not None else None,
        "times_use": times_use.tolist(),
        "times_use_idx": np.asarray(times_use_idx).tolist(),
        "erfdiffdata_dir": str(erfdiffdata_dir), "erfdiffdata_stem": erfdiffdata_stem,
        "files": {"jpg": str(jpg_dir / f"{stem}.jpg"), "pdf": str(pdf_dir / f"{stem}.pdf")},
    }
    with open(data_dir / f"{stem}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return {
        "figure": fig,
        "axes": axes_grid,          # original 2D grid, if you need [r,c] access
        "per_panel_outputs": per_panel_outputs,
        "sidecar_path": str(data_dir / f"{stem}.json"),
    }
    
    
    
    # if flag_plot_diff_separate:
    #     title_diff = f'Onlydiff {title}'
    #     # ------------------------ plot
    #     conds_name = [ "ERF Difference"]
    #     conds_colormap = ['k']
                        
    #     ylabel_name = f"Activity ({ylabel_reduce} {unit})"
    #     ylim = None
    #     print(cond_t_y[-1:, times_use_idx].shape)
        
    #     ax = plot_time_series_sigline(
    #         plotdata_save_folder_path = os.path.join(data_dir,_slug(title_diff)),
    #         data=cond_t_y[-1:, times_use_idx],
    #         ci_1D=ci_ctb[-1:, times_use_idx, :],
    #         linewidth_list=linewidth_list,
    #         linestyle_list=linestyle_list,
    #         xtick_interval=xtick_interval,
    #         err=None,
    #         t0=times_use[0],
    #         tend=times_use[-1],
    #         ax=None,
    #         colors=conds_colormap,
    #         vlines=vlines,
    #         vlines_colors=vlines_colors,
    #         vlines_linestyle=vlines_linestyle,
    #         vlines_linewidth=vlines_linewidth,
    #         xlim=None,
    #         ylim=ylim,
    #         xlabel="Time (s)",
    #         ylabel=ylabel_name,
    #         err_transparency=0.2,
    #         title=title_diff,
    #         square_fig=False,
    #         conditions=conds_name,
    #         do_legend=False,
    #         # patches=( [cond1_timewindow[0], cond1_timewindow[1]] if cond1_timewindow else None ),
    #         # patch_color="lightgrey",
    #         # patch_transparency=0.2,
    #         # sig_hatchedpatterns=window_list,
    #         # sig_hatchedpattern_dataidx=[0, 1],
    #         dpi=300,
    #     )

    #     # legend & save
    #     h, l = ax.get_legend_handles_labels()
    #     fig = ax.figure
    #     if h:
    #         fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)
    #     fig.savefig(jpg_dir / f"{_slug(title_diff)}.jpg", bbox_inches="tight", dpi=300)
    #     fig.savefig(pdf_dir / f"{_slug(title_diff)}.pdf", bbox_inches="tight", dpi=300)

    # return {
    #      "cond_t_y": cond_t_y,                 # (3, n_times) -> [cond1, cond2, diff]
    #      "ci": ci_ctb,                         # (3, n_times, 2)
    #      "cond_t_y_erf": cond_t_y_erf,        # RAW (2, n_times) 
    #      "erfdif_y_t": erfdif_y_t,            # RAW (n_times,)
    #      "times_use_idx": times_use_idx,     # indices of times_use in original time axis
    #      "times_use": times_use,             # cond_t_y[:, times_use_idx].shape[-1] == len(times_use)
    #     "platdata_cond_t": platdata_cond_t,  # (3, len(times_use))
    #     "ch_names_all":ch_names_all,
    #     "sig_chans_use": sig_chans_use,
    #     "unit": unit,
    #     "window_list":window_list,
    #     "factor": factor,
    #     "erfdiffdata_path": str(erfdiffdata_path),
    #     'peaks_table':peaks_table
        
    # }
#%% fun plot: jointplot_fullsensor


# def _extract_joint_figure(ret):
#     """Handle MNE versions that return a dict vs a plain Figure."""
#     # Common cases:
#     # - plain matplotlib.figure.Figure
#     # - dict with "joint" (older MNE) or "fig" (some viz functions)
#     if hasattr(ret, "savefig"):        # looks like a Figure
#         return ret
#     if isinstance(ret, dict):
#         if "joint" in ret and hasattr(ret["joint"], "savefig"):
#             return ret["joint"]
#         if "fig" in ret and hasattr(ret["fig"], "savefig"):
#             return ret["fig"]
#     raise TypeError("Unexpected return from plot_joint; cannot find Figure to save.")

def jointplot_fullsensor(
    evoked,  
    chan_type: str, 
    times_plot, 
    title: str,
    jpg_dir: str | Path, 
    pdf_dir: str | Path, 
    dpi: int = 300,
    ylim: tuple|None = None,
    highlight_window : list|None = None, 
    **kwargs
) -> None:
    """Create and save a full joint figure (TS + topomaps) for the given picks.
    """
    print('--- jointplot_fullsensor ---')
    print('ylim=',ylim)
          
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    
    ev = evoked.copy().pick(picks =chan_type )

    # vmax = abs(ev.data).max()  # find the maximum absolute value
    # ylim = {}
    # ylim[picks]=(-vmax, vmax)  # symmetric y-limits for EEG (adjust key if using 'mag', 'grad', etc.)
    # print(ylim), ylim=ylim
    ret = ev.plot_joint(
        times=times_plot,
        title=title,
        ts_args={'gfp': False,
                 'ylim': {chan_type: ylim}},
        topomap_args={'outlines': "head"},
    );
    
    fig = ret
    stem = _slug(title)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    return ev.data
#%% fun plot: butterflyplot_fullsensor

def butterflyplot_fullsensor(
    evoked,
    chan_type: str, 
    title: str,
    jpg_dir: str | Path, 
    pdf_dir: str | Path, 
    dpi: int = 300,
    highlight_window : list|None = None, 
        ylim: tuple|None = None,
    **kwargs
) -> None:
    """Create and save a full joint figure (TS + topomaps) for the given picks.
    """
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    
    ev = evoked.copy()
   
    # vmax = abs(ev.data).max()  # find the maximum absolute value
    # ylim = {}
    # ylim[picks]=(-vmax, vmax)  # symmetric y-limits for EEG (adjust key if using 'mag', 'grad', etc.)
    # print(ylim), ylim=ylim
    # ret = ev.plot_joint(
    #     times=times_plot,
    #     title=title,
    #     ts_args=dict(gfp=False),
    #     topomap_args=dict(outlines="head",spatial_colors=True,),
    # )
    
    ret = ev.plot(
        picks=chan_type,
        spatial_colors=True, 
        gfp=False, 
        highlight=highlight_window,
        ylim={chan_type: ylim},

)
    
    fig = ret
    fig.suptitle( title, y=1.01)

    stem = _slug(title)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    return ev.data
#%% fun plot: save_topo_of_window
def save_topo_of_window(
    evoked, 
    time_window,
    chan_type,
    chs_use,
    title,
    jpg_dir: str | Path, 
    pdf_dir: str | Path, 
    dpi: int = 300,
    mask_params = dict(marker='o',
                           markerfacecolor='w', 
                           markeredgecolor='k',
        linewidth=0, 
        markersize=5)
) -> None:
    """Create and save a full joint figure (TS + topomaps) for the given picks."""
    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)

    ev = evoked.copy().pick(picks=chan_type)
    picks = mne.pick_channels(
                        ch_names = ev.info['ch_names'], 
                        include=chs_use)

    t   = (time_window[1] + time_window[0])/2
    dur = time_window[1] - time_window[0]
    
    mask_sensor = np.array([x in chs_use for x in  ev.info['ch_names'] ])# shape (n_channels,) bool
    mask = np.tile(mask_sensor[:, None], (1, len(ev.times)))  # (n_sensors, n_maps)


    ret = ev.plot_topomap(
        times= t,
        average = dur,
        mask_params = mask_params,
        mask = mask,
        )
    fig = ret
    fig.suptitle( title, y=1.01)
    stem = _slug(title)
    fig.savefig(jpg_dir / f"{stem}.jpg", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    return ev.data
    



#%% fun three: trial_indices_multi
def trial_indices_multi(
    meta_col: pd.Series,
    cond_values: Dict[str, Iterable[str]],
) -> Dict[str, List[int]]:
    """
    Map each condition name -> list of trial indices whose label is in cond_values[name].
    Ensures no overlap between conditions.
    """
    labels = meta_col.fillna("no").tolist()
    idx_map: Dict[str, List[int]] = {
        name: [i for i, v in enumerate(labels) if v in values]
        for name, values in cond_values.items()
    }

    # Safety: no overlap
    all_sets = [set(v) for v in idx_map.values()]
    overlap = set().union(*(a & b for i, a in enumerate(all_sets) for b in all_sets[i+1:]))
    if overlap:
        raise ValueError(f"A trial belongs to multiple conditions: indices {sorted(overlap)}")

    return idx_map


#%% fun calc three 1: _validate_and_select_multi
def _validate_and_select_multi(
    sub_dict: dict,           # {participant: {cond: (n_ch, n_times)}}
    cond_names: list[str],    # list of condition names
    ch_names_all: list[str],  # all channel names
    sig_chans_use: Iterable[str],  # channels to select
    times_use: np.ndarray,    # time axis for plotting
    times_use_idx: np.ndarray,  # indices into full time array
):
    """
    Validate input and select channels for multiple conditions.
    
    Returns: participants (list), ch_sel_idx (list), ch_sel_names (list)
    """
    assert len(times_use) == len(times_use_idx), "times_use and times_use_idx mismatch."
    participants = list(sub_dict.keys())
    if not participants:
        raise ValueError("sub_dict is empty.")

    first = participants[0]
    # Check all conditions exist
    for cond in cond_names:
        if cond not in sub_dict[first]:
            raise KeyError(f"Missing condition '{cond}' in participant '{first}'.")

    # Check all conditions have same shape
    first_shape = sub_dict[first][cond_names[0]].shape
    for cond in cond_names[1:]:
        if sub_dict[first][cond].shape != first_shape:
            raise ValueError(f"Condition '{cond}' shape mismatch: {sub_dict[first][cond].shape} vs {first_shape}")

    if len(first_shape) != 2:
        raise ValueError(f"Condition arrays must be 2d(n_channels, n_times), got {len(first_shape)}.")

    if len(ch_names_all) != first_shape[0]:
        raise AssertionError(
            f"len(ch_names_all)={len(ch_names_all)} does not match data channels={first_shape[0]}"
        )

    wanted = set(sig_chans_use)
    ch_sel_idx = [i for i, ch in enumerate(ch_names_all) if ch in wanted]
    ch_sel_names = [ch_names_all[i] for i in ch_sel_idx]
    if not ch_sel_idx:
        raise ValueError("No channels left after intersecting with sig_chans_use.")

    return participants, ch_sel_idx, ch_sel_names


#%% fun calc three 2: _stack_chsubcondt_multi
def _stack_chsubcondt_multi(
    sub_dict: dict,           # {participant: {cond: (n_ch, n_times)}}
    participants: list[str],  # participant IDs
    ch_sel_idx: list[int],    # selected channel indices
    cond_names: list[str],    # condition names
) -> np.ndarray:
    """
    Build 4D array: (Ch, Sub, Cond, T)
    """
    per_ch = []
    for ch in ch_sel_idx:
        per_sub = []
        for p in participants:
            # Stack all conditions for this participant and channel
            cond_arrays = [sub_dict[p][cond][ch, :] for cond in cond_names]
            # Check shape consistency
            first_shape = cond_arrays[0].shape
            for i, arr in enumerate(cond_arrays[1:], 1):
                if arr.shape != first_shape:
                    raise ValueError(
                        f"Shape mismatch for participant {p}, channel {ch}, "
                        f"condition {cond_names[i]}: {arr.shape} vs {first_shape}"
                    )
            per_sub.append(np.stack(cond_arrays, axis=0))  # (Cond=N, T)
        per_ch.append(np.stack(per_sub, axis=0))           # (Sub, Cond, T)
    return np.stack(per_ch, axis=0)                        # (Ch, Sub, Cond, T)


#%% fun calc three 3: _stack_chsubt_diff_pair
def _stack_chsubt_diff_pair(
    sub_dict: dict,           # {participant: {cond: (n_ch, n_times)}}
    participants: list[str],  # participant IDs
    ch_sel_idx: list[int],    # selected channel indices
    cond1: str,               # first condition name
    cond2: str,               # second condition name
) -> np.ndarray:
    """
    Compute difference (cond1 - cond2) stack: (Ch, Sub, T)
    """
    per_ch = []
    for ch in ch_sel_idx:
        per_sub = []
        for p in participants:
            x1 = sub_dict[p][cond1][ch, :]
            x2 = sub_dict[p][cond2][ch, :]
            per_sub.append(x1 - x2)                        # (T,)
        per_ch.append(np.stack(per_sub, axis=0))           # (Sub, T)
    return np.stack(per_ch, axis=0)                        # (Ch, Sub, T)


#%% fun calc three 4: _compute_erf_core_multi
def _compute_erf_core_multi(
    sub_dict: dict,           # {participant: {cond: (n_ch, n_times)}}
    cond_names: list[str],    # condition names
    ch_names_all: list[str],  # all channel names
    sig_chans_use: Iterable[str],  # channels to use
    chan_type: str,           # channel type for scaling
    times_use: np.ndarray,    # time axis for plotting
    times_use_idx: np.ndarray,  # indices into full time array
    reduce: str,              # "rms" or "mean"
    flag_scale_channel_value: bool,  # whether to scale by channel type
    diff_pairs: list[list[int]] | None = None,  # [[idx1,idx2],...] for differences
    erfdiffdata_dir: str | Path | None = None,  # cache directory for diff data
    erfdiffdata_stem_prefix: str | None = None,  # prefix for diff cache files
) -> dict:
    """
    Compute ERFs for multiple conditions and optional differences.
    
    Returns dict with:
        - condsubt: (Cond, Sub, T)
        - condt: (Cond, T) group means
        - condt_ci: (Cond, T, 2) confidence intervals
        - diff_results: list of dicts with difference data
    """
    # --- validate & select
    participants, ch_sel_idx, ch_sel_names = _validate_and_select_multi(
        sub_dict, 
        cond_names, 
        ch_names_all, 
        sig_chans_use, 
        times_use, 
        times_use_idx
    )

    # --- stack all conditions -> chsubcondt: (Ch, Sub, Cond, T)
    chsubcondt = _stack_chsubcondt_multi(
        sub_dict, 
        participants, 
        ch_sel_idx, 
        cond_names
    )

    # --- reduce channels -> subcondt: (Sub, Cond, T)
    if reduce.lower() == "rms":
        subcondt_raw = np.sqrt(np.mean(chsubcondt ** 2, axis=0))
        ylabel_reduce = "RMS"
    elif reduce.lower() == "mean":
        subcondt_raw = chsubcondt.mean(axis=0)
        ylabel_reduce = "Mean"
    else:
        raise ValueError("reduce must be 'rms' or 'mean'.")

    # --- reorder -> condsubt: (Cond, Sub, T)
    condsubt_raw = np.transpose(subcondt_raw, (1, 0, 2))

    # --- scale by channel type
    if flag_scale_channel_value:
        condsubt, unit, factor = _scale_by_chan_type(condsubt_raw, chan_type)
    else:
        condsubt, unit, factor = condsubt_raw, "AU", 1.0

    # --- group mean & CI
    condt, condt_ci_raw = compute_mean_exp2(
        condsubt, axis_cond=0, axis_sub=1, zscore=False, design="within"
    )
    condt_ci = convert_to_ci_1D(condt, condt_ci_raw)

    # --- compute differences if requested
    diff_results = []
    if diff_pairs:
        for pair in diff_pairs:
            if len(pair) != 2:
                raise ValueError(f"Each diff_pair must have exactly 2 indices, got {pair}")
            idx1, idx2 = pair
            
            if idx1 < 0 or idx1 >= len(cond_names) or idx2 < 0 or idx2 >= len(cond_names):
                raise ValueError(
                    f"Diff pair indices {pair} out of range for {len(cond_names)} conditions"
                )
            
            cond1, cond2 = cond_names[idx1], cond_names[idx2]
            
            # Stack difference: (Ch, Sub, T)
            chsubt_diff = _stack_chsubt_diff_pair(
                sub_dict, participants, ch_sel_idx, cond1, cond2
            )
            
            # Reduce channels
            if reduce.lower() == "rms":
                diff_subt_raw = np.sqrt(np.mean(chsubt_diff ** 2, axis=0))  # (Sub, T)
            else:
                diff_subt_raw = chsubt_diff.mean(axis=0)                    # (Sub, T)

            # Scale
            if flag_scale_channel_value:
                diff_subt, unit_d, _ = _scale_by_chan_type(diff_subt_raw, chan_type)
                assert unit_d == unit, "Unit mismatch between conditions and difference."
            else:
                diff_subt = diff_subt_raw

            # Compute or load difference statistics
            erfdiffdata_path_str = None
            if erfdiffdata_dir and erfdiffdata_stem_prefix:
                erfdiffdata_dir = Path(erfdiffdata_dir)
                erfdiffdata_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{_slug(erfdiffdata_stem_prefix)}_diff_{idx1}_vs_{idx2}.pkl"
                erfdiffdata_path = erfdiffdata_dir / fname
                
                if erfdiffdata_path.exists():
                    with open(erfdiffdata_path, "rb") as f:
                        dd = pickle.load(f)
                    dift, dift_ci = dd["erfdif_y_t"], dd["erfdif_c12bound_t"]
                else:
                    dift, dift_ci = compute_mean_bootstrap_1d(diff_subt)
                    with open(erfdiffdata_path, "wb") as f:
                        pickle.dump({
                            "erfdif_y_t": dift,
                            "erfdif_c12bound_t": dift_ci
                        }, f)
                erfdiffdata_path_str = str(erfdiffdata_path)
            else:
                dift, dift_ci = compute_mean_bootstrap_1d(diff_subt)
            
            diff_results.append({
                "pair_idx": [idx1, idx2],
                "pair_names": [cond1, cond2],
                "diff_subt": diff_subt,  # (Sub, T)
                "dift": dift,            # (T,)
                "dift_ci": dift_ci,      # (T, 2)
                "erfdiffdata_path": erfdiffdata_path_str,
            })

    return dict(
        participants=participants,
        selected_channels=ch_sel_names,
        ylabel_reduce=ylabel_reduce,
        unit=unit,
        factor=factor,
        condsubt=condsubt,      # (Cond, Sub, T)
        condt=condt,            # (Cond, T)
        condt_ci=condt_ci,      # (Cond, T, 2)
        diff_results=diff_results,
    )


#%% fun plot three 1: _plot_subject_subplots_multi
def _plot_subject_subplots_multi(
    condsubt_use: np.ndarray,  # (n_cond, n_sub, n_times)
    times_use: np.ndarray,     # time axis
    title: str,                # figure title
    cond_names: list[str],     # condition names
    palette=None,              # color list for conditions
):
    """
    Plot each participant's ERF data in subplots for multiple conditions.
    
    Returns: fig, axes
    """
    n_cond, n_sub, n_times = condsubt_use.shape
    
    # Create color palette if not provided
    if palette is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        palette = prop_cycle.by_key()['color'][:n_cond]
    
    # Determine subplot layout (roughly square)
    n_cols = int(np.ceil(np.sqrt(n_sub)))
    n_rows = int(np.ceil(n_sub / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_sub == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each participant
    for sub_idx in range(n_sub):
        ax = axes[sub_idx]
        
        # Plot all conditions for this participant
        for cond_idx in range(n_cond):
            y_data = condsubt_use[cond_idx, sub_idx, :]
            ax.plot(times_use, y_data, 
                   label=cond_names[cond_idx],
                   color=palette[cond_idx],
                   linewidth=1.5,
                   alpha=0.8)
        
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_title(f"Participant {sub_idx + 1}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Activity", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        
        if sub_idx == 0:  # Legend only in first subplot
            ax.legend(fontsize=8, loc='best')
    
    # Hide unused subplots
    for idx in range(n_sub, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig, axes


#%% fun plot three 2: plot_erf_reduce_multi_and_save 
def plot_erf_reduce_multi_and_save(
    sub_dict: dict,           # {participant: {cond: (n_ch, n_times)}}
    cond_names: list[str],    # condition names
    ch_names_all: list[str],  # all channel names
    sig_chans_use: set[str] | list[str],  # channels to use
    chan_type: str,           # channel type for scaling
    fs: float,                # sampling frequency
    times_use: np.ndarray,    # time axis for plotting
    times_use_idx: np.ndarray,  # indices into full time array
    flag_preset_ylim: bool,   # whether to use preset ylim
    window_list: list | None,  # significance windows for hatching
    colordict: dict,          # {cond_name: color}
    flag_plot_each_participant: bool,  # plot individual participants
    reduce: str,              # "rms" or "mean"
    title: str,               # plot title
    jpg_dir: str | Path,      # save directory for JPG
    pdf_dir: str | Path,      # save directory for PDF
    data_dir: str | Path,     # save directory for data
    flag_scale_channel_value: bool,  # scale by channel type
    # difference parameters
    diff_pairs: list[list[int]] | None = None,  # [[idx1,idx2],...] for differences
    flag_plot_diff_separate: bool = False,  # create separate diff plots
    diff_color: str = "grey",  # color for diff plots
    peak_plot_alpha_diff: float = 0.0,  # alpha for peaks on diff plots (0=no peaks)
    erfdiffdata_dir: str | Path | None = None,  # cache dir for diff data
    erfdiffdata_stem_prefix: str | None = None,  # prefix for diff cache files
    # cosmetics
    ax_preset=None,           # preset axis (or None to create new)
    linewidth_list=None,      # line widths for each condition
    linestyle_list=None,      # line styles for each condition
    xtick_interval: float = 0.1,  # x-axis tick interval
    vlines=None,              # vertical lines positions
    vlines_colors=None,       # vertical lines colors
    vlines_linestyle=None,    # vertical lines styles
    vlines_linewidth=None,    # vertical lines widths
    ylim_erf=None,            # y-axis limits
    dpi: int = 300,           # plot resolution
    # peaks
    compute_peaks: bool = False,  # whether to compute peaks
    peak_windows: list[tuple[float, float]] | None = None,  # time windows for peaks
    peak_mode: str = "abs",   # 'pos', 'neg', or 'abs'
    peak_plot_alpha: float = 0.0,  # alpha for peaks on main plot (0=no peaks)
    peak_marker_size: float = 50.0,  # peak marker size
):
    """
    Plot ERFs for multiple conditions reduced over channels by mean or RMS.
    Optionally plot differences separately.
    
    Returns: dict with arrays and metadata
    """
    # --- compute core (with optional differences) ---
    core = _compute_erf_core_multi(
        sub_dict=sub_dict,
        cond_names=cond_names,
        ch_names_all=ch_names_all,
        sig_chans_use=sig_chans_use,
        chan_type=chan_type,
        times_use=times_use,
        times_use_idx=times_use_idx,
        reduce=reduce,
        flag_scale_channel_value=flag_scale_channel_value,
        diff_pairs=diff_pairs,
        erfdiffdata_dir=erfdiffdata_dir,
        erfdiffdata_stem_prefix=erfdiffdata_stem_prefix,
    )
    
    # --- low-pass filter & slice to requested window ---
    condt_filt = apply_lowpass_filter(core["condt"], fs=fs)                 # (Cond, T)
    condt_ci_filt = apply_lowpass_filter(core["condt_ci"], fs=fs)           # (Cond, T, 2)
    plat_condt = condt_filt[:, times_use_idx]                               
    plat_condt_ci = condt_ci_filt[:, times_use_idx, :]

    # --- plot main conditions ---
    conds_colors = [colordict[name] for name in cond_names]
    ylabel = f"Activity ({core['ylabel_reduce']} {core['unit']})"
    ylim = ylim_erf if flag_preset_ylim else None

    ax = plot_time_series_sigline(
        plotdata_save_folder_path=os.path.join(data_dir, _slug(title)),
        data=plat_condt,
        err=None,
        ci_1D=plat_condt_ci,
        t0=times_use[0],
        tend=times_use[-1],
        ax=ax_preset,
        linewidth_list=linewidth_list,
        linestyle_list=linestyle_list,
        xtick_interval=xtick_interval,
        colors=conds_colors,
        vlines=vlines,
        vlines_colors=vlines_colors,
        vlines_linestyle=vlines_linestyle,
        vlines_linewidth=vlines_linewidth,
        xlim=None,
        ylim=ylim,
        xlabel="Time (s)",
        ylabel=ylabel,
        err_transparency=0.2,
        title=title,
        square_fig=False,
        conditions=cond_names,
        do_legend=False,
        sig_hatchedpatterns=window_list,
        sig_hatchedpattern_dataidx=list(range(len(cond_names))),
        dpi=300,
    )

    # --- optional peaks on main plot ---
    peaks_table = None
    if compute_peaks:
        wins = peak_windows or [(times_use[0], times_use[-1])]
        rows = []
        for ci in range(plat_condt.shape[0]):  # each condition
            rows_ci = _peak_per_window(plat_condt[ci], times_use, wins, mode=peak_mode)
            for r in rows_ci:
                r["condition"] = cond_names[ci]
                rows.append(r)
                if peak_plot_alpha and peak_plot_alpha > 0:
                    print("plot the peaks x", r["t_peak"], "y", r["amp"])
                    ax.scatter(r["t_peak"], r["amp"],
                               s=peak_marker_size,
                               edgecolor=conds_colors[ci],
                               facecolor=conds_colors[ci],
                               alpha=peak_plot_alpha,
                               zorder=5)
        peaks_table = pd.DataFrame(rows)

    # --- legend & save main plot ---
    h, l = ax.get_legend_handles_labels()
    fig = ax.figure
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(1.16, 0.5), fontsize=12)

    jpg_dir = Path(jpg_dir); jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(jpg_dir / f"{_slug(title)}.jpg", bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_dir / f"{_slug(title)}.pdf", bbox_inches="tight", dpi=dpi)

    # --- optional: plot each participant ---
    if flag_plot_each_participant:
        title_each = f"Each Participant {title}"
        try:
            palette = sns.color_palette("husl", len(cond_names))
        except Exception:
            palette = conds_colors  # Use main plot colors
        fig_each, ax_each = _plot_subject_subplots_multi(
            condsubt_use=core["condsubt"][:, :, times_use_idx],
            times_use=times_use,
            title=title_each,
            cond_names=cond_names,
            palette=palette,
        )
        fig_each.savefig(jpg_dir / f"{_slug(title_each)}.jpg", bbox_inches="tight", dpi=dpi)
        fig_each.savefig(pdf_dir / f"{_slug(title_each)}.pdf", bbox_inches="tight", dpi=dpi)

    # --- process and optionally plot difference results ---
    diff_outputs = []
    if core["diff_results"]:
        for diff_idx, diff_res in enumerate(core["diff_results"]):
            idx1, idx2 = diff_res["pair_idx"]
            name1, name2 = diff_res["pair_names"]
            
            # Filter difference data
            dift_filt = apply_lowpass_filter(diff_res["dift"], fs=fs)
            dift_ci_filt = apply_lowpass_filter(diff_res["dift_ci"], fs=fs)
            plat_dift = dift_filt[times_use_idx]
            plat_dift_ci = dift_ci_filt[times_use_idx, :]
            
            diff_outputs.append({
                "pair_idx": [idx1, idx2],
                "pair_names": [name1, name2],
                "diff_subt": diff_res["diff_subt"],           # (Sub, T)
                "dift": dift_filt,                            # (T,)
                "plat_dift": plat_dift,                       # (len(times_use),)
                "dift_ci": dift_ci_filt,                      # (T, 2)
                "plat_dift_ci": plat_dift_ci,                 # (len(times_use), 2)
                "erfdiffdata_path": diff_res["erfdiffdata_path"],
            })
            
            # --- optional: plot this difference separately ---
            if flag_plot_diff_separate:
                print(f"--------------Plotting difference {name1} vs {name2}")
                diff_title = f"{title} Difference {name1} vs {name2}"
                ylabel_diff = f"Difference ({core['ylabel_reduce']} {core['unit']})"
                ylim_diff = ylim_erf if flag_preset_ylim else None
                
                fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
                
                # Plot difference line
                ax_diff.plot(times_use, plat_dift, 
                           color=diff_color, 
                           linewidth=2.5,
                           label=f"{name1} - {name2}")
                
                # Plot CI
                ax_diff.fill_between(times_use, 
                                    plat_dift_ci[:, 0], 
                                    plat_dift_ci[:, 1],
                                    color=diff_color, 
                                    alpha=0.2)
                
                # Reference lines
                ax_diff.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
                ax_diff.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
                
                # Optional vertical lines
                if vlines:
                    for i, vl in enumerate(vlines):
                        vl_color = vlines_colors[i] if vlines_colors else 'gray'
                        vl_style = vlines_linestyle[i] if vlines_linestyle else '--'
                        vl_width = vlines_linewidth[i] if vlines_linewidth else 1
                        ax_diff.axvline(vl, color=vl_color, 
                                      linestyle=vl_style, 
                                      linewidth=vl_width)
                
                # Optional peaks on difference plot
                if compute_peaks and peak_plot_alpha_diff > 0:
                    wins = peak_windows or [(times_use[0], times_use[-1])]
                    diff_peaks = _peak_per_window(plat_dift, times_use, wins, mode=peak_mode)
                    for pk in diff_peaks:
                        ax_diff.scatter(pk["t_peak"], pk["amp"],
                                      s=peak_marker_size,
                                      edgecolor=diff_color,
                                      facecolor=diff_color,
                                      alpha=peak_plot_alpha_diff,
                                      zorder=5)
                
                # Formatting
                ax_diff.set_xlabel("Time (s)", fontsize=12)
                ax_diff.set_ylabel(ylabel_diff, fontsize=12)
                ax_diff.set_title(diff_title, fontsize=14, fontweight='bold')
                if ylim_diff:
                    ax_diff.set_ylim(ylim_diff)
                ax_diff.legend(loc='best', fontsize=11)
                ax_diff.grid(True, alpha=0.3)
                
                # Save difference plot
                diff_fname = f"{_slug(title)}_diff_{idx1}_vs_{idx2}"
                fig_diff.savefig(jpg_dir / f"{diff_fname}.jpg", 
                               bbox_inches="tight", dpi=dpi)
                fig_diff.savefig(pdf_dir / f"{diff_fname}.pdf", 
                               bbox_inches="tight", dpi=dpi)

    return {
        "condt_raw": core["condt"],               # (Cond, T) before filtering
        "condt": condt_filt,                      # (Cond, T) after filtering
        "plat_condt": plat_condt,                 # (Cond, len(times_use))
        "condt_ci": condt_ci_filt,                # (Cond, T, 2) after filtering
        "times_use_idx": times_use_idx,
        "times_use": times_use,
        "condsubt": core["condsubt"],             # (Cond, Sub, T) scaled
        "participants": core["participants"],
        "ch_names_all": ch_names_all,
        "selected_channels": core["selected_channels"],
        "window_list": window_list,
        "unit": core["unit"],
        "factor": core["factor"],
        "peaks_table": peaks_table,
        "diff_results": diff_outputs,             # list of processed difference dicts
    }
#%% class SourceROIEpochs
# epochs-like adapter for source ROI data ---
@dataclass
class SourceROIEpochs:
    """
    Minimal epochs-like wrapper for source ROI data.
    Provides .get_data(copy=True) -> (n_trials, n_vertices, n_times)
    """
    tlvt: np.ndarray  # shape: (T, V, time)

    def get_data(self, copy: bool = True) -> np.ndarray:
        return self.tlvt.copy() if copy else self.tlvt
#%% fun: _reduce
# customize functions for flexible average
Axis = Literal["trials", "vertices"]
Reducer = Union[Literal["mean", "rms"], Callable[[np.ndarray, int], np.ndarray]]

# ------------------------- reducers -------------------------
# function to reduce an array along a specified axis using a given method (mean, RMS, or custom function)
def _reduce(x: np.ndarray, axis: int, how: Reducer) -> np.ndarray:
    """Apply mean / RMS / custom reducer along axis."""
    if callable(how):
        return how(x, axis)
    if how == "mean":
        return np.mean(x, axis=axis)
    if how == "rms":
        return np.sqrt(np.mean(np.square(x), axis=axis))
    raise ValueError(f"Unknown reducer: {how!r}")
#%% class AggregationPlan
# ------------------------- config ---------------------------
# Here create a class: AggregationPlan to define how to _aggregate epochs data (T, V, time)
@dataclass(frozen=True)
class AggregationPlan:
    """
    How to _aggregate epochs data (T, V, time).
    order: sequence of axes to reduce, chosen from {"trials","vertices"}.
    """
    order: Tuple[Axis, ...] = ("trials",)
    trial_reducer: Optional[Reducer]  = None            # = "mean"
    vertex_reducer: Optional[Reducer] = None            # None = keep vertices
    across_samples_reducer: Reducer = "mean"            # combining subsamples
#%% fun not use: trial_indices
# # ------------------------- selection ------------------------
# # select trial for each condition
# def trial_indices(
#     meta_col: pd.Series,
#     cond1_values: Iterable[str],
#     cond2_values: Iterable[str],
# ) -> tuple[list[int], list[int]]:
#     labels = meta_col.fillna("no")
#     idx1 = [i for i, v in enumerate(labels) if v in cond1_values]
#     idx2 = [i for i, v in enumerate(labels) if v in cond2_values]
#     # Safety: no overlap
#     if set(idx1) & set(idx2):
#         raise ValueError("A trial belongs to both conditions.")
#     return idx1, idx2
#%% fun: lesser_greater
def lesser_greater(
    name_a: str, name_b: str,
    idx_a: Sequence[int], idx_b: Sequence[int],
) -> tuple[str, str, int, int, list[int], list[int]]:
    """
    less, more, n_less, n_more, idx_less, idx_more = lesser_greater(
    "Condition A", "Condition B", [0,1,2], [3,4,5,6])

    Compare two sets of trial indices and return:
      - name of the lesser condition
      - name of the greater condition
      - number of trials in lesser
      - number of trials in greater
      - trial indices of lesser
      - trial indices of greater
    """
    # Safety: no overlap
    if set(idx_a) & set(idx_b):
        raise ValueError("A trial belongs to both conditions.")
    
    n_a, n_b = len(idx_a), len(idx_b)

    # Same number of trials: return them in given order
    if n_a == n_b:
        return name_a, name_b, n_a, n_b, list(idx_a), list(idx_b)

    # Decide lesser vs greater by count
    if n_a < n_b:
        return name_a, name_b, n_a, n_b, list(idx_a), list(idx_b)
    else:
        return name_b, name_a, n_b, n_a, list(idx_b), list(idx_a)
#%% fun _extract_epoch_data
# ------------------------- core ops -------------------------
def _extract_epoch_data(epochs, 
                        trial_idx: Sequence[int], 
                        vertex_idx: Optional[Sequence[int]]) -> np.ndarray:
    """
    epochs.get_data() -> (T, V, time)
    """
    X = epochs.get_data(copy=True)[trial_idx]

    if vertex_idx is not None:
        X = X[:, vertex_idx, :]
        
    assert X.ndim == 3, "Expected (T, V, times)"

    return X
#%% fun aggregate_condition
# _aggregate data according to the plan
def _aggregate(
    X: np.ndarray,
    X_dims_list: list[str, str, str], # e.g. ["trials", "channels", "times"]
    plan: AggregationPlan,
) -> np.ndarray:
    """
    Apply reductions in plan.order on X with axes:
      trials=0, vertices=1. (time axis is preserved)
    Returns:
      - both reduced -> (time,)
      - trials only  -> (V, time)
      - vertices only-> (T, time)
    """
    # ---- validation ----
    if X.ndim != len(X_dims_list):
        raise ValueError(f"X.ndim={X.ndim} but len(X_dims_list)={len(X_dims_list)}.")
    if len(set(X_dims_list)) != len(X_dims_list):
        raise ValueError(f"X_dims has duplicates: {X_dims_list}")

    # Guard: if we’re asked to reduce channels/vertices, a reducer must be provided
    if any(step in ("channels", "vertices") for step in plan.order) and plan.vertex_reducer is None:
        raise ValueError(
            "plan.order includes 'channels'/'vertices' but plan.vertex_reducer is None. "
            "Set vertex_reducer to 'mean', 'rms', or a custom function."
        )

    out = X
    cur_dims = list(X_dims_list)  # track current axis labels as we reduce
    for step in plan.order:
        ax = cur_dims.index(step)
        

        if step == "trials":
            reducer = plan.trial_reducer
        elif step in ["vertices","channels"]:  
            reducer = plan.vertex_reducer
            if reducer is None:
                # keep this axis; don't reduce or pop
                continue
        else:
            raise ValueError(f"Unknown axis label in plan.order: {step}")

        out = _reduce(out, 
                      axis=ax, 
                      how=reducer)
        cur_dims.pop(ax)  # dimension removed → indices shift left

    return out

    #     # if step == "trials":
    #     #     out = _reduce(out, 
    #     #                   axis=ax, 
    #     #                   how=plan.trial_reducer)
    #     # elif step in ["vertices","channels"]: 
    #     #     if plan.vertex_reducer is not None:
    #     #         out = _reduce(out, 
    #     #                       axis=ax, 
    #     #                       how=plan.vertex_reducer)
    #     # else:
    #     #     raise ValueError(f"Unknown axis in plan: {step}")
    # return out

# _aggregate epochs for a given condition
def aggregate_condition(
    epochs,
    trial_idx: Sequence[int],
    plan: AggregationPlan,
    X_dims_list:list[str, str, str],
    vertex_idx: Optional[Sequence[int]] = None,
    print_info: bool = False,
) -> np.ndarray:
    
    # ---- pull a data block (T, V, time) ----
    X = _extract_epoch_data(epochs=epochs, 
                            trial_idx=trial_idx, 
                            vertex_idx=vertex_idx)
    if print_info:
        print('Extracted data block with shape ', X_dims_list, X.shape)
    
    # ---- run aggregation per plan ----
    out = _aggregate(X=X,
                    X_dims_list= X_dims_list,
                    plan=plan)
    
    # ----  checks ----
    # dimension
    if X.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (trials, vertices/channels, time); got {X.ndim}D with shape {X.shape}."
        )

    T, V, Nt = X.shape
    if T != len(trial_idx):
        raise RuntimeError(
            f"Mismatch: selected {len(trial_idx)} trials but extracted block has T={T}."
        )
    if vertex_idx is not None and V != len(vertex_idx):
        raise RuntimeError(
            f"Mismatch: selected {len(vertex_idx)} vertices/channels but extracted block has V={V}."
        )
    if Nt <= 0:
        raise RuntimeError("Time axis appears empty after extraction.")

    # ---- post-aggregation sanity checks ----
    # Determine how many axes should have been reduced
    reduced = 0
    for step in plan.order:
        if step not in ("trials", "vertices","channels"):
            raise ValueError(f"Unknown reduction axis in plan.order: {step!r}")
        reduced += 1
        
    expected_ndim = 3 - reduced
    if out.ndim != expected_ndim:
        raise RuntimeError(
            f"Unexpected output rank: got {out.ndim}D, expected {expected_ndim}D "
            f"given plan.order={plan.order}.")
       
    if out.shape[-1] != Nt:
        raise RuntimeError(
            f"Time dimension changed during aggregation: before Nt={Nt}, after Nt'={out.shape[-1]}.")
    

    return out
#%% fun subsampling_balance
def subsample_balance(
    epochs,
    idx_more: Sequence[int],
    idx_less: Sequence[int],
    plan: AggregationPlan,
    n_samples: int ,
    seed: Optional[int] ,
    epochdataa_dims_list: list[str, str, str],
    vertex_idx: Optional[Sequence[int]] = None,
    on_progress: bool = False,  # (i, n_samples) callback
) -> np.ndarray:
    """
    Balance unequal trial counts by repeated subsampling and aggregation.

    Strategy
    --------
    1) Let k = len(idx_less). Repeatedly draw k trials from idx_more (without replacement).
    2) For each draw, _aggregate with the same plan used elsewhere (order, reducers).
    3) Combine the per-draw aggregates across samples using plan.across_samples_reducer.

    Parameters
    ----------
    epochs : mne.Epochs or compatible
        Object providing .get_data() -> (n_trials, n_vertices/chans, n_times).
    idx_more : Sequence[int]
        Trial indices for the condition with MORE trials.
    idx_less : Sequence[int]
        Trial indices for the condition with FEWER trials. Determines draw size k.
    plan : AggregationPlan
        Reduction plan specifying order, trial/vertex reducers, and across-sample reducer.
    n_samples : int
        Number of subsamples (draws) to average over.
    seed : int or None
        Random seed for reproducibility (uses numpy Generator).
    vertex_idx : Sequence[int] or None, default=None
        Optional vertices/channels subset to select before aggregating.
    on_progress : 

    Returns
    -------
    np.ndarray
        The subsample-balanced _aggregate. Shape matches a single _aggregate result, e.g.:
        - (n_times,)                if trials and vertices are both reduced
        - (n_vertices, n_times)     if only trials are reduced
        - (n_trials_drawn, n_times) if only vertices are reduced (rare)

    Raises
    ------
    ValueError
        If inputs are inconsistent (e.g., k==0, n_samples<1, |idx_more|<k).
    """

    # ---------- validation ----------
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    k = len(idx_less)
    if k == 0:
        raise ValueError("idx_less is empty: need at least 1 trial to define draw size.")
    if len(idx_more) < k:
        raise ValueError(
            f"idx_more has fewer trials ({len(idx_more)}) than k={k} drawn per sample."
        )
    # ensure indices are unique ints
    u_more = np.unique(idx_more)
    u_less = np.unique(idx_less)
    if len(u_more) != len(idx_more):
        raise ValueError("idx_more contains duplicate trials.")
    if len(u_less) != len(idx_less):
        raise ValueError("idx_less contains duplicate trials.")
    if set(u_more) & set(u_less):
        raise ValueError("Overlap detected between idx_more and idx_less.")

    # ---------- RNG & arrays ----------
    rng = np.random.default_rng(seed)
    idx_more_arr = np.asarray(u_more, dtype=int)  

    # ---------- first draw (to get the output shape) ----------
    take0 = rng.choice(idx_more_arr, size=k, replace=False)
    s0 = aggregate_condition(
        epochs=epochs,
        trial_idx=take0,
        plan=plan,
        X_dims_list = epochdataa_dims_list,
        vertex_idx=vertex_idx,
        print_info= True,
    )
    

    # Preallocate [n_samples, ...shape_of_single_aggregate...]
    samples = np.empty((n_samples,) + s0.shape, dtype=float)
    samples[0] = s0
    samples_index = np.empty((n_samples,) + take0.shape, dtype=int)
    samples_index[0] = take0
    

    # ---------- remaining draws ----------
    for i in range(1, n_samples):
        take_i = rng.choice(idx_more_arr, size=k, replace=False)
        samples_index[i] = take_i
        samples[i] = aggregate_condition(
            epochs=epochs,
            trial_idx=take_i,
            plan=plan,
            X_dims_list = epochdataa_dims_list,
            vertex_idx=vertex_idx,
            
        )
        if on_progress and (i % 100 == 0 or i == n_samples - 1):
            print("subsample ",i+1,'/',n_samples)

    # ---------- combine across samples (axis=0) ----------
    balanced = _reduce(
        x=samples,
        axis=0,
        how=plan.across_samples_reducer,
    )
    
    return balanced, samples_index

#%% fun compute_summaries
# ------------------------- top-level API --------------------
def compute_summaries(
    epochs,
    epochs_metadata: pd.DataFrame,
    plan: AggregationPlan,
    trial_type_col: str,
    cond1_name: str,
    cond2_name: str,
    cond1_values: Iterable[str],
    cond2_values: Iterable[str],
    n_subsamples: int ,
    seed: int,
    epochdataa_dims_list: list[str, str, str],
    vertex_idx: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int], Dict[str, str]]:
    """
    Compute condition aggregates with/without subsampling balance.

    Parameters
    ----------
    epochs : mne.Epochs (or compatible with .get_data())
        Shape expected: (n_trials, n_vertices/channels, n_times).
    epochs_metadata : pd.DataFrame
        Must contain ``trial_type_col``.
    trial_type_col : str
        Column in metadata that encodes trial condition membership.
    cond1_name, cond2_name : str
        Human-readable names for the two conditions (e.g., "Seen", "Unseen").
    cond1_values, cond2_values : Iterable[str]
        Metadata values mapping trials to cond1/cond2 (e.g., {"seen"}, {"unseen"}).
    plan : AggregationPlan
        Defines reduction order and reducers (trials/vertices) and across-sample reducer.
    vertex_idx : optional sequence of int
        Optional selection of vertices/channels to include before aggregating.
    n_subsamples : int
        If counts differ, draw this many subsamples from the larger condition.
        Use 1 to effectively disable subsampling (fast path).
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    ct_unbalanced : dict[str, np.ndarray]
        {cond1_name: _aggregate over cond1 trials, cond2_name: _aggregate over cond2 trials}.
    ct_balanced : dict[str, np.ndarray]
        {less_name: _aggregate over lesser-count trials,
         more_name:  subsample-balanced _aggregate over greater-count trials}.
    counts : dict[str, int]
        {cond1_name: n1, cond2_name: n2}
    roles : dict[str, str]
        {"less": less_name, "more": more_name}
    """
    print("Computed summaries:",plan)
    # ----- validation -----
    # if trial_type_col not in epochs_metadata.columns:
    #     raise ValueError(f"Missing column '{trial_type_col}' in epochs_metadata.")
    if isinstance(trial_type_col, str):
        if trial_type_col not in epochs_metadata.columns:
            raise ValueError(f"Missing column '{trial_type_col}' in epochs_metadata.")
    elif  isinstance(trial_type_col, (list, tuple)):
        assert all([col in epochs_metadata.columns for col in trial_type_col])
    else:
        raise TypeError("column must be str or list of str")
    
    # ----- select trials per condition -----
    idx1 = get_trl_indices(
                        df = epochs_metadata,
                        column = trial_type_col, 
                        condition= cond1_values)
    idx2 = get_trl_indices(
                    df = epochs_metadata,
                    column = trial_type_col, 
                    condition= cond2_values)
    # Safety: no overlap
    if set(idx1) & set(idx2):
        raise ValueError("A trial belongs to both conditions.")
    assert len(idx1) + len(idx2) == epochs.get_data().shape[0], "Selected trials not matching epochs."
    
    # idx1, idx2 = trial_indices(
    #     meta_col=epochs_metadata[trial_type_col],
    #     cond1_values=cond1_values,
    #     cond2_values=cond2_values,
    # )
    print('based on ',trial_type_col,
          f"found {len(idx1)} trials for '{cond1_name}",
          f"and {len(idx2)} for '{cond2_name}'."
            )

    # Who has fewer/more trials?
    less_name, more_name, n_less, n_more, idx_less, idx_more = lesser_greater(
        name_a=cond1_name, 
        name_b=cond2_name, 
        idx_a=idx1, 
        idx_b=idx2
    )

    # Counts for reference/reporting
    counts = {cond1_name: len(idx1), 
              cond2_name: len(idx2)}
    roles  = {"less": less_name, 
              "more": more_name}
    
    # ----- unbalanced aggregates per condition -----
    ct_unbalanced = {
        cond1_name: aggregate_condition(
            epochs=epochs, 
            trial_idx=idx1, 
            plan=plan, 
            X_dims_list = epochdataa_dims_list,
            vertex_idx=vertex_idx,
            print_info= True
        ),
        cond2_name: aggregate_condition(
            epochs=epochs, 
            trial_idx=idx2, 
            plan=plan, 
            X_dims_list = epochdataa_dims_list,
            vertex_idx=vertex_idx,
            print_info= True
        ),
    }

    # Sanity (shouldn’t happen because lesser_greater enforces n_more >= n_less)
    if n_more < n_less:
        raise RuntimeError(f"Invariant violated (n_more < n_less): {n_more} < {n_less}")

    # ----- balanced result for the 'more' side (if needed) -----
    if n_more == n_less or n_subsamples <= 1:
        # Equal counts or asked to skip subsampling → just _aggregate once.
        print("Skipping subsampling",f"({n_more} vs {n_less} trials)")
        agg_more_balanced = aggregate_condition(
            epochs=epochs, 
            trial_idx=idx_more, 
            plan=plan, 
            X_dims_list = epochdataa_dims_list,
            vertex_idx=vertex_idx,
            print_info= True
        )
        subsampling_index= None
    else:
        # Subsample the larger set down to k = n_less, repeat, then combine.
        print("subsampling",f"({n_more} vs {n_less} trials), {n_subsamples} samples")
        agg_more_balanced,subsampling_index = subsample_balance(
            epochs=epochs,
            idx_more=idx_more,
            idx_less=idx_less,
            plan=plan,
            n_samples=n_subsamples,
            seed=seed,
            epochdataa_dims_list= epochdataa_dims_list,
            vertex_idx=vertex_idx,
            on_progress= True,  
        )

    # Always compute the 'less' side directly (no subsampling needed there)
    agg_less = aggregate_condition(
        epochs=epochs, 
        trial_idx=idx_less, 
        plan=plan, 
        X_dims_list = epochdataa_dims_list,
        vertex_idx=vertex_idx,
        print_info= True
    )

    ct_balanced = {less_name: agg_less, 
                   more_name: agg_more_balanced}
    
    
    return ct_unbalanced, ct_balanced, counts, roles,subsampling_index

#%% fun compute_all_combinations
# ---  helper to combine info to create a key ---
def _reducer_str(how):
    if how is None:
        return ""  # unused when None
    return how.__name__ if callable(how) else str(how)

def make_plan_name(order, trial_reducer, vertex_reducer):
    """
    Build names like:
      T_mean_V_mean, T_rms, T_mean_C_rms, etc.
    If vertex_reducer is None, omit the V_/C_ part entirely.
    """
    parts = []
    for axis in order:
        if axis == "trials":
            parts.append(f"T_{_reducer_str(trial_reducer)}")
        elif axis in ("vertices", "channels"):
            if vertex_reducer :
                prefix = "V_" if axis == "vertices" else "C_"
                parts.append(f"{prefix}{_reducer_str(vertex_reducer)}")
        else:
            raise ValueError(f"Unknown axis: {axis}")

    return "_".join(parts) or "raw"  # fallback if everything was kept


def compute_all_combinations(
    epochs,
    epochs_metadata: pd.DataFrame,
    order: Tuple[Axis, ...],
    trial_type_col: str,
    cond1_name: str,
    cond2_name: str,
    cond1_values: Sequence[str],
    cond2_values: Sequence[str],
    n_subsamples: int,
    seed: int ,
    epochdataa_dims_list: Optional[list[str, str, str]] = ["trials", "channels", "times"],
    vertex_idx: Optional[Sequence[int]] = None,
    reducers: Tuple[Sequence, Sequence] = (("mean", "rms"), ("mean", "rms")),  # (trial_opts, vertex_opts)

):
    """
    Runs every combo of {mean,rms} over trials x {mean,rms} over vertices,
    with and without subsampling on the larger condition.

    Returns
    -------
    results : dict
        {
          ( 'trial=mean', 'vert=mean', 'subsample=no' ): {
              'unbalanced': {cond1: arr, cond2: arr},
              'balanced':   {less: arr, more: arr},   # if subsample=no, == unbalanced when counts equal
              'counts':     {cond1: n1, cond2: n2},
              'roles':      {'less': name, 'more': name},
              'shapes':     {'unbalanced': ( ... ), 'balanced': ( ... )}
          },
          ...
        }
    """
    # generate options for reducers based on order    
    trial_opts_all, vertex_opts_all = reducers

    def generate_reduce_options_from_order(order,trial_opts_all,vertex_opts_all):
        s = set(order)
        trial_opts  = list(trial_opts_all)  if "trials" in s else [None]
        vertex_opts = list(vertex_opts_all) if s & {"vertices", "channels"} else [None]
        return trial_opts, vertex_opts

    trial_opts, vertex_opts = generate_reduce_options_from_order(order,
                                                                 trial_opts_all,vertex_opts_all)


    
    # loop over all combinations of trial/vertex reducers
    results = {}
    for trial_reducer, vertex_reducer in product(trial_opts, 
                                                 vertex_opts):
        print('---','compute_all_combinations',
              f"trial_reducer={trial_reducer}, vertex_reducer={vertex_reducer}"
              )
       
        
        plan_name = make_plan_name(order, 
                                   trial_reducer, 
                                   vertex_reducer
                                   )
        plan = AggregationPlan(
            order=order,
            trial_reducer=trial_reducer,
            vertex_reducer=vertex_reducer,
            across_samples_reducer="mean",
        )

        # Compute condition aggregates with/without subsampling balance.
        # also handles the “if equal counts, skip subsampling” case.
        ct_unbal, ct_bal, counts, roles,subsampling_index = compute_summaries(
            epochs=epochs,
            epochs_metadata=epochs_metadata,
            trial_type_col=trial_type_col,
            cond1_name=cond1_name,
            cond2_name=cond2_name,
            cond1_values=cond1_values,
            cond2_values=cond2_values,
            plan=plan,
            epochdataa_dims_list = epochdataa_dims_list,
            vertex_idx=vertex_idx,
            n_subsamples=n_subsamples,  #if do_subsample else 1, 1 = effectively "no subsampling"
            seed=seed,
        )

        # Shapes (for quick sanity checks)
        # When order=("trials","vertices") both reduced -> shape is (n_times,)
        shape_unbal = {k: v.shape for k, v in ct_unbal.items()}
        shape_bal = {k: v.shape for k, v in ct_bal.items()}

        results[plan_name] = {
            "unbalanced": ct_unbal,     # dict by condition names
            "balanced":   ct_bal,       # dict by "less"/"more" names after balancing
            "subsampling_index": subsampling_index, # index of subsampling
            "counts":     counts,       # {cond1: n1, cond2: n2}
            "shapes":     {               # shapes of the arrays
                "unbalanced": shape_unbal,
                "balanced":   shape_bal,},
            "roles":      roles,        # {"less": name, "more": name}
            "plan":       plan,         # keep for traceability
            "plan_name":  plan_name,    # readable name
        }

    return results



# -------------------------
# below are functions for 3 conditions

# Reuse your existing bits:
#  - _reduce(how={"mean","rms"} or callable)
#  - AggregationPlan(order, trial_reducer, vertex_reducer, across_samples_reducer)
#  - _extract_epoch_data(epochs, trial_idx, vertex_idx) -> (T, V, time)
#  - _aggregate(X, X_dims_list, plan)
#  - aggregate_condition(epochs, trial_idx, plan, X_dims_list, vertex_idx, print_info)
# -------------------------
#%% fun three: rank_by_size3
# ---------- rank like lesser_greater, but for 3 ----------
def rank_by_size3(
    name_a: str, idx_a: Sequence[int],
    name_b: str, idx_b: Sequence[int],
    name_c: str, idx_c: Sequence[int],
) -> tuple[
    str, str, str, int, int, int, list[int], list[int], list[int]
]:
    """
    Rank three conditions by trial count (ascending), with stable tie behavior.

    Parameters
    ----------
    name_a, name_b, name_c : str
        Human-readable names for the three conditions (e.g., "Seen", "Unseen", "Scrambled").
    idx_a, idx_b, idx_c : Sequence[int]
        Trial indices for each condition. They must be mutually disjoint.

    Returns
    -------
    less_name, mid_name, more_name : str
        Names ordered by count (ascending). Ties are broken by the original input order.
    n_less, n_mid, n_more : int
        Corresponding trial counts.
    idx_less, idx_mid, idx_more : list[int]
        Corresponding trial indices (lists).

    Raises
    ------
    ValueError
        If any overlap exists between the index sets.
    """
    A, B, C = set(idx_a), set(idx_b), set(idx_c)
    overlap = (A & B) | (A & C) | (B & C)
    if overlap:
        raise ValueError(f"A trial belongs to multiple conditions: indices {sorted(overlap)}")

    # Stable sort by length → preserves input order for ties
    items = [
        (name_a, list(idx_a)),
        (name_b, list(idx_b)),
        (name_c, list(idx_c)),
    ]
    items.sort(key=lambda x: len(x[1]))

    (less_name, less_idx), (mid_name, mid_idx), (more_name, more_idx) = items
    return (less_name, mid_name, more_name,
            len(less_idx), len(mid_idx), len(more_idx),
            less_idx, mid_idx, more_idx)

#%% fun three: subsample_balance_to_k
# ---------- general subsampling to k (works for any single condition) ----------
def subsample_balance_to_k(
    epochs,
    trial_idx_all: Sequence[int],
    k: int,
    plan: 'AggregationPlan',
    n_samples: int,
    seed: Optional[int],
    X_dims_list: list[str, str, str],
    vertex_idx: Optional[Sequence[int]] = None,
    on_progress: bool = False,
):
    """
    Subsample a single condition down to k trials, repeat n_samples times, then combine.

    Strategy
    --------
    1) Draw k distinct trials from `trial_idx_all` (without replacement).
    2) For each draw, compute `aggregate_condition(...)`.
    3) Combine the per-draw aggregates across the sample axis (axis=0) using
       `plan.across_samples_reducer` (e.g., "mean").

    Parameters
    ----------
    epochs : mne.Epochs or compatible
        Must implement ``.get_data() -> (n_trials, n_vertices/channels, n_times)``.
    trial_idx_all : Sequence[int]
        All available trials for the condition to be downsampled.
        Must contain unique integers.
    k : int
        Target number of trials per draw (k <= len(trial_idx_all)).
    plan : AggregationPlan
        Reduction recipe (order + reducers + across_samples_reducer).
    n_samples : int
        Number of independent draws to average. Use larger values for stability.
    seed : int or None
        Reproducible RNG seed (NumPy Generator). Use None for non-deterministic runs.
    X_dims_list : list[str, str, str]
        Axis names for the data fed to `_aggregate`, typically
        ``["trials", "channels" or "vertices", "times"]``.
    vertex_idx : Optional[Sequence[int]]
        Optional vertex/channel subset to select **before** aggregation.
    on_progress : bool
        If True, prints progress occasionally during the loop.

    Returns
    -------
    balanced : np.ndarray
        Aggregated result after combining across `n_samples`.
        Shape matches a single `aggregate_condition` output, e.g.:
        - (n_times,) if both trials and vertices are reduced
        - (n_vertices, n_times) if only trials are reduced
        - (k, n_times) if only vertices are reduced
    samples_index : np.ndarray
        Array of shape (n_samples, k) with the exact trial indices drawn per sample.

    Raises
    ------
    ValueError
        If k <= 0, if duplicates exist, or if there are fewer than k trials available.
    """
    if k <= 0:
        raise ValueError("k must be >= 1.")

    # Unique and validate cardinality
    u = np.unique(trial_idx_all)
    if len(u) < k:
        raise ValueError(f"Not enough trials to draw k={k} (have {len(u)}).")
    if len(u) != len(trial_idx_all):
        raise ValueError("trial_idx_all contains duplicate trials.")

    rng = np.random.default_rng(seed)

    # First draw to infer output shape
    take0 = rng.choice(u, size=k, replace=False)
    s0 = aggregate_condition(
        epochs=epochs,
        trial_idx=take0,
        plan=plan,
        X_dims_list=X_dims_list,
        vertex_idx=vertex_idx,
        print_info=False,
    )

    # Preallocate [n_samples, ...shape_of_single_aggregate...]
    samples = np.empty((n_samples,) + s0.shape, dtype=float)
    samples[0] = s0
    samples_index = np.empty((n_samples, k), dtype=int)
    samples_index[0] = take0

    # Remaining draws
    for i in range(1, n_samples):
        take_i = rng.choice(u, size=k, replace=False)
        samples_index[i] = take_i
        samples[i] = aggregate_condition(
            epochs=epochs,
            trial_idx=take_i,
            plan=plan,
            X_dims_list=X_dims_list,
            vertex_idx=vertex_idx,
            print_info=False,
        )
        if on_progress and (i % 100 == 0 or i == n_samples - 1):
            print("subsample", i + 1, "/", n_samples)

    # Combine across samples (axis=0)
    balanced = _reduce(samples, axis=0, how=plan.across_samples_reducer)
    return balanced, samples_index

#%% fun three: compute_summaries_three
# ---------- 3-condition summaries ----------
def compute_summaries_three(
    epochs,
    epochs_metadata: pd.DataFrame,
    plan: 'AggregationPlan',
    trial_type_col: str,
    cond1_name: str,
    cond2_name: str,
    cond3_name: str,
    cond1_values: Iterable[str],
    cond2_values: Iterable[str],
    cond3_values: Iterable[str],
    n_subsamples: int,
    seed: Optional[int],
    epochdataa_dims_list: list[str, str, str],
    vertex_idx: Optional[Sequence[int]] = None,
):
    """
    Compute unbalanced and balanced (to the global minimum k) aggregates for 3 conditions.

    Balancing rule
    --------------
    Let (n1, n2, n3) be the trial counts and k = min(n1, n2, n3).
    For each condition:
      - If n_i == k: aggregate once on its full set (no subsampling).
      - If n_i > k and n_subsamples > 1: repeatedly subsample to k, then combine.
      - If n_i > k and n_subsamples <= 1: draw a single k-subset and aggregate.

    Parameters
    ----------
    epochs : mne.Epochs or compatible
        Provides the data for aggregation.
    epochs_metadata : pd.DataFrame
        Must contain the `trial_type_col` that maps trials to condition labels.
    plan : AggregationPlan
        Reduction plan (order + trial/vertex reducers + across-sample reducer).
    trial_type_col : str
        Name of the metadata column encoding condition labels.
    cond1_name, cond2_name, cond3_name : str
        Human-friendly display names for the three conditions.
    cond1_values, cond2_values, cond3_values : Iterable[str]
        Label value sets defining each condition's membership.
    n_subsamples : int
        Number of subsamples to average when downsampling (if needed).
        Set to 1 to effectively disable repeated subsampling (single draw).
    seed : int or None
        RNG seed for reproducibility; None for non-deterministic draws.
    epochdataa_dims_list : list[str, str, str]
        Axis labels for your data fed into `_aggregate`, e.g.
        ``["trials", "channels", "times"]`` or ``["trials", "vertices", "times"]``.
    vertex_idx : Optional[Sequence[int]]
        Optional selection of vertices/channels prior to aggregation.

    Returns
    -------
    ct_unbalanced : dict[str, np.ndarray]
        Per-condition aggregates computed on the full (unbalanced) sets.
    ct_balanced : dict[str, np.ndarray]
        Per-condition aggregates after balancing each to k trials.
    counts : dict[str, int]
        Trial counts per condition: {cond1_name: n1, cond2_name: n2, cond3_name: n3}.
    ranks : dict[str, str]
        Ranking by size: {"less": name, "mid": name, "more": name}.
    subsampling_index : dict[str, np.ndarray or None]
        For each condition, either:
          - an array of shape (n_subsamples, k) with drawn trial indices, or
          - None (if no draw was needed).

    Raises
    ------
    ValueError
        If `trial_type_col` is missing or if any condition has zero trials.
    """
    X_dims_list= epochdataa_dims_list
     
    if isinstance(trial_type_col, str):
        if trial_type_col not in epochs_metadata.columns:
            raise ValueError(f"Missing column '{trial_type_col}' in epochs_metadata.")
    elif  isinstance(trial_type_col, (list, tuple)):
        
        assert all([col in epochs_metadata.columns for col in trial_type_col])
    else:
        raise TypeError("column must be str or list of str")
    

   


    # Indices per condition
    idx1 = get_trl_indices(
                        df = epochs_metadata,
                        column = trial_type_col, 
                        condition= cond1_values)
    idx2 = get_trl_indices(
                    df = epochs_metadata,
                    column = trial_type_col, 
                    condition= cond2_values)
    idx3 = get_trl_indices(
                    df = epochs_metadata,
                    column = trial_type_col, 
                    condition= cond3_values)
    # Safety: no overlaps across pairs
    s1, s2, s3 = set(idx1), set(idx2), set(idx3)
    overlap = (s1 & s2) | (s1 & s3) | (s2 & s3)
    if overlap:
        raise ValueError(f"A trial belongs to multiple conditions: indices {sorted(overlap)}")

    k = min(len(idx1), len(idx2), len(idx3))
    # Guard: zero-trial conditions are not meaningful for balancing
    if k == 0:
        zeros = [nm for nm, idx in [(cond1_name, idx1), (cond2_name, idx2), (cond3_name, idx3)] if len(idx) == 0]
        raise ValueError(f"Some conditions have zero trials: {zeros}")

    # Rank for reporting (stable on ties)
    (less_name, mid_name, more_name,
     n_less, n_mid, n_more,
     idx_less, idx_mid, idx_more) = rank_by_size3(
        cond1_name, idx1,
        cond2_name, idx2,
        cond3_name, idx3
     )
    # Balance all to the minimum
    counts = {cond1_name: len(idx1),
              cond2_name: len(idx2),
              cond3_name: len(idx3)}
    ranks = {"less": less_name, "mid": mid_name, "more": more_name}

    # --- unbalanced aggregates (full sets) ---
    ct_unbalanced = {
        cond1_name: aggregate_condition(epochs, idx1, plan, X_dims_list, vertex_idx, print_info=False),
        cond2_name: aggregate_condition(epochs, idx2, plan, X_dims_list, vertex_idx, print_info=False),
        cond3_name: aggregate_condition(epochs, idx3, plan, X_dims_list, vertex_idx, print_info=False),
    }

    # --- balanced to k per condition ---
    ct_balanced: Dict[str, np.ndarray] = {}
    subsampling_index: Dict[str, Optional[np.ndarray]] = {}

    for name, idxs in [(cond1_name, idx1), (cond2_name, idx2), (cond3_name, idx3)]:
        n_i = len(idxs)
        if n_i == k:
            # Already at the minimum → no subsampling
            ct_balanced[name] = aggregate_condition(
                epochs=epochs,
                trial_idx=idxs,
                plan=plan,
                X_dims_list=X_dims_list,
                vertex_idx=vertex_idx,
                print_info=False,
            )
            subsampling_index[name] = None
        elif n_subsamples <= 1:
            # Single draw path (no repetition)
            rng = np.random.default_rng(seed)
            take = rng.choice(idxs, size=k, replace=False)
            ct_balanced[name] = aggregate_condition(
                epochs=epochs,
                trial_idx=take,
                plan=plan,
                X_dims_list=X_dims_list,
                vertex_idx=vertex_idx,
                print_info=False,
            )
            # Store actual drawn indices (shape: (1, k))
            subsampling_index[name] = np.array([take], dtype=int)
        else:
            # Repeated draws + combine
            agg_k, samp_idx = subsample_balance_to_k(
                epochs=epochs,
                trial_idx_all=idxs,
                k=k,
                plan=plan,
                n_samples=n_subsamples,
                seed=seed,
                X_dims_list=X_dims_list,
                vertex_idx=vertex_idx,
                on_progress=False,
            )
            ct_balanced[name] = agg_k
            subsampling_index[name] = samp_idx

    return ct_unbalanced, ct_balanced, counts, ranks, subsampling_index

#%% fun three: compute_all_combinations_three
# ---------- compute all combinations (3 conditions) ----------
def compute_all_combinations_three(
    epochs,
    epochs_metadata: pd.DataFrame,
    order:  Tuple[Axis, ...],
    trial_type_col: str,
    cond1_name: str, cond2_name: str, cond3_name: str,
    cond1_values: Sequence[str], cond2_values: Sequence[str], cond3_values: Sequence[str],
    n_subsamples: int,
    seed: Optional[int],
    epochdataa_dims_list: list[str, str, str] = ["trials", "channels", "times"],
    vertex_idx: Optional[Sequence[int]] = None,
    reducers: Tuple[Sequence, Sequence] = (("mean", "rms"), ("mean", "rms")),  # (trial_opts, vertex_opts)

):
    """
    Grid over reducer combinations (trial x vertex), compute 3-condition summaries for each.

    Parameters
    ----------
    epochs, epochs_metadata : see `compute_summaries_three`.
    order : tuple of {"trials","vertices"/"channels"}
        Reduction order passed to `AggregationPlan`. Time is preserved implicitly.
        Examples: `("trials",)`, `("trials","vertices")`, `("vertices","trials")`.
    trial_type_col : str
        Metadata column name with condition labels.
    cond1_name, cond2_name, cond3_name : str
        Condition display names.
    cond1_values, cond2_values, cond3_values : Sequence[str]
        Label sets defining each condition.
    n_subsamples : int
        Number of repeated subsamples when balancing (if needed).
    seed : int or None
        RNG seed for reproducible draws across all plans.
    epochdataa_dims_list : list[str, str, str], default=["trials","channels","times"]
        Axis names for your data block fed to `_aggregate`.
    vertex_idx : Optional[Sequence[int]]
        Optional vertex/channel subset prior to aggregation.
    reducers : tuple of (Sequence, Sequence), default=(("mean","rms"),("mean","rms"))
        Reducers to consider for (trial, vertex) axes separately.
        Examples:
          - (["mean"], ["rms"]) → force trial=mean, vertex=rms
          - (["mean","rms"], ["mean"]) → 2 x 1 grid
          - (["mean","rms"], ["mean","rms"]) → full 2 x 2 grid
    Returns
    -------
    results : dict[str, dict]
        Dictionary keyed by a readable `plan_name` (e.g. "T_mean_V_rms"), each containing:
          - "unbalanced": {cond1: arr, cond2: arr, cond3: arr}
          - "balanced":   {cond1: arr_k, cond2: arr_k, cond3: arr_k}
          - "subsampling_index": {cond: (n_samples,k) array or None}
          - "counts": {cond1: n1, cond2: n2, cond3: n3}
          - "ranks": {"less": name, "mid": name, "more": name}
          - "shapes": {"unbalanced": {cond: shape}, "balanced": {cond: shape}}
          - "plan": AggregationPlan instance
          - "plan_name": str

    Notes
    -----
    - Reducer options are automatically inferred from `order`:
        if "trials" in order → trial_reducer in {"mean","rms"}
        if {"vertices","channels"} ∩ order → vertex_reducer in {"mean","rms"}
        otherwise the corresponding reducer is `None` (axis kept).
    """

    trial_opts_all, vertex_opts_all = reducers

    def generate_reduce_options_from_order(order,trial_opts_all,vertex_opts_all):
        s = set(order)
        trial_opts  = list(trial_opts_all)  if "trials" in s else [None]
        vertex_opts = list(vertex_opts_all) if s & {"vertices", "channels"} else [None]
        return trial_opts, vertex_opts

    trial_opts, vertex_opts = generate_reduce_options_from_order(order,
                                                                 trial_opts_all,vertex_opts_all)



    results = {}
    for trial_reducer, vertex_reducer in product(trial_opts, vertex_opts):
        plan = AggregationPlan(
            order=order,
            trial_reducer=trial_reducer,
            vertex_reducer=vertex_reducer,
            across_samples_reducer="mean",
        )
        print('-----','compute_all_combinations_three',
              f"trial_reducer={trial_reducer}, vertex_reducer={vertex_reducer}"
              )
       
        # Build a readable plan name like your make_plan_name
        def _reducer_str(how):
            if how is None:
                return ""
            return how.__name__ if callable(how) else str(how)

        parts = []
        for ax in order:
            if ax == "trials":
                parts.append(f"T_{_reducer_str(trial_reducer)}")
            elif ax in ("vertices", "channels"):
                if vertex_reducer:
                    prefix = "V_" if ax == "vertices" else "C_"
                    parts.append(f"{prefix}{_reducer_str(vertex_reducer)}")
        plan_name = "_".join([p for p in parts if p]) or "raw"

        ct_unbal, ct_bal, counts, ranks, subs_idx = compute_summaries_three(
            epochs=epochs,
            epochs_metadata=epochs_metadata,
            plan=plan,
            trial_type_col=trial_type_col,
            cond1_name=cond1_name, cond2_name=cond2_name, cond3_name=cond3_name,
            cond1_values=cond1_values, cond2_values=cond2_values, cond3_values=cond3_values,
            n_subsamples=n_subsamples,
            seed=seed,
            epochdataa_dims_list=epochdataa_dims_list,
            vertex_idx=vertex_idx,
        )

        results[plan_name] = {
            "unbalanced": ct_unbal,
            "balanced":   ct_bal,
            "subsampling_index": subs_idx,
            "counts":     counts,
            "ranks":      ranks,    # {"less": ..., "mid": ..., "more": ...}
            "shapes": {
                "unbalanced": {k: v.shape for k, v in ct_unbal.items()},
                "balanced":   {k: v.shape for k, v in ct_bal.items()},
            },
            "plan":      plan,
            "plan_name": plan_name,
        }

    return results

#%% fun two: subsample_balance_conditions
def subsample_balance_conditions(
    cond1_tlt: np.ndarray,
    cond2_tlt: np.ndarray,
    n_samples: int,
    trial_reducer: Literal["mean", "median", "rms"],
    seed: Optional[int] = None,
    sample_reducer: Literal["mean", "median", "rms"] = "mean",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Balance two conditions via subsampling and return both direct and balanced results.
    
    Returns BOTH:
    1. Direct aggregation (all trials, unbalanced)
    2. Subsampled aggregation (balanced trial counts)
    
    Parameters
    ----------
    cond1_tlt, cond2_tlt : np.ndarray, shape (n_trials, n_times)
        Data matrices for two conditions (trials × timepoints)
    n_samples : int
        Number of repetitions for subsamples from larger condition
    trial_reducer : {"mean", "median", "rms"}
        How to aggregate across trials
    seed : int or None
        Random seed for reproducibility
    sample_reducer : {"mean", "median", "rms"}, default="mean"
        How to combine subsamples
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    t_cond1_unbalanced : np.ndarray, shape (n_times,)
        Condition 1 aggregated directly (all trials)
    t_cond2_unbalanced : np.ndarray, shape (n_times,)
        Condition 2 aggregated directly (all trials)
    t_cond1_balanced : np.ndarray, shape (n_times,)
        Condition 1 after subsampling balance
    t_cond2_balanced : np.ndarray, shape (n_times,)
        Condition 2 after subsampling balance
    info : dict
        Keys: 'n_trials_cond1', 'n_trials_cond2', 'subsampled_condition', 
              'n_trials_used', 'sample_indices'
    """
    
    print(f"subsample_balance_conditions with repeat {n_samples} randseed {seed}",
          '\n', f"input shapes: {cond1_tlt.shape} vs {cond2_tlt.shape} ", 
          datetime.now())
    
    # Validate inputs
    if cond1_tlt.ndim != 2 or cond2_tlt.ndim != 2:
        raise ValueError("Both inputs must be 2D (trials × time)")
    
    n1, nt1 = cond1_tlt.shape
    n2, nt2 = cond2_tlt.shape
    
    if nt1 != nt2:
        raise ValueError(f"Time mismatch: {nt1} vs {nt2}")
    if n1 == 0 or n2 == 0:
        raise ValueError("Both conditions need ≥1 trial")
    if n_samples < 1:
        raise ValueError("n_samples must be ≥1")
    
    # Setup reducers
    def get_reducer(name: str):
        if name == "mean":
            return np.mean
        elif name == "median":
            return np.median
        elif name == "rms":
            return lambda x, axis: np.sqrt(np.mean(x**2, axis=axis))
        else:
            raise ValueError(f"Unknown reducer: {name}")
    
    trial_fn = get_reducer(trial_reducer)
    sample_fn = get_reducer(sample_reducer)
    
    # Direct aggregation (all trials, no balancing)
    t_cond1_unbalanced = trial_fn(cond1_tlt, axis=0)
    t_cond2_unbalanced = trial_fn(cond2_tlt, axis=0)
    
    # Info dict
    info = {
        'n_trials_cond1': n1,
        'n_trials_cond2': n2,
        'n_trials_used': min(n1, n2),
        'subsampled_condition': None,
        'sample_indices': None,
    }
    
    # Equal trials - no subsampling needed
    if n1 == n2:
        if verbose:
            print(f"Equal trials ({n1}) - no subsampling")
        info['subsampled_condition'] = 'neither'
        return t_cond1_unbalanced, t_cond2_unbalanced, t_cond1_unbalanced, t_cond2_unbalanced, info
    
    # Determine which to subsample
    if n1 > n2:
        larger, smaller = cond1_tlt, cond2_tlt
        k, subsample_name = n2, 'cond1'
    else:
        larger, smaller = cond2_tlt, cond1_tlt
        k, subsample_name = n1, 'cond2'
    
    if verbose:
        print(f"Subsampling {subsample_name}: {larger.shape[0]} → {k} trials, {n_samples} samples")
    
    # Subsample the larger condition
    rng = np.random.default_rng(seed)
    subsamples = np.empty((n_samples, nt1))
    sample_indices = np.empty((n_samples, k), dtype=int)
    
    for i in range(n_samples):
        idx = rng.choice(larger.shape[0], size=k, replace=False)
        sample_indices[i] = idx
        subsamples[i] = trial_fn(larger[idx], axis=0)
        
        if verbose and (i % 100 == 0 or i == n_samples - 1):
            print(f"  {i+1}/{n_samples}")
    
    larger_balanced = sample_fn(subsamples, axis=0)
    smaller_balanced = trial_fn(smaller, axis=0)
    
    # Package results in correct order
    if n1 > n2:
        t_cond1_balanced, t_cond2_balanced = larger_balanced, smaller_balanced
    else:
        t_cond1_balanced, t_cond2_balanced = smaller_balanced, larger_balanced
    
    info['subsampled_condition'] = subsample_name
    info['sample_indices'] = sample_indices
    
    return t_cond1_unbalanced, t_cond2_unbalanced, t_cond1_balanced, t_cond2_balanced, info

#%% fun plot: _plot_tlt_subject_subplots
def _plot_tlt_subject_subplots(subtlt_use, 
                           times_use, 
                           title,
                           n_cols=6, 
                           trial_alpha=0.05,
                           trial_color='grey',
                           flag_plot_mean=True,
                           flag_plot_rms=True,
                           mean_linewidth=2,
                           rms_linewidth=2,
                           linewidth=1,
                           y_suptitle=1.01,
                           vlines=None,
                           vlines_colors=None,
                           vlines_linestyle=None,
                           vlines_linewidth=None,
                           hlines=None,
                           hlines_colors=None,
                           hlines_linestyle=None,
                           hlines_linewidth=None,
                           baseline_window=None):
    """
    Plot each participant's trial time series in separate subplots.

    Parameters
    ----------
    subtlt_use : np.ndarray
        Array of shape (num_subjects, num_trials, num_time_points).
    times_use : np.ndarray
        1D array of length num_time_points, the x-axis values.
    title : str
        Main title for the figure.
    n_cols : int, optional
        Number of subplot columns (default: 6).
    trial_alpha : float, optional
        Transparency of individual trial lines (default: 0.05).
    trial_color : str, optional
        Color for trial lines (default: 'grey').
    flag_plot_mean : bool, optional
        If True, plot mean across trials as red dashed line (default: True).
    flag_plot_rms : bool, optional
        If True, plot RMS across trials as blue dashed line (default: True).
    mean_linewidth : float, optional
        Width of mean line (default: 2).
    rms_linewidth : float, optional
        Width of RMS line (default: 2).
    linewidth : float, optional
        Width of trial lines (default: 1).
    y_suptitle : float, optional
        Vertical position of suptitle (default: 1.01).
    vlines : float or list, optional
        Vertical line positions.
    vlines_colors : list, optional
        Colors for vertical lines.
    vlines_linestyle : list, optional
        Line styles for vertical lines.
    vlines_linewidth : list, optional
        Line widths for vertical lines.
    hlines : float or list, optional
        Horizontal line positions.
    hlines_colors : list, optional
        Colors for horizontal lines.
    hlines_linestyle : list, optional
        Line styles for horizontal lines.
    hlines_linewidth : list, optional
        Line widths for horizontal lines.
    baseline_window : tuple of float, optional
        Time window (start, end) to use as baseline for mean and RMS correction.
        If provided, the mean of each metric within this window is subtracted
        from the entire time series, bringing the baseline to near zero.
        Example: (-0.2, 0) for 200ms pre-stimulus baseline.
    """
    num_subjects, num_trials, num_time_points = subtlt_use.shape
    
    # Validate that times_use matches the time dimension
    if len(times_use) != num_time_points:
        raise ValueError(
            f"Length of times_use ({len(times_use)}) does not match "
            f"the time dimension of subtlt_use ({num_time_points})"
        )
    
    print(f"Plotting {num_subjects} subjects with {num_trials} trials, "
          f"{num_time_points} time points each.")
    
    # Get baseline indices if baseline_window is provided
    baseline_indices = None
    if baseline_window is not None:
        baseline_start, baseline_end = baseline_window
        baseline_indices = np.where((times_use >= baseline_start) & 
                                   (times_use <= baseline_end))[0]
        if len(baseline_indices) == 0:
            print(f"Warning: No time points found in baseline window {baseline_window}")
            baseline_indices = None
        else:
            print(f"Using baseline window {baseline_window} "
                  f"({len(baseline_indices)} time points)")

    # Subplot grid
    n_rows = int(np.ceil(num_subjects / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows),
        sharex=True, sharey=True
    )
    # Ensure axes is always iterable (handle single subplot case)
    if num_subjects == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    def _add_line(ax, lines, colors, linestyles, linewidths, line_type, 
                  default_color='k', default_linestyle='-', default_linewidth=1):
        """Helper function to add lines of a given type (vertical or horizontal)."""
        if lines is not None:
            lines = [lines] if not isinstance(lines, (list, tuple)) else lines
            for idx, line in enumerate(lines):
                color = colors[idx] if colors and idx < len(colors) else default_color
                linestyle = linestyles[idx] if linestyles else default_linestyle
                linewidth = linewidths[idx] if linewidths else default_linewidth
                line_func = ax.axvline if line_type == 'v' else ax.axhline
                line_func(line, color=color, linestyle=linestyle, linewidth=linewidth)

    for subj_idx in range(num_subjects):
        ax = axes[subj_idx]
        
        # Plot individual trials with alpha
        for trial_idx in range(num_trials):
            trial_data = subtlt_use[subj_idx, trial_idx, :]
            ax.plot(times_use, trial_data, 
                   color=trial_color, 
                   alpha=trial_alpha, 
                   linewidth=linewidth)
        
        # Plot mean if requested
        if flag_plot_mean:
            mean_data = np.mean(subtlt_use[subj_idx, :, :], axis=0)
            
            # Apply baseline correction if specified
            if baseline_indices is not None:
                baseline_mean = np.mean(mean_data[baseline_indices])
                mean_data = mean_data - baseline_mean
            
            ax.plot(times_use, mean_data, 
                   color='blue', 
                   linestyle='-', 
                   linewidth=mean_linewidth,
                   label='Mean' if subj_idx == 0 else '')
        
        # Plot RMS if requested
        if flag_plot_rms:
            rms_data = np.sqrt(np.mean(subtlt_use[subj_idx, :, :]**2, axis=0))
            
            # Apply baseline correction if specified
            if baseline_indices is not None:
                baseline_rms = np.mean(rms_data[baseline_indices])
                rms_data = rms_data - baseline_rms
            
            ax.plot(times_use, rms_data, 
                   color='red', 
                   linestyle='-', 
                   linewidth=rms_linewidth,
                   label='RMS' if subj_idx == 0 else '')
        
        ax.set_title(f"Subj{subj_idx+1}")
        
        # Add vertical lines
        _add_line(ax, vlines, vlines_colors, vlines_linestyle, 
                 vlines_linewidth, 'v')
        
        # Add horizontal lines
        _add_line(ax, hlines, hlines_colors, hlines_linestyle, 
                 hlines_linewidth, 'h')
    
    # Remove empty panels
    for j in range(num_subjects, n_rows * n_cols):
        fig.delaxes(axes[j])

    # Add legend if mean or RMS is plotted
    if flag_plot_mean or flag_plot_rms:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(title, y=y_suptitle)
    plt.tight_layout()
    plt.show()
    return fig, axes

# %% fun: merge_windows
# # Merge overlapping/adjacent windows
def merge_windows(window_list):
    if not window_list:
        return []
    
    # Sort by start time
    sorted_windows = sorted(window_list, key=lambda x: x[0])
    
    merged = [sorted_windows[0]]
    
    for current in sorted_windows[1:]:
        last = merged[-1]
        
        # If current overlaps or is adjacent to last, merge them
        if current[0] <= last[1]:  # Overlapping or touching
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # Gap exists, start new window
            merged.append(current)
    
    return merged
# %%


