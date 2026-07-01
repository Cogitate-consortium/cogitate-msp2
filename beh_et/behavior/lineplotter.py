import numpy as np
import gc
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import data_saver

"""
This module is in charge of plotting line-plots. It has the capability to plot up to 2 datasets containing line data
(reflecting an average of some parameter) and (if we want) this data's SE (to plot as "sleeves"). For example, one
set can be some "average" data, and the second one can be the same data's moving average calulation - so we can plot
both of these on the same plot. 
The module also saves the plot to the appropriate place in the BIDS-compatible hierarchy. 

@author: RonyHirsch
"""

W = 10
H = 7.5
DPI = 1000

BUFFER = 0.1
MOV_AVG = "Moving Average"
JITTER_WIDTH = 0.08

F_HEADER = 16
F_AXES_TITLE = 14
F_HORIZ_LINES = 11
XLABELPAD = 20
YLABELPAD = 20


def plot_avg_line(title, trial_df_list, avg_col_list, se_col_list=None, label_list=None, x_name="Trial", y_name="",
                  color_list=None, x_tick_intervals=4, significance_bars_dict=None,
                  save=False, save_name="", save_path="", sub_folder=""):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    # x axis
    trials_num = [t.shape[0] for t in trial_df_list]
    trials = [np.arange(0, n, 1) for n in trials_num]  # this will be the x axis

    # plot Y boundary range
    minimal = 1000000000
    maximal = -100000000
    if se_col_list is not None:
        for i in range(len(se_col_list)):
            if se_col_list[i] is not None:
                trial_df_list[i][se_col_list[i]].fillna(0, inplace=True)
                lower = [x - se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])]
                upper = [x + se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])]
            else:
                lower = trial_df_list[i][avg_col_list[i]]
                upper = trial_df_list[i][avg_col_list[i]]
            lowest = min(lower)
            highest = max(upper)
            minimal = min(minimal, lowest)
            maximal = max(maximal, highest)

    else:
        for i in range(len(avg_col_list)):
            if avg_col_list[i] is not None:
                lower = trial_df_list[i][avg_col_list[i]]
                upper = trial_df_list[i][avg_col_list[i]]
                lowest = min(lower)
                highest = max(upper)
                minimal = min(minimal, lowest)
                maximal = max(maximal, highest)

    # colors
    if color_list is None:
        colors = sns.color_palette("colorblind", len(avg_col_list))
    else:
        colors = color_list

    # plot
    for i in range(len(trial_df_list)):
        plt.plot(trials[i], trial_df_list[i][avg_col_list[i]], ls='-', color=colors[i], label=label_list[i])
        if se_col_list is not None:  # if data has SE
            if se_col_list[i] is not None:
                plt.fill_between(trials[i],
                                 [x - se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])],
                                 [x + se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])],
                                 alpha=.2, color=colors[i])

    # do we want to add gray bars that denote significance
    if significance_bars_dict is not None:
        for k in significance_bars_dict.keys():
            for x_start, x_end in significance_bars_dict[k]["x"]:
                xbar = [x for x in range(int(x_start), int(x_end) + 1)]
                ybar = [significance_bars_dict[k]["y"]] * len(xbar)
                plt.plot(xbar, ybar, color="gray", linewidth=6, alpha=0.5)

    min_y = 0
    max_y = 1
    if minimal < min_y:
        min_y = minimal
    if maximal > max_y:
        max_y = maximal
    min_y = round(min_y, 1)
    max_y = round(max_y, 1)
    plt.ylim([min_y - BUFFER, max_y + BUFFER])
    #yticks_labels = [f"{x * 100:.0f}%" for x in np.arange(min_y, max_y + BUFFER, 0.2)]
    #plt.yticks(np.arange(min_y, max_y + BUFFER, 0.2), yticks_labels)

    yticks_labels = [f"{x * 100:.0f}%" for x in np.arange(min_y, max_y + BUFFER, 0.1)]
    plt.yticks(np.arange(min_y, max_y + BUFFER, 0.1), yticks_labels)
    plt.yticks(fontsize=14)

    plt.xlim([0, trial_df_list[0].shape[0]])
    x_tick_div = int(max(trials_num) / x_tick_intervals)
    plt.xticks(np.arange(0, max(trials_num) + x_tick_intervals, x_tick_div))

    plot_title = f"{title.title()}"
    plt.title(plot_title, fontsize=F_HEADER)
    plt.xlabel(x_name, fontsize=F_AXES_TITLE, labelpad=XLABELPAD)
    plt.ylabel(y_name, fontsize=F_AXES_TITLE, labelpad=YLABELPAD)
    plt.legend()

    if save:
        folder_path = os.path.join(save_path, sub_folder)
        data_saver.create_dir(folder_path)
        if not save_name:
            plot_save_name = plot_title
        else:
            plot_save_name = save_name
        plot_save_name = plot_save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
        data_saver.safe_save(folder_path, f"LINE_{plot_save_name}.png")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(folder_path, f"LINE_{plot_save_name}.png"), dpi=DPI)
        plt.savefig(os.path.join(folder_path, f"LINE_{plot_save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def plot_individual_lines(dataframe_list, x_col, y_col, labels, x_name, y_name, title, color_list=None, save=False,
                          save_name="", save_path=""):
    # each item in the dataframe_list is a line which will be plotted
    plt.clf()
    plt.figure()
    sns.reset_orig()
    # colors
    if color_list is None:
        colors = sns.color_palette("colorblind", len(dataframe_list))
    else:
        colors = color_list

    for i in range(len(dataframe_list)):
        df = dataframe_list[i]
        plt.plot(df[x_col], df[y_col], ls='-', color=colors[i], label=labels[i])

    plot_title = f"{title.title()}"
    plt.title(plot_title, fontsize=F_HEADER)
    plt.xlabel(x_name, fontsize=F_AXES_TITLE, labelpad=XLABELPAD)
    plt.ylabel(y_name, fontsize=F_AXES_TITLE, labelpad=YLABELPAD)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

    if save:
        if not save_name:
            plot_save_name = plot_title
        else:
            plot_save_name = save_name
        plot_save_name = plot_save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(save_path, f"LINE_{plot_save_name}.png"), dpi=DPI, bbox_inches="tight")
        plt.savefig(os.path.join(save_path, f"LINE_{plot_save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')
    plt.close()
    return


def face_coloring(row, hollow_circles_column, hue, hue_order, palette):
    if row[hollow_circles_column] == 1:  # if passed test - circle not hollow
        result = palette[hue_order.index(row[hue])]
    else:  # hollow circle
        result = 'white'
    return result


def edge_coloring(row, hue, hue_order, palette):
    result = palette[hue_order.index(row[hue])]
    return result


def plot_scatter(title, data, x_col, y_col, x_name, y_name, hue=None, hue_order=None, palette=None,
                 save=False, save_name="", save_path="", line=False, hollow_circles_column=None, x_percent=True,
                 x_mapping=None, y_percent=True, jitter=False):
    plt.clf()
    plt.figure()
    sns.reset_orig()

    data_c = data.copy()
    del data

    if x_mapping is not None:
        data_c[x_col] = data_c[x_col].map(x_mapping)

    if jitter:
        jitters = list(np.random.rand(len(data_c[x_col])) * (JITTER_WIDTH * 1.5) / 2.)
        x = [list(data_c[x_col])[i] + jitters[i] * random.choice([-1, 1]) for i in range(len(jitters))]
    else:
        x = data_c[x_col]

    if hue is not None:
        if hollow_circles_column is not None:
            data_c.loc[:, 'face_colors'] = data_c.apply(lambda row: face_coloring(row, hollow_circles_column, hue, hue_order, palette), axis=1)
            data_c.loc[:, 'edge_colors'] = data_c.apply(lambda row: edge_coloring(row, hue, hue_order, palette), axis=1)
            plt.scatter(x=x, y=data_c[y_col], facecolors=data_c['face_colors'], edgecolors=data_c['edge_colors'])
        else:
            data_c.loc[:, 'colors'] = data_c.apply(lambda row: edge_coloring(row, hue, hue_order, palette), axis=1)
            plt.scatter(x=x, y=data_c[y_col], facecolors=data_c['colors'], edgecolors=data_c['colors'])
    else:
        plt.scatter(x=x, y=data_c[y_col])

    if hue_order is not None:
        legend_names = list()
        for i in range(len(hue_order)):
            g_name = hue_order[i]
            g_col = palette[i]
            if g_col not in data_c['colors'].unique():
                continue
            pop_g = mpatches.Patch(color=g_col, label=g_name)
            legend_names.append(pop_g)

    if line:  # if we want to add the best-fit line to this scatter
        idx = np.isfinite(data_c[x_col]) & np.isfinite(data_c[y_col])
        try:
            m, b = np.polyfit(data_c[x_col][idx], data_c[y_col][idx], 1)  # m=slope, b=intercept
            plt.plot(data_c[x_col], m * data_c[x_col] + b, color="black", linewidth=0.5, alpha=0.7)
        except Exception:
            print(f"Failed to converge; either too few subjects or NaNs in columns: {x_col} / {y_col}, check your data")

    plot_title = f"{title}"
    plt.title(plot_title, fontsize=F_HEADER+5)
    plt.xlabel(x_name, fontsize=F_AXES_TITLE+3, labelpad=XLABELPAD-3)
    plt.ylabel(y_name, fontsize=F_AXES_TITLE+3, labelpad=YLABELPAD-3)

    ticks = np.arange(-0.1, 1.1, 0.1).tolist()
    labels = [""] + [f"{100 * x:.0f}%" for x in ticks[1:-1]] + [""]
    if x_percent:
        plt.xlim([-0.1, 1])
        plt.xticks(ticks, labels, fontsize=F_AXES_TITLE+3)
        plt.axhline(y=0, xmin=-0.1, xmax=1, color="lightgrey")
        plt.axvline(x=0, ymin=-0.1, ymax=1, color="lightgrey")
    elif x_mapping is not None:
        plt.xlim([min(list(x_mapping.values())) - BUFFER, max(list(x_mapping.values())) + BUFFER])
        plt.xticks(ticks=sorted(list(x_mapping.values())), labels=sorted(list(x_mapping.keys())), fontsize=F_AXES_TITLE + 3)
    else:
        plt.xticks(sorted(data_c[x_col].unique()), sorted(data_c[x_col].unique()), fontsize=F_AXES_TITLE + 3)

    if y_percent:
        plt.ylim([-0.1, 1])
        plt.yticks(ticks, labels, fontsize=F_AXES_TITLE+3)
    else:
        ticks = np.arange(0, round(max(data_c[y_col]) + BUFFER, 1), 5 * BUFFER).tolist()
        labels = [f"{x:.1f}" for x in ticks]
        plt.yticks(ticks, labels, fontsize=F_AXES_TITLE + 3)

    if hue_order is not None:
        plt.legend(handles=legend_names, prop={'size': F_AXES_TITLE})

    if save:
        if not save_name:
            plot_save_name = plot_title
        else:
            plot_save_name = save_name
        plot_save_name = plot_save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(save_path, f"SCATTER_{plot_save_name}.png"), dpi=DPI, bbox_inches="tight")

    del data_c
    plt.close()
    gc.collect()
    plt.close()
    return

