import pandas as pd
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import data_saver
import data_reader

"""
This module, given a cross-lab dataframe which contains at least one of the following data columns:
hit (TP), false alarm (FA) , miss (FN),correct rejection (TN), and can plot this data in 1 of 2 ways:
1. box plot (plotter method) : creates both a boxplot with TP,FA,FN,TN stats across all subjects, and a scatterplot 
in which each subject is represented by a dot, and the color represents the lab the subject had the experiment in.
There is an option to add a line connecting dots which belong to the same subject. 
2. raincloud plot (make_it_rain method) : creates a raincloud plot (by combining a half-violin, scatter and boxplots)

The module assumes each subject is represented as a dataframe row which at the very least includes a lab column, 
and at least one rate column (TP/FA/FN/TN). A subject who does not have this data (for example, if this plot is
used for behavioral screening checks and there are subjects who don't have a behavioral screening session) will be
omitted from the plot.

This module is called by using the "plot" method. According to whether "raincloud" parameter is True/False, the correct
plotting function is called (make_it_rain/plotter). 

NOTE ABOUT THE BOXPLOT'S NOTCHES: 
Notched box plots apply a "notch" or narrowing of the box around the median. 
Notches are useful in offering a rough guide to significance of difference of medians; if the notches of two boxes do 
not overlap, this offers evidence of a statistically significant difference between the medians. The width of the 
notches is proportional to the interquartile range of the sample and inversely proportional to the square root of the 
size of the sample. However, there is uncertainty about the most appropriate multiplier (as this may vary depending on 
the similarity of the variances of the samples). One convention is to use +/-1.58*IQR/sqrt(n). 
If we have this weird "flipped" appearance in the notched box plots, it means that the 1st quartile has a lower 
value than the confidence of the mean and vice versa for the 3rd quartile. A
lthough it looks ugly, it's actually useful information about the (un)confidence of the median. 

@author: RonyHirsch """

# taken from here: https://twcf-arc.slab.com/posts/institutional-abbreviations-rsi4obcd
LAB_DICT = {'SA': 'Birmingham', 'SB': 'Peking', 'SC': 'Donders', 'SD': 'Yale', 'SE': 'Harvard', 'SF': 'NYU',
            'SG': 'Madison', 'SX': 'MPI', 'SY': 'Reed', 'SZ': 'TAU'}

# the amount of (horizontal) jitter to create in the plot so that points with the same %hit/fa rate won't overlap
JITTER_WIDTH = 0.08
VIOLIN_OFFSET = 0.15

F_HEADER = 16
F_AXES_TITLE = 14
F_HORIZ_LINES = 11
XLABELPAD = 20
YLABELPAD = 20

# names for the plot title anx axes
TP = 'True Positive'
FP = 'False Positive'
TN = 'True Negative'
FN = 'False Negative'

W = 10
H = 7.5
DPI = 1000


def plot_histogram(data, column_name, plot_title, plot_x_label, save_path, save_name, plot_y_label="Percent",
                   num_of_bins=20, bins=None, hist_shape=True, color="darkgray", vertical_line=None,
                   vertical_line_color="deeppink", xmin=0.5, xmax=0.605, xstep=0.1, ystep=1, y_max=None,
                   stat_type="percent", hue=None, hue_order=None, palette=None, starting_alpha=0.5, reducing_alpha=0.2):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()
    b = bins if bins is not None else num_of_bins
    if hue is None:
        sns.histplot(data=data, x=column_name, bins=b, stat=stat_type, kde=hist_shape, color=color, legend=False)
    else:
        alpha = starting_alpha
        for i in range(len(hue_order)):
            specific = data[data[hue] == hue_order[i]]
            if specific.empty:
                continue
            color = palette[i]
            plt.hist(specific[column_name], bins=b, alpha=alpha, color=color, label=hue_order[i])
            del specific
            #alpha = alpha - reducing_alpha
        plt.legend(prop={'size': F_AXES_TITLE})

    del data
    gc.collect()

    if vertical_line:
        plt.axvline(vertical_line, ls='--', lw=5, color=vertical_line_color)

    if xmin and xmax and xstep:
        plt.xticks(np.arange(xmin, xmax, xstep), fontsize=F_AXES_TITLE + 5)
    elif bins is not None:
        plt.xticks(np.arange(min(bins), max(bins) + xstep/2, xstep), fontsize=F_AXES_TITLE + 5)
    else:
        plt.xticks(fontsize=F_AXES_TITLE + 5)
    if y_max is None:
        ymax = plt.gca().get_ylim()[1]
    else:
        ymax = y_max
    plt.yticks(np.arange(0, ymax + (ystep/2), ystep), fontsize=F_AXES_TITLE + 5)
    plt.title(plot_title, fontsize=F_HEADER + 15)
    plt.xlabel(plot_x_label, fontsize=F_AXES_TITLE + 15, labelpad=XLABELPAD)
    plt.ylabel(plot_y_label, fontsize=F_AXES_TITLE + 15, labelpad=XLABELPAD)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W + 5, H + 5)
    plt.savefig(os.path.join(save_path, f"hist_{save_name}.png"), dpi=DPI)
    plt.savefig(os.path.join(save_path, f"hist_{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()
    return


def plot_barh(data, plot_title, plot_x_label, plot_y_label, save_path, save_name):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    palette = sns.color_palette("Blues_r", data.shape[1])
    sns.barplot(data=data, orient='h', palette=palette)

    plt.xticks(fontsize=F_AXES_TITLE + 5)
    plt.yticks(fontsize=F_AXES_TITLE + 5)
    plt.title(plot_title, fontsize=F_HEADER + 10)
    plt.xlabel(plot_x_label, fontsize=F_AXES_TITLE + 10, labelpad=XLABELPAD)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W + 12, H + 2.5)
    plt.savefig(os.path.join(save_path, f"barh_{save_name}.png"), dpi=DPI)
    plt.clf()
    plt.close()
    return


def confusion_matrix(model, X_test, y_test, save_path, save_name, labels=None, display_labels=None):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    if labels is None:
        plot_confusion_matrix(model, X_test, y_test)
    else:
        plot_confusion_matrix(model, X_test, y_test, labels=labels, display_labels=display_labels)
    plt.title("Confusion Matrix", fontsize=F_HEADER + 10)
    plt.ylabel("True Label", fontsize=F_AXES_TITLE + 7, labelpad=YLABELPAD)
    plt.xlabel("Predicted Label", fontsize=F_AXES_TITLE + 7, labelpad=XLABELPAD)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W + 4, H)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=DPI)
    plt.clf()
    plt.close()
    return


def raincloud_multi_source(data, id_col, id_color_dict, data_cols_list, data_cols_names_dict, plot_title,
                           plot_x_label, plot_y_label, sub_line, save_plot,
                           save_path, save_name, alpha_col=None, alpha_mapping=None, initial_alpha=1,
                           horizontal_lines=None, alpha_step=0.4, legend_show=True):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    alpha = initial_alpha
    for group_id in id_color_dict.keys():  # sub group of all data
        group_data = data[data[id_col] == group_id]
        if group_data.empty:
            continue
        group_color = id_color_dict[group_id]
        x_list = list()
        y_list = list()
        alpha_list = list()
        num = 1
        for i in range(len(data_cols_list)):  # each data column
            group_data_c = group_data[data_cols_list[i]]
            if alpha_col is not None:
                group_data_alpha = group_data[alpha_col]
            # we must get rid of nans:
            # first, if an entire column is empty (no data from any subject), skip it as we have no data
            if group_data_c.isnull().all():
                num += 1
                scat_x = [np.nan] * len(group_data_c)
                x_list.append(scat_x)
                y_list.append(group_data_c.tolist())
                continue
            # then, some subjects might not have specific data (missing session etc) - get rid of that
            # ELSE 0 : OTHERWISE THE MISSING DATA WILL MESS THE PLOT UP
            group_data_c = [x if not (np.isnan(x)) else 0 for x in group_data_c]
            # violin plot
            violin = plt.violinplot(group_data_c, positions=[num], showmeans=False, showextrema=False, showmedians=False)
            for part in violin['bodies']:
                part.set_alpha(alpha)
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                try:
                    b.set_color(group_color)
                except ValueError:
                    group_color = [group_color]
                    b.set_color(group_color)

            # then scatter
            scat_x = (np.ones(len(group_data_c)) * num) + VIOLIN_OFFSET + (np.random.rand(len(group_data_c)) * (JITTER_WIDTH * 1.5) / 2.)
            if alpha_col is None:
                plt.scatter(x=scat_x, y=group_data_c, marker="o", color=group_color)
            else:
                face_colors = [group_color if x == 1 else 'white' for x in group_data_alpha]  # failed QC will be hollow dots
                alphas = [alpha_mapping[x] for x in group_data_alpha]
                plt.scatter(x=scat_x, y=group_data_c, marker="o", facecolor=face_colors, edgecolors=group_color)#, alpha=alphas)

            # complete with a boxplot
            plt.boxplot(group_data_c, positions=[num + VIOLIN_OFFSET], notch=True, medianprops=dict(color=group_color),
                        showfliers=False, boxprops=dict(color=group_color), whiskerprops=dict(color=group_color),
                        capprops=dict(color=group_color))

            # for cases we'll want to add subject lines
            x_list.append(scat_x)
            y_list.append(group_data_c)
            if alpha_col is not None:
                alpha_list.append(alphas)

            num += 1

        # if we want to plot subject line
        if sub_line is not None:  # add lines between dots that indicate the same subject
            for line in sub_line:  # each item in sub_line is a line between a the number of dots we want
                for sub in range(len(x_list[0])):
                    # if there's absolutely no data for this line
                    if np.isnan(x_list[line[0]]).all() or np.isnan(x_list[line[1]]).all():
                        break
                    else:
                        m_list = [x_list[l][sub] for l in line]
                        n_list = [y_list[l][sub] for l in line]
                        if alpha_col is not None:
                            a = float([alpha_list[l][sub] for l in line][0])  # one line = one alpha
                            l = 0.5 if a == 1.0 else 0.45
                            c = "darkgray" if a == 1.0 else "lightgray"
                            plt.plot(m_list, n_list, color=c, linewidth=l, alpha=a)
                        else:
                            plt.plot(m_list, n_list, color="darkgray", linewidth=0.5)

        # between groups
        alpha -= alpha_step

    # figure in general
    # set y axis limit and range
    if horizontal_lines is not None:
        plt.ylim([0, 1.15])
    else:
        plt.ylim([0, 1])
    yticks_labels = [""]
    yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, 1.01, 0.1)])
    yticks_labels.append("")
    if horizontal_lines is not None:
        plt.yticks(np.arange(-0.1, 1.09, 0.1), yticks_labels[:-1])
    else:
        plt.yticks(np.arange(-0.1, 1.1, 0.1), yticks_labels)

    # if we want a horizontal line for some explanation
    if horizontal_lines is not None:
        for line_dict in horizontal_lines:  # every element in this list is a horizontal line information
            if line_dict['y'] is not None:
                y_plot = line_dict['y']
            else:
                y_plot = 1.05
            plt.hlines(y=y_plot, xmin=line_dict['xmin'], xmax=line_dict['xmax'], color='grey', linestyle='-',
                        linewidth=1)
            plt.text(x=(line_dict['xmin'] + line_dict['xmax']) / 2, y=y_plot + 0.03, s=line_dict['label'],
                        fontsize=F_HORIZ_LINES+5, va='center', ha='center')

    lim = num - 0.5
    plt.xlim([0.5, lim])
    plt.xticks(ticks=[(n + 1) for n in range(num - 1)], labels=[data_cols_names_dict[data_cols_list[i]] for i in range(len(data_cols_list))], fontsize=F_AXES_TITLE+5)
    plt.title(plot_title, fontsize=F_HEADER+10)
    plt.ylabel(plot_y_label, fontsize=F_AXES_TITLE+5, labelpad=YLABELPAD-2)
    plt.xlabel(plot_x_label, fontsize=F_AXES_TITLE+5, labelpad=XLABELPAD-2)

    if legend_show:
        existing_labs = {k: id_color_dict[k] for k in id_color_dict if k in data[id_col].unique()}
        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in existing_labs.values()]
        plt.legend(markers, existing_labs.keys(), numpoints=1, prop={'size': F_AXES_TITLE}, bbox_to_anchor=(1.01, 1.0), loc='upper left')

    if save_plot:
        figure = plt.gcf()  # get current figure
        figure.tight_layout()
        figure.set_size_inches(W+10, H+3)
        plt.savefig(os.path.join(save_path, f"RAINCLOUD_{save_name}.png"), dpi=DPI)
        plt.savefig(os.path.join(save_path, f"RAINCLOUD_{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')

    #plt.show()
    return


def make_it_rain(data, cols, col_names, color_list, plot_title, plot_x_label, plot_y_label, sub_line, horizontal_lines,
                 save_plot, save_path, save_folder, save_name, custom_ylim=None, skip=None):
    x_list = list()
    y_list = list()
    num = 1
    for i in range(len(cols)):
        data_c = data[cols[i]]
        # we must get rid of nans:
        # first, if an entire column is empty (no data from any subject), skip it as we have no data
        if data_c.isnull().all():
            num += 1
            continue
        # then, some subjects might not have specific data (missing session etc) - get rid of that
        # ELSE 0 : OTHERWISE THE MISSING DATA WILL MESS THE PLOT UP
        data_c = [x if not (np.isnan(x)) else 0 for x in data_c]
        # violin plot
        violin = plt.violinplot(data_c, positions=[num], showmeans=False, showextrema=False, showmedians=False)
        # make it a half-violin plot (only to the LEFT of center)
        for b in violin['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(color_list[i])

        # then scatter
        scat_x = (np.ones(len(data_c)) * num) + VIOLIN_OFFSET + (np.random.rand(len(data_c)) * JITTER_WIDTH / 2.)
        plt.scatter(x=scat_x, y=data_c, marker="o", color=color_list[i], alpha=0.5, edgecolor=color_list[i], s=45, zorder=2)

        # complete with a boxplot
        plt.boxplot(data_c, positions=[num + VIOLIN_OFFSET], notch=False, medianprops=dict(color='black'),
                    showfliers=False)

        # for cases we'll want to add subject lines
        x_list.append(scat_x)
        y_list.append(data_c)

        num += 1

    # if we want to plot subject line
    if sub_line is not None:  # add lines between dots that indicate the same subject
        for line in sub_line:  # each item in sub_line is a line between a the number of dots we want
            for sub in range(len(x_list[0])):
                m_list = [x_list[l][sub] for l in line]
                n_list = [y_list[l][sub] for l in line]
                plt.plot(m_list, n_list, color="darkgray", linewidth=0.5, zorder=1)

    # set y axis limit and range
    if horizontal_lines is not None:
        plt.ylim([0, 1.15])
    else:
        if custom_ylim is not None:
            plt.ylim([0, custom_ylim])
        else:
            plt.ylim([0, 1])
    yticks_labels = [""]
    if custom_ylim is not None:
        upper_lim = custom_ylim + skip
        yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, upper_lim, skip)])
    else:
        yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, 1.01, 0.1)])
    yticks_labels.append("")

    if horizontal_lines is not None:
        plt.yticks(np.arange(-0.1, 1.09, 0.1), yticks_labels[:-1])
    else:
        if custom_ylim is not None:
            lower_lim = 0 - skip
            plt.yticks(np.arange(lower_lim, upper_lim, skip), yticks_labels)
        else:
            plt.yticks(np.arange(-0.1, 1.1, 0.1), yticks_labels)

    # if we want a horizontal line for some explanation
    if horizontal_lines is not None:
        for line_dict in horizontal_lines:  # every element in this list is a horizontal line information
            if line_dict['y'] is not None:
                y_plot = line_dict['y']
            else:
                y_plot = 1.05
            plt.hlines(y=y_plot, xmin=line_dict['xmin'], xmax=line_dict['xmax'], color='grey', linestyle='-', linewidth=1)
            plt.text(x=(line_dict['xmin']+line_dict['xmax'])/2, y=y_plot+0.03, s=line_dict['label'],
                     fontsize=F_HORIZ_LINES, va='center', ha='center')

    lim = num - 0.5
    plt.xlim([0.5, lim])
    plt.xticks([(n + 1) for n in range(num - 1)], col_names, fontsize=F_AXES_TITLE-1)
    plt.yticks(fontsize=F_AXES_TITLE - 1)
    plt.title(plot_title, fontsize=F_HEADER+5, pad=YLABELPAD-2)
    plt.ylabel(plot_y_label.title(), fontsize=F_AXES_TITLE+3, labelpad=YLABELPAD-1)
    plt.xlabel(plot_x_label.title(), fontsize=F_AXES_TITLE+3, labelpad=XLABELPAD-2)

    if save_plot:
        folder_path = os.path.join(save_path, save_folder)
        data_saver.create_dir(folder_path)
        # make sure it is saved in the right sub-folder
        save_name = save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
        data_saver.safe_save(folder_path, f"RAINCLOUD_{save_name}.png")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(folder_path, f"RAINCLOUD_{save_name}.png"), dpi=DPI)
        plt.savefig(os.path.join(folder_path, f"RAINCLOUD_{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')

    return


def rain_per_mod(data_all, cols, col_names, color_dict, plot_title, plot_x_label, plot_y_label, sub_line, horizontal_lines,
                 save_plot, save_path, save_folder, save_name, custom_ylim=None, skip=None):
    labels = list()
    label_colors = list()
    for mod in [data_reader.FMRI, data_reader.MEEG]:
        data = data_all[data_all[data_reader.MODALITY] == mod]
        x_list = list()
        y_list = list()
        num = 1
        for i in range(len(cols)):
            labels.append(f"{mod} {col_names[i]}")
            label_colors.append(color_dict[mod][i])
            data_c = data[cols[i]]
            # we must get rid of nans:
            # first, if an entire column is empty (no data from any subject), skip it as we have no data
            if data_c.isnull().all():
                num += 1
                continue
            # then, some subjects might not have specific data (missing session etc) - get rid of that
            # ELSE 0 : OTHERWISE THE MISSING DATA WILL MESS THE PLOT UP
            data_c = [x if not (np.isnan(x)) else 0 for x in data_c]
            # violin plot
            violin = plt.violinplot(data_c, positions=[num], showmeans=False, showextrema=False, showmedians=False)
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(color_dict[mod][i])
                b.set_alpha(0.35)

            # then scatter
            scat_x = (np.ones(len(data_c)) * num) + VIOLIN_OFFSET + (np.random.rand(len(data_c)) * JITTER_WIDTH / 2.)
            plt.scatter(x=scat_x, y=data_c, marker="o", color=color_dict[mod][i], alpha=0.35,
                        edgecolor=color_dict[mod][i], s=45, zorder=2)

            # complete with a boxplot
            plt.boxplot(data_c, positions=[num + VIOLIN_OFFSET], notch=False, medianprops=dict(color=color_dict[mod][i], linewidth=3),
                        showfliers=False, boxprops=dict(color=color_dict[mod][i], linewidth=3),
                        whiskerprops=dict(color=color_dict[mod][i], linewidth=3), capprops=dict(color=color_dict[mod][i], linewidth=3),
                        zorder=3)

            # for cases we'll want to add subject lines
            x_list.append(scat_x)
            y_list.append(data_c)

            num += 1

        # if we want to plot subject line
        if sub_line is not None:  # add lines between dots that indicate the same subject
            for line in sub_line:  # each item in sub_line is a line between a the number of dots we want
                for sub in range(len(x_list[0])):
                    m_list = [x_list[l][sub] for l in line]
                    n_list = [y_list[l][sub] for l in line]
                    plt.plot(m_list, n_list, color="darkgray", linewidth=0.4, zorder=0)

    # set y axis limit and range
    if horizontal_lines is not None:
        plt.ylim([0, 1.15])
    else:
        if custom_ylim is not None:
            plt.ylim([0, custom_ylim])
        else:
            plt.ylim([0, 1])
    yticks_labels = [""]
    if custom_ylim is not None:
        upper_lim = custom_ylim + skip
        yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, upper_lim, skip)])
    else:
        yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, 1.01, 0.1)])
    yticks_labels.append("")

    if horizontal_lines is not None:
        plt.yticks(np.arange(-0.1, 1.09, 0.1), yticks_labels[:-1])
    else:
        if custom_ylim is not None:
            lower_lim = 0 - skip
            plt.yticks(np.arange(lower_lim, upper_lim, skip), yticks_labels)
        else:
            plt.yticks(np.arange(-0.1, 1.1, 0.1), yticks_labels)

    # if we want a horizontal line for some explanation
    if horizontal_lines is not None:
        for line_dict in horizontal_lines:  # every element in this list is a horizontal line information
            if line_dict['y'] is not None:
                y_plot = line_dict['y']
            else:
                y_plot = 1.05
            plt.hlines(y=y_plot, xmin=line_dict['xmin'], xmax=line_dict['xmax'], color='grey', linestyle='-', linewidth=1)
            plt.text(x=(line_dict['xmin']+line_dict['xmax'])/2, y=y_plot+0.03, s=line_dict['label'],
                     fontsize=F_HORIZ_LINES, va='center', ha='center')

    lim = num - 0.5
    plt.xlim([0.5, lim])
    plt.xticks([(n + 1) for n in range(num - 1)], col_names, fontsize=F_AXES_TITLE-1)
    plt.yticks(fontsize=F_AXES_TITLE - 1)
    plt.title(plot_title, fontsize=F_HEADER+5, pad=YLABELPAD-2)
    plt.ylabel(plot_y_label.title(), fontsize=F_AXES_TITLE+3, labelpad=YLABELPAD-1)
    plt.xlabel(plot_x_label.title(), fontsize=F_AXES_TITLE+3, labelpad=XLABELPAD-2)
    for i, label in enumerate(labels):
        plt.plot([], [], color=label_colors[i], label=label)
    plt.legend(title="")

    if save_plot:
        folder_path = os.path.join(save_path, save_folder)
        data_saver.create_dir(folder_path)
        # make sure it is saved in the right sub-folder
        save_name = save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
        data_saver.safe_save(folder_path, f"RAINCLOUD_{save_name}.png")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(folder_path, f"RAINCLOUD_{save_name}.png"), dpi=DPI)
        plt.savefig(os.path.join(folder_path, f"RAINCLOUD_{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight')

    return


def plotter(data, lab_names, sub_names, colors, colors_per_lab, data_col_order: list(), data_name_order: dict(), scatter,
            sub_line, plot_title, plot_x_label, plot_y_label, save_plot, save_path, ga_folder):
    """
    Regular BoxPlot, with / without scatter dots (individual subjects) and lines between data points belonging to the
    same sub. Params are like "Plot" function, see documentation there.
    """
    # plot the box plot
    num = 1
    final_labels = []
    for i in range(len(data_col_order)):
        data_c = data[data_col_order[i]]
        # we must get rid of nans since some subjects won't have this data (missing session, or no screening)
        # ELSE 0 : OTHERWISE THE MISSING DATA WILL MESS THE PLOT UP
        data_c = [x if not (np.isnan(x)) else 0 for x in data_c]
        plt.boxplot(data_c, positions=[num], notch=True, medianprops=dict(color='black'), showfliers=False)
        final_labels.append(data_name_order[data_col_order[i]])
        num += 1

    if scatter:  # if we also want a scatter in which each point is a subject
        if colors_per_lab == 1:  # color is per lab not per subject
            lab_colors = {lab_names[i]: colors[i] for i in range(len(lab_names))}
            for i in range(len(list(lab_names))):
                lab = lab_names[i]
                lab_label = LAB_DICT[lab]
                lab_data = data[data['Lab'] == lab]
                lab_color = lab_colors[lab]
                lab_flag = False
                x_list = []
                y_list = []
                ind = 1
                for j in range(len(data_col_order)):
                    x = np.ones(lab_data[data_col_order[j]].shape[0]) * ind + \
                        (np.random.rand(lab_data[data_col_order[j]].shape[0]) * JITTER_WIDTH - JITTER_WIDTH / 2.)
                    if lab_flag or lab_data[data_col_order[j]].isnull().all():
                        l = ""
                    else:
                        l = lab_label
                        lab_flag = True
                    plt.scatter(x=x, y=lab_data[data_col_order[j]], marker="o", color=lab_color, label=l)
                    x_list.append(x)
                    y_list.append(list(lab_data[data_col_order[j]]))
                    ind += 1

                if sub_line:  # add lines between dots that indicate the same subject
                    for k in range(len(x_list[0])):
                        m_list = [x[k] for x in x_list]
                        n_list = [y[k] for y in y_list]
                        plt.plot(m_list, n_list, color=lab_color, linewidth=0.5)

        else:  # color each subject independently
            sub_colors = {sub_names[i]: colors[i] for i in range(len(sub_names))}
            for i in range(len(list(sub_names))):
                sub = sub_names[i]
                sub_data = data[data['Subject'] == sub]
                sub_color = sub_colors[sub]
                lab_flag = False
                x_list = []
                y_list = []
                ind = 1
                for j in range(len(data_col_order)):
                    x = np.ones(sub_data[data_col_order[j]].shape[0]) * ind + \
                        (np.random.rand(sub_data[data_col_order[j]].shape[0]) * JITTER_WIDTH - JITTER_WIDTH / 2.)
                    plt.scatter(x=x, y=sub_data[data_col_order[j]], marker="o", color=sub_color, label="")
                    x_list.append(x)
                    y_list.append(list(sub_data[data_col_order[j]]))
                    ind += 1

                if sub_line:  # add lines between dots that indicate the same subject
                    for k in range(len(x_list[0])):
                        m_list = [x[k] for x in x_list]
                        n_list = [y[k] for y in y_list]
                        plt.plot(m_list, n_list, color=sub_color, linewidth=0.5)

    plt.ylim([0, 1])
    yticks_labels = [""]
    yticks_labels.extend([f"{x * 100:.0f}%" for x in np.arange(0.0, 1.01, 0.1)])
    yticks_labels.append("")
    plt.yticks(np.arange(-0.1, 1.1, 0.1), yticks_labels)
    lim = num - 0.5
    plt.xlim([0.5, lim])
    plt.xticks([(n + 1) for n in range(num - 1)], final_labels)
    if colors_per_lab == 1:
        plt.legend()
    plt.title(f"{plot_title}", fontsize=F_HEADER)
    plt.ylabel(plot_y_label, fontsize=F_AXES_TITLE, labelpad=YLABELPAD)
    plt.xlabel(plot_x_label, fontsize=F_AXES_TITLE, labelpad=XLABELPAD)

    if save_plot:
        folder_path = os.path.join(save_path, ga_folder)  # make sure it is saved in the right sub-folder
        data_saver.create_dir(folder_path)
        # make sure there aren't dangerous stuff the prevent normal filesave
        plot_title = plot_title.replace(":", "").replace(" ", "_")
        data_saver.safe_save(folder_path, f"BOXPLOT_{plot_title}.png")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(W, H)
        plt.savefig(os.path.join(folder_path, f"BOXPLOT_{plot_title}.png"), dpi=DPI)

    return


def plot(data: pd.DataFrame, data_col_order=[], data_name_order={}, raincloud=False, scatter=True,
         sub_line=None, plot_title="", plot_x_label="", plot_y_label="", colors_per_lab=1, color_list=None,
         horizontal_lines=None, save_plot=False, save_path="", save_name="", sub_folder="", custom_ylim=None,
         skip=None):
    """
    This function is called by external modules who wish to create a boxplot of the hit/miss/FA/CR data
    :param data: a dataframe which contains at least a lab column and at least one of the hit/miss/FA/CR columns
    :param check: whether this is a quality-check plot, or an analysis plot. Based on the answer, the plot will be
    saved in the hierarchy folder.
    :param data_col_order: the column names (as they appear in the df) of the to-be-plotted data
    :param data_name_order: the presentable names of the data_col_order columns. Dictionary containing a mapping between
    each column (as it appears in the df) and its "presentable" name
    :param raincloud: whether we want a regular boxplot (with or without scatter and lines between them), or a full-on
    raincloud plot
    :param scatter: whether to add a scatter plot on top of the box plot
    :param sub_line: whether to add a line connecting the scatterplot dots: None if no lines are to be plotted.
    Else: sub_line is a list of lists, such that each item in sub_line is a list of x axis points to connect between.
    For example, sub_line = [[0, 1], [2, 3]] will cause dots on x=0 to be connected to their corresponding dots in x=1,
    and dots on x=2 to be connected to their corresponding dots in x=3 - so each subject will have 4 dots, with 3 lines
    connecting them. sub_line = [[0, 1, 2, 3]] will cause each subject to have 3 connective lines.
    :param plot_title: the specific title of the boxplot
    :param plot_x_label: the label of the X-axis in the plot
    :param plot_y_label: the label of the Y-axis in the plot
    :param colors_per_lab : whether to color the scatter plot with different colors per LAB (1 color per lab) or per
    subject (1 color per subject). 0 = color per SUBJECT, 1 = coloer per lab
    :param save_plot: whether to save this plot
    :param save_path: the name of the highest folder in the BIDS-compatible file hierarchy
    :param ga_folder: if check=False, we have several options within the "general_analysis" folder-hierarchy in which
    the plot can be saved, so we could enter the path we want.
    """
    plt.clf()
    plt.figure()
    sns.reset_orig()
    # check that we actually have data to plot:
    nans = 0
    for col in data_col_order:
        col_data = data[col]
        if col_data.isnull().all():
            nans += 1
    if nans == len(data_col_order):
        print(f"Columns {data_col_order} are empty (None), cannot generate plot")
        return

    if not raincloud:
        if color_list is not None:
            colors = color_list
        else:
            all_lab_names = data['Lab'].unique()  # all lab codes that exist in the analyzed data
            all_sub_names = data['Subject'].unique()  # all sub codes that exist in the analyzed data
            color_source = all_lab_names if colors_per_lab == 1 else all_sub_names
            colors = sns.color_palette("colorblind", len(color_source))  # a unique color per lab/sub
        plotter(data, all_lab_names, all_sub_names, colors, colors_per_lab, data_col_order, data_name_order, scatter, sub_line,
                plot_title,
                plot_x_label, plot_y_label, save_plot, save_path, ga_folder=sub_folder)
    else:
        if len(save_name) == 0:
            plot_savename = plot_title.replace(":", "_").replace(r"\(.*\)", "")
        else:
            plot_savename = save_name.replace(":", "").replace(" ", "_").replace(r"\(.*\)", "")
        if color_list is not None:
            colors = color_list
        else:
            colors = sns.color_palette("colorblind", len(data_col_order))
        make_it_rain(data, cols=data_col_order, col_names=[data_name_order[i] for i in data_col_order],
                     color_list=colors, sub_line=sub_line,
                     plot_title=plot_title, plot_x_label=plot_x_label, plot_y_label=plot_y_label, save_plot=save_plot,
                     save_path=save_path, save_folder=sub_folder, horizontal_lines=horizontal_lines,
                     save_name=f"{plot_savename}", custom_ylim=custom_ylim, skip=skip)
    plt.clf()
    plt.close()
    gc.collect()
    return

